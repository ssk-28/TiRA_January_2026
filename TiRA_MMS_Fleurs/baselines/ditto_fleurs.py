#!/usr/bin/env python3
# ditto_lora_fl_fleurs.py
#
# Ditto+LoRA on ML-SUPERB FLEURS (one language = one client).
#
# Core Ditto mechanism (Algorithm 1):
# - Server maintains global model w^t (here: LoRA params only).
# - Each client k maintains a persistent personalized model v_k^t (LoRA params only), never sent to server.
# - Each round:
#   Phase A (global): client starts from w^t, trains to get w_k^{t+1}, sends for aggregation.
#   Phase B (personal): client updates v_k by minimizing F_k(v_k) + (lambda/2)||v_k - w^t||^2.
#
# LoRA:
# - Backbone frozen, only LoRA matrices trainable.
#
import os
import re
import csv
import time
import math
import json
import random
import argparse
import hashlib
import unicodedata
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

import torchaudio
from transformers import AutoProcessor, Wav2Vec2ForCTC

import splits as splits_mod


_PARENS = re.compile(r"\([^)]*\)")
_BAD = re.compile(r"[^\w\s'’]+", flags=re.UNICODE)
_WS = re.compile(r"\s+")


# -----------------------------
# MMS target lang map
# -----------------------------
def resolve_mms_target_lang(tokenizer, lang_id: str) -> Optional[str]:
    opts = set(tokenizer.vocab.keys())
    if lang_id in opts:
        return lang_id
    pref = [k for k in opts if k.startswith(lang_id + "-")]
    if len(pref) == 1:
        return pref[0]
    if len(pref) > 1:
        return sorted(pref)[0]
    alias = {
        "aze": "azj-script_latin",
        "cmn": "cmn-script_simplified",
        "yue": "yue-script_traditional",
        "srp": "srp-script_latin",
        "urd": "urd-script_arabic",
        "uzb": "uzb-script_latin",
        "nep": "npi",
        "fil": "tgl",
        "msa": "zlm",
        "swa": "swh",
        "ori": "ory",
    }
    if lang_id in alias and alias[lang_id] in opts:
        return alias[lang_id]
    return None


def filter_langs_supported_by_mms(
    model_id: str, cache_dir: Optional[str], data_langs: List[str]
) -> Tuple[List[str], Dict[str, str]]:
    proc = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    kept = []
    mapping: Dict[str, str] = {}
    for l in data_langs:
        t = resolve_mms_target_lang(proc.tokenizer, l)
        if t is None:
            print(f"[drop] lang {l}: no MMS target_lang mapping")
            continue
        kept.append(l)
        mapping[l] = t
    return kept, mapping


# -----------------------------
# Text normalization + CER/WER
# -----------------------------
def normalize_transcript(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _PARENS.sub(" ", s)
    s = _BAD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    ref = ref.strip()
    hyp = hyp.strip()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(hyp, ref) / float(len(ref))


def _edit_counts(ref_tokens, hyp_tokens):
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = "C"
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                bt[i][j] = "S" if best == sub else ("D" if best == dele else "I")

    i, j = n, m
    S = D = I = C = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "C":
            C += 1
            i -= 1
            j -= 1
        elif op == "S":
            S += 1
            i -= 1
            j -= 1
        elif op == "D":
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1

    N = max(1, n)
    return S, D, I, C, N


def wer_and_acc(ref_text: str, hyp_text: str):
    ref_tokens = ref_text.strip().split()
    hyp_tokens = hyp_text.strip().split()
    S, D, I, C, N = _edit_counts(ref_tokens, hyp_tokens)
    wer_v = (S + D + I) / N
    word_correct_acc = C / N
    sent_acc = 1.0 if (S + D + I) == 0 else 0.0
    return wer_v, word_correct_acc, sent_acc, (S, D, I, C, N)


def pctl(values: List[float], q: float) -> float:
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return float("nan")
    xs.sort()
    idx = int(round((q / 100.0) * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def tail_mean(values: List[float], tail_frac: float, higher_is_worse: bool) -> float:
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return float("nan")
    xs.sort()
    k = max(1, int(math.ceil(tail_frac * len(xs))))
    tail = xs[-k:] if higher_is_worse else xs[:k]
    return float(sum(tail) / len(tail))


def bytes_of_parameters(p: fl.common.Parameters) -> int:
    if p is None or getattr(p, "tensors", None) is None:
        return 0
    return int(sum(len(t) for t in p.tensors))


def l2_delta(nds_a: List[np.ndarray], nds_b: List[np.ndarray]) -> float:
    if nds_a is None or nds_b is None:
        return float("nan")
    if len(nds_a) != len(nds_b):
        return float("nan")
    tot = 0.0
    for a, b in zip(nds_a, nds_b):
        d = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
        tot += float(np.sum(d * d))
    return float(math.sqrt(tot))


# -----------------------------
# LoRA modules
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else None

        for p in self.base.parameters():
            p.requires_grad = False

        dev = base.weight.device
        dt = base.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros((self.r, self.in_features), device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.r), device=dev, dtype=dt))

        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        x_in = self.dropout(x) if self.dropout is not None else x

        if x_in.dim() == 3:
            xA = torch.matmul(x_in, self.lora_A.t())
            dY = torch.matmul(xA, self.lora_B.t()) * self.scaling
            return y0 + dY

        if x_in.dim() == 2:
            xA = torch.matmul(x_in, self.lora_A.t())
            dY = torch.matmul(xA, self.lora_B.t()) * self.scaling
            return y0 + dY

        raise ValueError(f"Unsupported x.dim={x_in.dim()} for LoRALinear")


def _get_parent_and_attr(root: nn.Module, qualname: str):
    parts = qualname.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets=("q_proj", "k_proj", "v_proj", "out_proj"),
    require_square: bool = True,
) -> int:
    # Collect first, then replace, to avoid mutating while iterating
    to_replace: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        last = name.split(".")[-1]
        if last not in set(targets):
            continue
        if require_square and module.in_features != module.out_features:
            continue
        if isinstance(module, LoRALinear):
            continue
        to_replace.append((name, module))

    for name, lin in to_replace:
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, LoRALinear(lin, r=r, alpha=alpha, dropout=dropout))

    return len(to_replace)




# -----------------------------
# Data loading
# -----------------------------
def _ft_bucket(lang_id: str, audio_path: str, uid: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}|{lang_id}|{audio_path}|{uid}".encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 1_000_000) / 1_000_000.0


def _load_audio_16k(path: str, max_audio_s: float) -> Tuple[np.ndarray, int]:
    wav, sr = torchaudio.load(path)
    if wav.numel() == 0:
        raise ValueError("empty audio")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000
    max_len = int(max_audio_s * sr)
    if wav.numel() > max_len:
        wav = wav[:max_len]
    return wav.cpu().numpy(), sr


class LocalFleursIterable(IterableDataset):
    def __init__(
        self,
        lang_id: str,
        lang_dir: str,
        split: str,
        ft_frac: float,
        want_ft: bool,
        max_hours: float,
        max_audio_s: float,
        seed: int,
    ):
        super().__init__()
        self.lang_id = lang_id
        self.lang_dir = lang_dir
        self.split = split
        self.ft_frac = float(ft_frac)
        self.want_ft = bool(want_ft)
        self.max_hours = float(max_hours)
        self.max_audio_s = float(max_audio_s)
        self.seed = int(seed)

        csv_path = os.path.join(lang_dir, f"{split}.csv")
        if split == "validation":
            csv_path = os.path.join(lang_dir, "validation.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing split csv: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw = [r for r in reader]

        rows = []
        seen = set()
        for r in raw:
            p = (r.get("path") or "").strip()
            tx = normalize_transcript(r.get("text") or "")
            uid = (r.get("uid") or os.path.splitext(os.path.basename(p))[0]).strip()
            if not p or not tx:
                continue
            if not os.path.isabs(p):
                p = os.path.join(lang_dir, p)
            if not os.path.isfile(p):
                continue
            if p in seen:
                continue
            seen.add(p)
            rows.append({"path": p, "text": tx, "uid": uid})

        rng = random.Random(self.seed + (hash(split) % 100000))
        rng.shuffle(rows)

        def is_ft_row(rr):
            return _ft_bucket(self.lang_id, rr["path"], rr["uid"], self.seed) < self.ft_frac

        if self.want_ft:
            filtered = [rr for rr in rows if is_ft_row(rr)]
            if len(filtered) == 0:
                print(f"[warn] {self.lang_id} split={self.split}: ft bucket empty, using full split.")
                filtered = rows
        else:
            filtered = [rr for rr in rows if not is_ft_row(rr)]

        if len(filtered) == 0:
            raise RuntimeError(f"{self.lang_id} split={self.split}: no usable rows")

        self.rows = filtered

    def __iter__(self):
        total_sec = 0.0
        max_sec = self.max_hours * 3600.0 if self.max_hours and self.max_hours > 0 else float("inf")
        for r in self.rows:
            try:
                arr, sr = _load_audio_16k(r["path"], self.max_audio_s)
            except Exception:
                continue
            dur = float(arr.shape[0]) / float(sr)
            total_sec += dur
            yield {"audio": arr, "sr": sr, "text": r["text"]}
            if total_sec >= max_sec:
                break


class CTCCollator:
    def __init__(self, processor, pad_to_multiple_of: Optional[int] = None):
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        audios = [b["audio"] for b in batch]
        srs = [b["sr"] for b in batch]
        if len(set(srs)) != 1:
            raise ValueError("mixed sampling rates in batch")
        texts = [b["text"] for b in batch]

        inputs = self.processor(
            audios,
            sampling_rate=srs[0],
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        lab = self.processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        lab[lab == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = lab
        return inputs


# -----------------------------
# Trainable weights helpers
# -----------------------------
def trainable_named_parameters(module: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    pairs = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    pairs.sort(key=lambda x: x[0])
    return pairs


def get_trainable_weights(module: nn.Module) -> Tuple[List[str], List[np.ndarray]]:
    items = trainable_named_parameters(module)
    names = [n for n, _ in items]
    arrays = [p.detach().cpu().numpy().copy() for _, p in items]
    return names, arrays


def set_trainable_weights(module: nn.Module, names: List[str], arrays: List[np.ndarray]) -> None:
    name_to_param = {n: p for n, p in trainable_named_parameters(module)}
    for n, a in zip(names, arrays):
        p = name_to_param[n]
        t = torch.from_numpy(a).to(p.device)
        if p.data.shape != t.shape:
            raise ValueError(f"Shape mismatch for {n}: {p.data.shape} vs {t.shape}")
        p.data.copy_(t)


def copy_arrays(arrs: List[np.ndarray]) -> List[np.ndarray]:
    return [a.copy() for a in arrs]


# -----------------------------
# Flower client: Ditto+LoRA
# -----------------------------
class LocalFleursDittoClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        lang_id: str,
        lang_dir: str,
        lang_idx: int,
        all_langs: List[str],
        model_id: str,
        device: str,
        fp16: bool,
        use_mms_adapters: bool,
        batch_size: int,
        global_steps: int,
        personal_steps: int,
        lr_global: float,
        lr_personal: float,
        weight_decay: float,
        max_grad_norm: float,
        ft_frac: float,
        max_train_hours: float,
        max_valid_hours: float,
        max_audio_s: float,
        eval_utterances: int,
        trainable_names: List[str],
        seed: int,
        cache_dir: Optional[str],
        mms_target_lang: Optional[str],
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        ditto_lambda: float,
    ):
        self.cid = cid
        self.lang_id = lang_id
        self.lang_dir = lang_dir
        self.lang_idx = int(lang_idx)
        self.all_langs = all_langs
        self.model_id = model_id
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self.use_mms_adapters = bool(use_mms_adapters)
        self.batch_size = int(batch_size)
        self.global_steps = int(global_steps)
        self.personal_steps = int(personal_steps)
        self.lr_global = float(lr_global)
        self.lr_personal = float(lr_personal)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.ft_frac = float(ft_frac)
        self.max_train_hours = float(max_train_hours)
        self.max_valid_hours = float(max_valid_hours)
        self.max_audio_s = float(max_audio_s)
        self.eval_utterances = int(eval_utterances)
        self.trainable_names = trainable_names
        self.seed = int(seed)
        self.cache_dir = cache_dir
        self.mms_target_lang = mms_target_lang

        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.ditto_lambda = float(ditto_lambda)

        # Persistent personalized state v_k (LoRA params only)
        self.personal_arrays: Optional[List[np.ndarray]] = None

        self._init()

    def _init(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        inject_lora(self.model, r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout)
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                print(n, p.device, p.dtype)
                break

        self.model = self.model.to(self.device)
        if self.use_mms_adapters:
            if not self.mms_target_lang:
                raise ValueError(f"Missing MMS target lang for {self.lang_id}")
            self.processor.tokenizer.set_target_lang(self.mms_target_lang)
            self.model.load_adapter(self.mms_target_lang)

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # Inject LoRA and unfreeze only LoRA params
        inject_lora(self.model, r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout)
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        self.opt_global = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr_global,
            weight_decay=self.weight_decay,
        )
        self.opt_personal = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr_personal,
            weight_decay=self.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.fp16 and self.device.type == "cuda"))

        train_ds = LocalFleursIterable(
            lang_id=self.lang_id,
            lang_dir=self.lang_dir,
            split="train",
            ft_frac=self.ft_frac,
            want_ft=True,
            max_hours=self.max_train_hours,
            max_audio_s=self.max_audio_s,
            seed=self.seed,
        )
        valid_ds = LocalFleursIterable(
            lang_id=self.lang_id,
            lang_dir=self.lang_dir,
            split="validation",
            ft_frac=1.0,
            want_ft=True,
            max_hours=self.max_valid_hours,
            max_audio_s=self.max_audio_s,
            seed=self.seed + 17,
        )

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, collate_fn=CTCCollator(self.processor))
        self.valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, collate_fn=CTCCollator(self.processor))

    def get_parameters(self, config):
        _, arrays = get_trainable_weights(self.model)
        return arrays

    def set_parameters(self, parameters):
        set_trainable_weights(self.model, self.trainable_names, parameters)

    def _prox_mean_square(self, anchor_arrays: List[np.ndarray]) -> torch.Tensor:
        # Returns mean squared L2 distance across all trainable params
        items = trainable_named_parameters(self.model)
        if not items:
            return torch.zeros((), device=self.device)
        sq = torch.zeros((), device=self.device)
        ne = 0
        for (_, p), a in zip(items, anchor_arrays):
            at = torch.from_numpy(a).to(p.device, dtype=p.dtype)
            d = p - at
            sq = sq + (d * d).sum()
            ne += d.numel()
        denom = max(1, int(ne))
        return sq / float(denom)

    def fit(self, parameters, config):
        # Receive global model w^t (LoRA params)
        self.set_parameters(parameters)
        global_anchor = copy_arrays(parameters)  # This is w^t, the proximal anchor in Ditto

        # Initialize v_k on first participation
        if self.personal_arrays is None:
            self.personal_arrays = copy_arrays(global_anchor)

        self.model.train()
        t0 = time.time()

        # -----------------
        # Phase A: global update starting from w^t
        # -----------------
        g_step = 0
        g_total = 0.0
        seen_examples = 0

        for batch in self.train_loader:
            if self.global_steps > 0 and g_step >= self.global_steps:
                break
            bs = int(batch["input_values"].shape[0])
            seen_examples += bs

            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.opt_global.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.fp16 and self.device.type == "cuda")):
                out = self.model(**batch)
                loss = out.loss

            self.scaler.scale(loss).backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                self.scaler.unscale_(self.opt_global)
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm)

            self.scaler.step(self.opt_global)
            self.scaler.update()

            g_total += float(loss.detach().item())
            g_step += 1

        new_global = self.get_parameters(config={})
        g_avg = g_total / max(1, g_step) if g_step > 0 else float("nan")

        # -----------------
        # Phase B: personalized update on v_k with proximal to w^t
        # -----------------
        # Load v_k into model
        self.set_parameters(self.personal_arrays)

        p_step = 0
        p_total_ctc = 0.0
        p_total_prox = 0.0
        lam = float(self.ditto_lambda)

        for batch in self.train_loader:
            if self.personal_steps > 0 and p_step >= self.personal_steps:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.opt_personal.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.fp16 and self.device.type == "cuda")):
                out = self.model(**batch)
                loss_ctc = out.loss
                prox = self._prox_mean_square(global_anchor)
                loss = loss_ctc + (0.5 * lam * prox)

            self.scaler.scale(loss).backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                self.scaler.unscale_(self.opt_personal)
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm)

            self.scaler.step(self.opt_personal)
            self.scaler.update()

            p_total_ctc += float(loss_ctc.detach().item())
            p_total_prox += float(prox.detach().item())
            p_step += 1

        # Save updated v_k back to persistent client state
        self.personal_arrays = self.get_parameters(config={})

        dt = time.time() - t0
        num_ex = max(1, int(seen_examples))

        metrics = {
            "lang_id": self.lang_id,
            "lang_idx": int(self.lang_idx),
            "train_steps_global": int(g_step),
            "train_steps_personal": int(p_step),
            "train_loss_global": float(g_avg),
            "train_loss_personal": float(p_total_ctc / max(1, p_step) if p_step > 0 else float("nan")),
            "train_prox_mean": float(p_total_prox / max(1, p_step) if p_step > 0 else float("nan")),
            "ditto_lambda": float(self.ditto_lambda),
            "fit_time_s": float(dt),
            "train_examples": int(seen_examples),
        }
        # Return only updated global model for aggregation, never return v_k
        return new_global, num_ex, metrics

    def evaluate(self, parameters, config):
        # Sync global w^t (not used directly for eval, but keeps client aligned)
        self.set_parameters(parameters)
        if self.personal_arrays is None:
            self.personal_arrays = copy_arrays(parameters)

        # Evaluate personalized v_k
        self.set_parameters(self.personal_arrays)
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        cer_list: List[float] = []
        tot_S = tot_D = tot_I = tot_C = tot_N = 0
        sent_ok = 0
        num_utts = 0
        seen_utts = 0

        with torch.inference_mode():
            for batch in self.valid_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                total_loss += float(loss.detach().item())
                n_batches += 1

                logits = out.logits
                pred_ids = torch.argmax(logits, dim=-1)

                hyps = self.processor.batch_decode(pred_ids)
                labels = batch["labels"].detach().cpu().numpy()
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                refs = self.processor.batch_decode(labels, group_tokens=False)

                for hyp_raw, ref_raw in zip(hyps, refs):
                    hyp = normalize_transcript(hyp_raw.replace("|", " "))
                    ref = normalize_transcript(ref_raw.replace("|", " "))

                    cer_list.append(cer(hyp, ref))

                    w, wacc, sacc, (S, D, I, C, N) = wer_and_acc(ref, hyp)
                    tot_S += S
                    tot_D += D
                    tot_I += I
                    tot_C += C
                    tot_N += N
                    sent_ok += int(sacc)
                    num_utts += 1

                    seen_utts += 1
                    if self.eval_utterances > 0 and seen_utts >= self.eval_utterances:
                        break

                if self.eval_utterances > 0 and seen_utts >= self.eval_utterances:
                    break

        avg_loss = total_loss / max(1, n_batches)
        mean_cer = float(sum(cer_list) / max(1, len(cer_list)))
        max_cer = float(max(cer_list)) if cer_list else float("nan")

        corpus_wer = (tot_S + tot_D + tot_I) / max(1, tot_N)
        corpus_word_correct_acc = tot_C / max(1, tot_N)
        corpus_sent_acc = sent_ok / max(1, num_utts)

        metrics = {
            "lang_id": self.lang_id,
            "lang_idx": int(self.lang_idx),
            "val_loss": float(avg_loss),
            "val_cer_mean": float(mean_cer),
            "val_cer_max": float(max_cer),
            "val_wer": float(corpus_wer),
            "val_word_correct_acc": float(corpus_word_correct_acc),
            "val_sent_acc": float(corpus_sent_acc),
            "eval_utts": int(seen_utts),
            "eval_word_tokens": int(tot_N),
            "eval_personalized": 1,
        }
        return float(avg_loss), int(seen_utts), metrics


# -----------------------------
# Strategy: FedAvg + logging
# -----------------------------
class DittoLoggingFedAvg(FedAvg):
    def __init__(
        self,
        log_csv: str,
        tail_fracs: List[float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.log_csv = log_csv
        self.tail_fracs = [float(x) for x in tail_fracs]

        self._round_start_ts: Optional[float] = None
        self._clients_fit_configured_this_round: int = 0
        self._payload_bytes_this_round: float = float("nan")
        self._sent_ndarrays_this_round: Optional[List[np.ndarray]] = None
        self._fit_round_stats: Dict[int, Dict[str, float]] = {}

        os.makedirs(os.path.dirname(self.log_csv) or ".", exist_ok=True)

        base_fields = [
            "round",
            "wall_time_s",
            "clients_fit",
            "fit_mean_train_loss_global",
            "fit_p95_train_loss_global",
            "fit_mean_train_loss_personal",
            "fit_p95_train_loss_personal",
            "fit_mean_prox",
            "fit_mean_delta_l2",
            "fit_max_delta_l2",
            "payload_bytes",
            "round_upload_bytes",
            "round_download_bytes",
            "clients_eval",
            "mean_loss",
            "mean_cer",
            "max_cer",
            "p95_cer",
            "mean_wer",
            "max_wer",
            "p95_wer",
            "mean_word_correct_acc",
            "min_word_correct_acc",
            "p05_word_correct_acc",
            "mean_sent_acc",
            "min_sent_acc",
            "p05_sent_acc",
        ]

        tail_fields = []
        for tf in self.tail_fracs:
            p = int(round(tf * 100))
            tail_fields += [
                f"tail{p}_cer",
                f"tail{p}_wer",
                f"tail{p}_word_correct_acc",
                f"tail{p}_sent_acc",
            ]

        self._csv_fields = base_fields + tail_fields

        if not os.path.exists(self.log_csv):
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start_ts = time.time()
        cfgs = super().configure_fit(server_round, parameters, client_manager)

        self._clients_fit_configured_this_round = int(len(cfgs))
        if len(cfgs) > 0:
            fit_ins0 = cfgs[0][1]
            self._payload_bytes_this_round = float(bytes_of_parameters(fit_ins0.parameters))
            try:
                self._sent_ndarrays_this_round = parameters_to_ndarrays(fit_ins0.parameters)
            except Exception:
                self._sent_ndarrays_this_round = None
        else:
            self._payload_bytes_this_round = float("nan")
            self._sent_ndarrays_this_round = None

        return cfgs

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print(f"[warn] Round {server_round}: no fit results")
            return None

        clients_fit = int(len(results))
        g_losses: List[float] = []
        p_losses: List[float] = []
        prox_vals: List[float] = []
        delta_l2s: List[float] = []

        sent_nds = self._sent_ndarrays_this_round
        payload_bytes = float(self._payload_bytes_this_round) if self._payload_bytes_this_round == self._payload_bytes_this_round else float("nan")

        for _, fit_res in results:
            m = fit_res.metrics or {}
            if "train_loss_global" in m:
                g_losses.append(float(m["train_loss_global"]))
            if "train_loss_personal" in m:
                p_losses.append(float(m["train_loss_personal"]))
            if "train_prox_mean" in m:
                prox_vals.append(float(m["train_prox_mean"]))

            if sent_nds is not None:
                try:
                    client_nds = parameters_to_ndarrays(fit_res.parameters)
                    d = l2_delta(client_nds, sent_nds)
                    if not math.isnan(d):
                        delta_l2s.append(float(d))
                except Exception:
                    pass

        fit_mean_train_loss_global = float(np.mean(g_losses)) if g_losses else float("nan")
        fit_p95_train_loss_global = float(np.percentile(g_losses, 95)) if g_losses else float("nan")
        fit_mean_train_loss_personal = float(np.mean(p_losses)) if p_losses else float("nan")
        fit_p95_train_loss_personal = float(np.percentile(p_losses, 95)) if p_losses else float("nan")
        fit_mean_prox = float(np.mean(prox_vals)) if prox_vals else float("nan")

        fit_mean_delta_l2 = float(np.mean(delta_l2s)) if delta_l2s else float("nan")
        fit_max_delta_l2 = float(np.max(delta_l2s)) if delta_l2s else float("nan")

        if payload_bytes == payload_bytes:
            round_download_bytes = float(payload_bytes * float(self._clients_fit_configured_this_round))
            round_upload_bytes = float(payload_bytes * float(clients_fit))
        else:
            round_download_bytes = float("nan")
            round_upload_bytes = float("nan")

        self._fit_round_stats[int(server_round)] = {
            "clients_fit": float(clients_fit),
            "fit_mean_train_loss_global": float(fit_mean_train_loss_global),
            "fit_p95_train_loss_global": float(fit_p95_train_loss_global),
            "fit_mean_train_loss_personal": float(fit_mean_train_loss_personal),
            "fit_p95_train_loss_personal": float(fit_p95_train_loss_personal),
            "fit_mean_prox": float(fit_mean_prox),
            "fit_mean_delta_l2": float(fit_mean_delta_l2),
            "fit_max_delta_l2": float(fit_max_delta_l2),
            "payload_bytes": float(payload_bytes),
            "round_upload_bytes": float(round_upload_bytes),
            "round_download_bytes": float(round_download_bytes),
        }

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        wall_s = float(time.time() - self._round_start_ts) if self._round_start_ts is not None else float("nan")
        fit_stats = self._fit_round_stats.get(int(server_round), {})

        if not results:
            row = {k: float("nan") for k in self._csv_fields}
            row["round"] = int(server_round)
            row["wall_time_s"] = wall_s
            for k, v in fit_stats.items():
                row[k] = float(v)
            row["clients_eval"] = 0
            with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)
            return float("nan"), row

        cer_by_client = []
        wer_by_client = []
        wacc_by_client = []
        sacc_by_client = []
        loss_by_client = []

        for _, eval_res in results:
            loss_by_client.append(float(eval_res.loss))
            m = eval_res.metrics or {}
            if "val_cer_mean" in m:
                cer_by_client.append(float(m["val_cer_mean"]))
            if "val_wer" in m:
                wer_by_client.append(float(m["val_wer"]))
            if "val_word_correct_acc" in m:
                wacc_by_client.append(float(m["val_word_correct_acc"]))
            if "val_sent_acc" in m:
                sacc_by_client.append(float(m["val_sent_acc"]))

        mean_loss = float(sum(loss_by_client) / max(1, len(loss_by_client))) if loss_by_client else float("nan")

        mean_cer = float(sum(cer_by_client) / max(1, len(cer_by_client))) if cer_by_client else float("nan")
        max_cer = float(max(cer_by_client)) if cer_by_client else float("nan")
        p95_cer = pctl(cer_by_client, 95.0)

        mean_wer = float(sum(wer_by_client) / max(1, len(wer_by_client))) if wer_by_client else float("nan")
        max_wer = float(max(wer_by_client)) if wer_by_client else float("nan")
        p95_wer = pctl(wer_by_client, 95.0)

        mean_wacc = float(sum(wacc_by_client) / max(1, len(wacc_by_client))) if wacc_by_client else float("nan")
        min_wacc = float(min(wacc_by_client)) if wacc_by_client else float("nan")
        p05_wacc = pctl(wacc_by_client, 5.0)

        mean_sacc = float(sum(sacc_by_client) / max(1, len(sacc_by_client))) if sacc_by_client else float("nan")
        min_sacc = float(min(sacc_by_client)) if sacc_by_client else float("nan")
        p05_sacc = pctl(sacc_by_client, 5.0)

        row: Dict[str, float] = {
            "round": int(server_round),
            "wall_time_s": wall_s,
            "clients_eval": int(len(cer_by_client) if cer_by_client else len(results)),
            "mean_loss": mean_loss,
            "mean_cer": mean_cer,
            "max_cer": max_cer,
            "p95_cer": p95_cer,
            "mean_wer": mean_wer,
            "max_wer": max_wer,
            "p95_wer": p95_wer,
            "mean_word_correct_acc": mean_wacc,
            "min_word_correct_acc": min_wacc,
            "p05_word_correct_acc": p05_wacc,
            "mean_sent_acc": mean_sacc,
            "min_sent_acc": min_sacc,
            "p05_sent_acc": p05_sacc,
        }

        for k, v in fit_stats.items():
            row[k] = float(v)

        for tf in self.tail_fracs:
            p = int(round(tf * 100))
            row[f"tail{p}_cer"] = tail_mean(cer_by_client, tf, higher_is_worse=True)
            row[f"tail{p}_wer"] = tail_mean(wer_by_client, tf, higher_is_worse=True)
            row[f"tail{p}_word_correct_acc"] = tail_mean(wacc_by_client, tf, higher_is_worse=False)
            row[f"tail{p}_sent_acc"] = tail_mean(sacc_by_client, tf, higher_is_worse=False)

        print(
            f"[Round {server_round}] meanCER={mean_cer:.4f} maxCER={max_cer:.4f} "
            f"tail10CER={row.get('tail10_cer', float('nan')):.4f}"
        )

        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)

        return mean_loss, row


# -----------------------------
# Run one experiment
# -----------------------------
def suffix_log_csv(base_csv: str, ft_frac_value: float) -> str:
    root, ext = os.path.splitext(base_csv)
    p = int(round(ft_frac_value * 100))
    return f"{root}_ft{p:02d}{ext if ext else '.csv'}"


def run_one(args, ft_frac_value: float, log_csv_path: str) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root not found: {args.data_root}")

    exclude = {x.strip() for x in args.exclude_langs.split(",") if x.strip()}

    if args.no_fp16:
        fp16 = False
    elif args.fp16:
        fp16 = True
    else:
        fp16 = args.device.startswith("cuda")

    if args.no_mms_adapters:
        use_mms_adapters = False
    elif args.use_mms_adapters:
        use_mms_adapters = True
    else:
        use_mms_adapters = True

    if args.num_clients is not None:
        n_clients = int(args.num_clients)
    else:
        n_clients = int(args.num_langs)

    if args.langs.strip():
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
        if n_clients > 0:
            langs = langs[:n_clients]
    else:
        discovered = splits_mod.discover_langs(args.data_root, exclude=exclude)
        langs = discovered[: max(1, n_clients)] if n_clients > 0 else discovered

    langs = [l for l in langs if l not in exclude]
    if not langs:
        raise RuntimeError("No languages selected")

    ok_langs = splits_mod.build_manifests_for_langs(
        data_root=args.data_root,
        langs=langs,
        force_rebuild=args.force_rebuild_manifests,
        fail_fast=args.fail_fast,
    )

    ok_langs, lang_to_mms = filter_langs_supported_by_mms(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        data_langs=ok_langs,
    )
    if not ok_langs:
        raise RuntimeError("No usable languages after MMS mapping filter")

    if args.write_langs_json:
        with open(args.write_langs_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_root": args.data_root,
                    "langs": ok_langs,
                    "num_clients": len(ok_langs),
                    "ft_frac": float(ft_frac_value),
                    "method": "ditto+lora",
                    "ditto_lambda": float(args.ditto_lambda),
                    "global_steps": int(args.global_steps),
                    "personal_steps": int(args.personal_steps),
                },
                f,
                indent=2,
            )

    if args.build_manifests_only:
        return

    all_langs = ok_langs

    clients_per_round = int(args.clients_per_round)
    if clients_per_round > len(all_langs):
        print(f"[warn] clients_per_round={clients_per_round} > num_clients={len(all_langs)}; clamping.")
        clients_per_round = len(all_langs)

    eval_utterances = int(args.eval_utterances)
    global_steps = int(args.global_steps)
    personal_steps = int(args.personal_steps)
    max_train_hours = float(args.max_train_hours)
    max_valid_hours = float(args.max_valid_hours)
    ft_frac_use = float(ft_frac_value)

    if args.unbounded_budget_baseline:
        clients_per_round = len(all_langs)
        ft_frac_use = 1.0
        max_train_hours = 0.0
        max_valid_hours = 0.0
        eval_utterances = 0
        global_steps = 0
        personal_steps = 0

    # Initialize global LoRA params (w^0) from a freshly injected model
    tmp = Wav2Vec2ForCTC.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    for p in tmp.parameters():
        p.requires_grad = False
    inject_lora(tmp, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    for n, p in tmp.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
    trainable_names, init_arrays = get_trainable_weights(tmp)
    init_parameters = ndarrays_to_parameters(init_arrays)

    def client_fn(cid: str):
        i = int(cid)
        lang_id = all_langs[i]
        mms_target = lang_to_mms.get(lang_id, None)
        return LocalFleursDittoClient(
            cid=cid,
            lang_id=lang_id,
            lang_dir=os.path.join(args.data_root, lang_id),
            lang_idx=i,
            all_langs=all_langs,
            model_id=args.model_id,
            device=args.device,
            fp16=fp16,
            use_mms_adapters=use_mms_adapters,
            batch_size=args.batch_size,
            global_steps=global_steps,
            personal_steps=personal_steps,
            lr_global=args.lr_global,
            lr_personal=args.lr_personal,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            ft_frac=ft_frac_use,
            max_train_hours=max_train_hours,
            max_valid_hours=max_valid_hours,
            max_audio_s=args.max_audio_s,
            eval_utterances=eval_utterances,
            trainable_names=trainable_names,
            seed=args.seed,
            cache_dir=args.cache_dir,
            mms_target_lang=mms_target,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            ditto_lambda=args.ditto_lambda,
        )

    tail_fracs = [float(x.strip()) for x in args.tail_fracs.split(",") if x.strip()]
    if not tail_fracs:
        tail_fracs = [0.10, 0.20, 0.30]

    strategy = DittoLoggingFedAvg(
        log_csv=log_csv_path,
        tail_fracs=tail_fracs,
        fraction_fit=min(1.0, clients_per_round / max(1, len(all_langs))),
        fraction_evaluate=1.0,
        min_fit_clients=min(clients_per_round, len(all_langs)),
        min_available_clients=len(all_langs),
        min_evaluate_clients=len(all_langs),
        initial_parameters=init_parameters,
    )

    client_resources = {"num_cpus": float(args.num_cpus_per_client), "num_gpus": float(args.num_gpus_per_client)}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(all_langs),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"include_dashboard": False},
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="../ML_SUPERB/fleurs")
    ap.add_argument("--langs", type=str, default="")
    ap.add_argument("--num_langs", type=int, default=20)
    ap.add_argument("--num_clients", type=int, default=None)
    ap.add_argument("--exclude_langs", type=str, default="cmn,jpn,kor,zh_cn,ja_jp,ko_kr")

    ap.add_argument("--build_manifests_only", action="store_true")
    ap.add_argument("--force_rebuild_manifests", action="store_true")
    ap.add_argument("--fail_fast", action="store_true")
    ap.add_argument("--write_langs_json", type=str, default="langs_used.json")

    ap.add_argument("--model_id", type=str, default="facebook/mms-1b-fl102")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--use_mms_adapters", action="store_true")
    ap.add_argument("--no_mms_adapters", action="store_true")
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--clients_per_round", type=int, default=4)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ft_frac", type=float, default=0.30)
    ap.add_argument("--ft_fracs", type=str, default="")

    ap.add_argument("--max_train_hours", type=float, default=0.50)
    ap.add_argument("--max_valid_hours", type=float, default=0.10)
    ap.add_argument("--max_audio_s", type=float, default=16.0)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--global_steps", type=int, default=50)
    ap.add_argument("--personal_steps", type=int, default=50)

    ap.add_argument("--lr_global", type=float, default=5e-4)
    ap.add_argument("--lr_personal", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--eval_utterances", type=int, default=128)
    ap.add_argument("--log_csv", type=str, default="runs/ditto_lora_fleurs.csv")

    ap.add_argument("--num_cpus_per_client", type=float, default=2.0)
    ap.add_argument("--num_gpus_per_client", type=float, default=1.0 if torch.cuda.is_available() else 0.0)

    ap.add_argument("--unbounded_budget_baseline", action="store_true")
    ap.add_argument("--tail_fracs", type=str, default="0.10,0.20,0.30")

    # LoRA knobs
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    # Ditto knob
    ap.add_argument("--ditto_lambda", type=float, default=1.0)

    args = ap.parse_args()

    if args.ft_fracs.strip():
        ft_list = [float(x.strip()) for x in args.ft_fracs.split(",") if x.strip()]
        if not ft_list:
            ft_list = [float(args.ft_frac)]
    else:
        ft_list = [float(args.ft_frac)]

    for k, ftv in enumerate(ft_list):
        out_csv = args.log_csv if len(ft_list) == 1 else suffix_log_csv(args.log_csv, ftv)
        print(f"[run] method=ditto+lora ft_frac={ftv:.2f} log_csv={out_csv}")
        run_one(args, ft_frac_value=ftv, log_csv_path=out_csv)

        if k + 1 < len(ft_list):
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
