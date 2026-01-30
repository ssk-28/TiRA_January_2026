#!/usr/bin/env python3
# afl_lora_fl_fleurs.py
#
# AFL-LoRA: Agnostic Federated Learning with a single global LoRA (no HyperLoRA, no routing heads).
#
# Key changes vs tarp_hyperlora_fl_fleurs.py:
#   1) Remove HyperLoRA entirely, inject simple LoRA modules into attention projections.
#   2) Make AFL active: the server maintains client mixture weights (lambda),
#      uses them for (a) client sampling each round and (b) aggregation weights,
#      and updates them via exponentiated-gradient using per-client eval risk.
#
# References:
#   - Mohri, Sivek, Suresh. "Agnostic Federated Learning." ICML 2019 (PMLR 97).
#     https://proceedings.mlr.press/v97/mohri19a/mohri19a.pdf
#   - Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." 2021.
#     https://arxiv.org/abs/2106.09685
#
# Example:
#   python afl_lora_fl_fleurs.py --num_clients 20 --clients_per_round 6 --ft_frac 0.30
#   python afl_lora_fl_fleurs.py --num_clients 30 --clients_per_round 10 --afl_eta 0.5 --afl_risk_metric cer
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
from flwr.common import FitIns, EvaluateIns, ndarrays_to_parameters, parameters_to_ndarrays
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


# -----------------------------
# Stats helpers
# -----------------------------
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


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p.astype(np.float64), eps, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)))


def gini(p: np.ndarray, eps: float = 1e-12) -> float:
    x = np.clip(p.astype(np.float64), eps, None)
    x = x / np.sum(x)
    x = np.sort(x)
    n = x.size
    if n <= 1:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2 * idx - n - 1) * x)) / (n - 1))


# -----------------------------
# Simple LoRA modules
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear base")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(max(1, self.r))
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else None

        in_f = int(base.in_features)
        out_f = int(base.out_features)

        # IMPORTANT: create LoRA params on the same device (and dtype) as the base layer
        dev = base.weight.device
        dtype = base.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros(self.r, in_f, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, device=dev, dtype=dtype))

        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        x_in = self.drop(x) if self.drop is not None else x

        if x_in.dim() == 3:
            xa = torch.einsum("bti,ri->btr", x_in, self.lora_A)
            dy = torch.einsum("btr,or->bto", xa, self.lora_B) * self.scaling
            return y0 + dy
        if x_in.dim() == 2:
            xa = torch.einsum("bi,ri->br", x_in, self.lora_A)
            dy = torch.einsum("br,or->bo", xa, self.lora_B) * self.scaling
            return y0 + dy

        raise ValueError(f"Unsupported input dim for LoRALinear: {x_in.dim()}")


def inject_lora_into_wav2vec2_attn(
    model: nn.Module,
    targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    enc = model.wav2vec2.encoder
    replaced = 0
    for layer in enc.layers:
        attn = getattr(layer, "attention", None)
        if attn is None:
            continue
        for t in targets:
            lin = getattr(attn, t, None)
            if isinstance(lin, nn.Linear):
                lora = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
                # extra safety: ensure module is on same device as the base projection
                lora = lora.to(lin.weight.device)
                setattr(attn, t, lora)
                replaced += 1
    return replaced


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


def weighted_average_params(weighted_nds: List[Tuple[float, List[np.ndarray]]]) -> List[np.ndarray]:
    if not weighted_nds:
        raise ValueError("weighted_average_params: empty input")
    sw = float(sum(w for w, _ in weighted_nds))
    if sw <= 0:
        return [a.copy() for a in weighted_nds[0][1]]
    n_params = len(weighted_nds[0][1])
    out: List[np.ndarray] = []
    for j in range(n_params):
        acc = np.zeros_like(weighted_nds[0][1][j], dtype=np.float32)
        for w, nds in weighted_nds:
            acc += (float(w) / sw) * nds[j].astype(np.float32)
        out.append(acc)
    return out


# -----------------------------
# Flower client: simple LoRA
# -----------------------------
class LocalFleursAFLLoraClient(fl.client.NumPyClient):
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
        mms_target_lang: Optional[str],
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_targets: Tuple[str, ...],
        train_lm_head: bool,
        batch_size: int,
        local_steps: int,
        lr: float,
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
        self.mms_target_lang = mms_target_lang

        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_targets = tuple(lora_targets)

        self.train_lm_head = bool(train_lm_head)

        self.batch_size = int(batch_size)
        self.local_steps = int(local_steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)

        self.ft_frac = float(ft_frac)
        self.max_train_hours = float(max_train_hours)
        self.max_valid_hours = float(max_valid_hours)
        self.max_audio_s = float(max_audio_s)
        self.eval_utterances = int(eval_utterances)

        self.trainable_names = list(trainable_names)
        self.seed = int(seed)
        self.cache_dir = cache_dir

        self._init()

    def _init(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)

        # Optional MMS language adapters (kept frozen)
        if self.use_mms_adapters:
            if not self.mms_target_lang:
                raise ValueError(f"Missing MMS target lang for {self.lang_id}")
            self.processor.tokenizer.set_target_lang(self.mms_target_lang)
            self.model.load_adapter(self.mms_target_lang)

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # Inject LoRA and enable only LoRA params (and optionally lm_head)
        replaced = inject_lora_into_wav2vec2_attn(
            self.model,
            targets=self.lora_targets,
            r=self.lora_r,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        if replaced == 0:
            raise RuntimeError("No attention Linear layers were replaced by LoRA. Check lora_targets and model version.")
        if self.train_lm_head:
            for p in self.model.lm_head.parameters():
                p.requires_grad = True

        # Build optimizer on trainable params only
        trainables = [p for _, p in trainable_named_parameters(self.model)]
        if not trainables:
            raise RuntimeError("No trainable parameters found. LoRA injection failed.")
        self.optimizer = torch.optim.AdamW(trainables, lr=self.lr, weight_decay=self.weight_decay)
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

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        step = 0
        total_loss = 0.0
        seen_examples = 0
        t0 = time.time()

        for batch in self.train_loader:
            if self.local_steps > 0 and step >= self.local_steps:
                break

            bs = int(batch["input_values"].shape[0])
            seen_examples += bs

            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.fp16 and self.device.type == "cuda")):
                out = self.model(**batch)
                loss = out.loss

            self.scaler.scale(loss).backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainable_named_parameters(self.model)],
                    self.max_grad_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach().item())
            step += 1

        dt = time.time() - t0
        if step == 0:
            new_params = parameters
            num_ex = 1
            avg_loss = float("nan")
        else:
            avg_loss = total_loss / step
            new_params = self.get_parameters(config={})
            num_ex = max(1, int(seen_examples))

        metrics = {
            "lang_id": self.lang_id,
            "lang_idx": int(self.lang_idx),
            "train_steps": int(step),
            "train_loss": float(avg_loss),
            "fit_time_s": float(dt),
            "train_examples": int(seen_examples),
        }
        return new_params, num_ex, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

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

                    w, wacc_correct, sacc, (S, D, I, C, N) = wer_and_acc(ref, hyp)
                    tot_S += S
                    tot_D += D
                    tot_I += I
                    tot_C += C
                    tot_N += N
                    sent_ok += int(sacc)
                    num_utts += 1

                    seen_utts += 1
                    if self.eval_utterances and self.eval_utterances > 0 and seen_utts >= self.eval_utterances:
                        break

                if self.eval_utterances and self.eval_utterances > 0 and seen_utts >= self.eval_utterances:
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
        }
        return float(avg_loss), int(seen_utts), metrics


# -----------------------------
# Strategy: AFL active sampling + AFL weighted aggregation + logging
# -----------------------------
class AFLLoraStrategy(FedAvg):
    def __init__(
        self,
        num_clients_total: int,
        trainable_names: List[str],
        log_csv: str,
        tail_fracs: List[float],
        afl_eta: float,
        afl_mixing: float,
        afl_risk_metric: str,
        afl_warmup_rounds: int,
        afl_use_num_examples: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_clients_total = int(num_clients_total)
        self.trainable_names = list(trainable_names)
        self.log_csv = log_csv
        self.tail_fracs = [float(x) for x in tail_fracs]

        self.afl_eta = float(afl_eta)
        self.afl_mixing = float(afl_mixing)
        self.afl_risk_metric = str(afl_risk_metric)
        self.afl_warmup_rounds = int(afl_warmup_rounds)
        self.afl_use_num_examples = bool(afl_use_num_examples)

        # AFL mixture weights lambda on simplex
        self._lam = np.ones((self.num_clients_total,), dtype=np.float64) / float(max(1, self.num_clients_total))

        # Per-round caches
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
            "fit_mean_train_loss",
            "fit_p95_train_loss",
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

            "afl_eta",
            "afl_mixing",
            "afl_risk_metric_code",
            "afl_lambda_entropy",
            "afl_lambda_gini",
            "afl_lambda_max",
            "afl_lambda_min",
            "afl_lambda_top5_mean",
            "afl_updated",
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

    def _lam_mixed(self) -> np.ndarray:
        lam = self._lam.astype(np.float64, copy=True)
        if self.afl_mixing and self.afl_mixing > 0:
            u = 1.0 / float(max(1, self.num_clients_total))
            lam = (1.0 - self.afl_mixing) * lam + self.afl_mixing * u
        lam = np.clip(lam, 1e-12, None)
        lam = lam / np.sum(lam)
        return lam

    def _weighted_sample_clients(self, client_manager, num_clients: int) -> List[fl.server.client_proxy.ClientProxy]:
        # Active AFL sampling: sample clients according to lambda (with uniform mixing).
        all_clients = list(client_manager.all().values())
        if not all_clients:
            return []
        # Flower client ids are strings, usually "0".."K-1" in simulation.
        cids = [int(c.cid) for c in all_clients]
        lam = self._lam_mixed()
        probs = np.array([lam[c] if 0 <= c < lam.size else 0.0 for c in cids], dtype=np.float64)
        if probs.sum() <= 0:
            probs = np.ones_like(probs, dtype=np.float64) / float(len(probs))
        else:
            probs = probs / probs.sum()

        k = min(int(num_clients), len(all_clients))
        idxs = np.random.choice(len(all_clients), size=k, replace=False, p=probs)
        chosen = [all_clients[i] for i in idxs.tolist()]

        # Debug print so AFL is visibly active
        chosen_ids = [c.cid for c in chosen]
        top5 = np.sort(self._lam_mixed())[::-1][:5]
        print(f"[AFL] sample round: chosen={chosen_ids} lambda_top5={np.round(top5, 4).tolist()}")
        return chosen

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start_ts = time.time()

        # Respect FedAvg sizing knobs, but use explicit AFL sampling
        num_available = client_manager.num_available()
        if num_available == 0:
            return []

        sample_size, min_num_clients = self.num_fit_clients(num_available)
        sample_size = int(min(sample_size, num_available))
        min_num_clients = int(min(min_num_clients, num_available))
        if sample_size < min_num_clients:
            sample_size = min_num_clients

        clients = self._weighted_sample_clients(client_manager, sample_size)

        # Cache payload size and sent ndarrays (for delta norms)
        self._clients_fit_configured_this_round = int(len(clients))
        self._payload_bytes_this_round = float(bytes_of_parameters(parameters))
        try:
            self._sent_ndarrays_this_round = parameters_to_ndarrays(parameters)
        except Exception:
            self._sent_ndarrays_this_round = None

        cfg = {"server_round": int(server_round)}
        fit_ins = FitIns(parameters, cfg)
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Evaluate all clients each round so AFL lambda update is well-defined and clearly active.
        all_clients = list(client_manager.all().values())
        cfg = {"server_round": int(server_round)}
        ev_ins = EvaluateIns(parameters, cfg)
        return [(c, ev_ins) for c in all_clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print(f"[warn] Round {server_round}: no fit results")
            return None, {}

        clients_fit = int(len(results))
        fit_losses: List[float] = []
        delta_l2s: List[float] = []

        sent_nds = self._sent_ndarrays_this_round
        payload_bytes = float(self._payload_bytes_this_round)

        weighted: List[Tuple[float, List[np.ndarray]]] = []
        for client, fit_res in results:
            m = fit_res.metrics or {}
            if "train_loss" in m:
                try:
                    fit_losses.append(float(m["train_loss"]))
                except Exception:
                    pass

            try:
                client_nds = parameters_to_ndarrays(fit_res.parameters)
            except Exception:
                continue

            if sent_nds is not None:
                try:
                    d = l2_delta(client_nds, sent_nds)
                    if not math.isnan(d):
                        delta_l2s.append(float(d))
                except Exception:
                    pass

            # AFL-weighted aggregation (active): use current lambda for this client id
            try:
                cid_int = int(client.cid)
            except Exception:
                cid_int = int(m.get("lang_idx", -1))
            lam = self._lam_mixed()
            w = float(lam[cid_int]) if 0 <= cid_int < lam.size else 0.0

            if self.afl_use_num_examples:
                w *= float(getattr(fit_res, "num_examples", 1.0))

            if w <= 0:
                w = 1e-12
            weighted.append((w, client_nds))

        new_nds = weighted_average_params(weighted)
        params_agg = ndarrays_to_parameters(new_nds)

        fit_mean_train_loss = float(np.mean(fit_losses)) if fit_losses else float("nan")
        fit_p95_train_loss = float(np.percentile(fit_losses, 95)) if fit_losses else float("nan")
        fit_mean_delta_l2 = float(np.mean(delta_l2s)) if delta_l2s else float("nan")
        fit_max_delta_l2 = float(np.max(delta_l2s)) if delta_l2s else float("nan")

        round_download_bytes = float(payload_bytes * float(self._clients_fit_configured_this_round)) if payload_bytes == payload_bytes else float("nan")
        round_upload_bytes = float(payload_bytes * float(clients_fit)) if payload_bytes == payload_bytes else float("nan")

        self._fit_round_stats[int(server_round)] = {
            "clients_fit": float(clients_fit),
            "fit_mean_train_loss": float(fit_mean_train_loss),
            "fit_p95_train_loss": float(fit_p95_train_loss),
            "fit_mean_delta_l2": float(fit_mean_delta_l2),
            "fit_max_delta_l2": float(fit_max_delta_l2),
            "payload_bytes": float(payload_bytes),
            "round_upload_bytes": float(round_upload_bytes),
            "round_download_bytes": float(round_download_bytes),
        }

        return params_agg, {}

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

        # Risk vector for AFL update
        risk = np.full((self.num_clients_total,), np.nan, dtype=np.float64)

        for client, eval_res in results:
            loss_by_client.append(float(eval_res.loss))
            m = eval_res.metrics or {}

            # Prefer lang_idx from metrics, fall back to client.cid
            if "lang_idx" in m:
                idx = int(m["lang_idx"])
            else:
                try:
                    idx = int(client.cid)
                except Exception:
                    idx = -1

            vloss = float(m.get("val_loss", eval_res.loss))
            vcer = float(m.get("val_cer_mean", float("nan")))
            vwer = float(m.get("val_wer", float("nan")))
            vwacc = float(m.get("val_word_correct_acc", float("nan")))
            vsacc = float(m.get("val_sent_acc", float("nan")))

            if not math.isnan(vcer):
                cer_by_client.append(vcer)
            if not math.isnan(vwer):
                wer_by_client.append(vwer)
            if not math.isnan(vwacc):
                wacc_by_client.append(vwacc)
            if not math.isnan(vsacc):
                sacc_by_client.append(vsacc)

            if 0 <= idx < self.num_clients_total:
                if self.afl_risk_metric == "cer" and not math.isnan(vcer):
                    risk[idx] = float(vcer)
                else:
                    risk[idx] = float(vloss)

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

        # Active AFL lambda update (exponentiated gradient), after warmup
        afl_updated = 0.0
        if int(server_round) > self.afl_warmup_rounds and self.afl_eta > 0:
            mask = ~np.isnan(risk)
            if np.any(mask):
                r = risk.copy()
                # Stabilize exponent: shift by mean on observed entries
                r_obs = r[mask]
                r[mask] = r_obs - float(np.mean(r_obs))
                self._lam[mask] = self._lam[mask] * np.exp(self.afl_eta * r[mask])
                # Keep missing entries as-is, then renormalize
                self._lam = np.clip(self._lam, 1e-12, None)
                self._lam = self._lam / float(np.sum(self._lam))
                afl_updated = 1.0

        lam_m = self._lam_mixed()
        top5_mean = float(np.mean(np.sort(lam_m)[::-1][:5])) if lam_m.size >= 5 else float(np.mean(lam_m))

        row: Dict[str, float] = {
            "round": int(server_round),
            "wall_time_s": wall_s,

            "clients_eval": int(len(results)),
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

            "afl_eta": float(self.afl_eta),
            "afl_mixing": float(self.afl_mixing),
            "afl_risk_metric_code": float(1.0 if self.afl_risk_metric == "cer" else 0.0),
            "afl_lambda_entropy": float(entropy(lam_m)),
            "afl_lambda_gini": float(gini(lam_m)),
            "afl_lambda_max": float(np.max(lam_m)),
            "afl_lambda_min": float(np.min(lam_m)),
            "afl_lambda_top5_mean": float(top5_mean),
            "afl_updated": float(afl_updated),
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
            f"lambda_max={row['afl_lambda_max']:.4f} ent={row['afl_lambda_entropy']:.3f} updated={int(afl_updated)}"
        )

        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)

        return mean_loss, row


# -----------------------------
# Run one experiment
# -----------------------------
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
                    "method": "AFL-LoRA",
                    "afl_eta": float(args.afl_eta),
                    "afl_mixing": float(args.afl_mixing),
                    "afl_risk_metric": str(args.afl_risk_metric),
                    "lora_r": int(args.lora_r),
                    "lora_alpha": float(args.lora_alpha),
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
    local_steps = int(args.local_steps)
    max_train_hours = float(args.max_train_hours)
    max_valid_hours = float(args.max_valid_hours)
    ft_frac_use = float(ft_frac_value)

    if args.unbounded_budget_baseline:
        clients_per_round = len(all_langs)
        ft_frac_use = 1.0
        max_train_hours = 0.0
        max_valid_hours = 0.0
        eval_utterances = 0
        local_steps = 0

    lora_targets = tuple([x.strip() for x in args.lora_targets.split(",") if x.strip()])
    if not lora_targets:
        lora_targets = ("q_proj", "k_proj", "v_proj", "out_proj")

    # Initialize global LoRA parameters by constructing a template model once
    tmp_proc = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    tmp_model = Wav2Vec2ForCTC.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    # Freeze and inject LoRA
    for p in tmp_model.parameters():
        p.requires_grad = False
    replaced = inject_lora_into_wav2vec2_attn(
        tmp_model, targets=lora_targets, r=int(args.lora_r), alpha=float(args.lora_alpha), dropout=float(args.lora_dropout)
    )
    if replaced == 0:
        raise RuntimeError("Template model LoRA injection replaced 0 layers. Check --lora_targets.")
    if args.train_lm_head:
        for p in tmp_model.lm_head.parameters():
            p.requires_grad = True

    trainable_names, init_arrays = get_trainable_weights(tmp_model)
    init_parameters = ndarrays_to_parameters(init_arrays)

    def client_fn(cid: str):
        i = int(cid)
        lang_id = all_langs[i]
        mms_target = lang_to_mms.get(lang_id, None)

        return LocalFleursAFLLoraClient(
            cid=cid,
            lang_id=lang_id,
            lang_dir=os.path.join(args.data_root, lang_id),
            lang_idx=i,
            all_langs=all_langs,
            model_id=args.model_id,
            device=args.device,
            fp16=fp16,
            use_mms_adapters=use_mms_adapters,
            mms_target_lang=mms_target,
            lora_r=int(args.lora_r),
            lora_alpha=float(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            lora_targets=lora_targets,
            train_lm_head=bool(args.train_lm_head),
            batch_size=int(args.batch_size),
            local_steps=local_steps,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            max_grad_norm=float(args.max_grad_norm),
            ft_frac=float(ft_frac_use),
            max_train_hours=float(max_train_hours),
            max_valid_hours=float(max_valid_hours),
            max_audio_s=float(args.max_audio_s),
            eval_utterances=int(eval_utterances),
            trainable_names=trainable_names,
            seed=int(args.seed),
            cache_dir=args.cache_dir,
        )

    tail_fracs = [float(x.strip()) for x in args.tail_fracs.split(",") if x.strip()]
    if not tail_fracs:
        tail_fracs = [0.10, 0.20, 0.30]

    strategy = AFLLoraStrategy(
        num_clients_total=len(all_langs),
        trainable_names=trainable_names,
        log_csv=log_csv_path,
        tail_fracs=tail_fracs,
        afl_eta=float(args.afl_eta),
        afl_mixing=float(args.afl_mixing),
        afl_risk_metric=str(args.afl_risk_metric),
        afl_warmup_rounds=int(args.afl_warmup_rounds),
        afl_use_num_examples=bool(args.afl_use_num_examples),
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


def _suffix_log_csv(base_csv: str, ft_frac_value: float) -> str:
    root, ext = os.path.splitext(base_csv)
    p = int(round(ft_frac_value * 100))
    return f"{root}_ft{p:02d}{ext if ext else '.csv'}"


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
    ap.add_argument("--local_steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--eval_utterances", type=int, default=128)
    ap.add_argument("--log_csv", type=str, default="runs/afl_lora_fleurs.csv")

    ap.add_argument("--num_cpus_per_client", type=float, default=2.0)
    ap.add_argument("--num_gpus_per_client", type=float, default=1.0 if torch.cuda.is_available() else 0.0)

    ap.add_argument("--unbounded_budget_baseline", action="store_true")

    ap.add_argument("--tail_fracs", type=str, default="0.10,0.20,0.30")

    # LoRA knobs
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,out_proj")
    ap.add_argument("--train_lm_head", action="store_true")

    # AFL knobs (active)
    ap.add_argument("--afl_eta", type=float, default=0.1, help="Exponentiated-gradient step size for lambda update")
    ap.add_argument("--afl_mixing", type=float, default=0.05, help="Uniform mixing to prevent starvation")
    ap.add_argument("--afl_risk_metric", type=str, default="loss", choices=["loss", "cer"], help="Risk used for lambda update")
    ap.add_argument("--afl_warmup_rounds", type=int, default=0, help="Rounds before enabling AFL lambda update")
    ap.add_argument("--afl_use_num_examples", action="store_true", help="If set, multiply AFL weight by num_examples in aggregation")

    args = ap.parse_args()

    if args.ft_fracs.strip():
        ft_list = [float(x.strip()) for x in args.ft_fracs.split(",") if x.strip()]
        if not ft_list:
            ft_list = [float(args.ft_frac)]
    else:
        ft_list = [float(args.ft_frac)]

    for k, ftv in enumerate(ft_list):
        out_csv = args.log_csv if len(ft_list) == 1 else _suffix_log_csv(args.log_csv, ftv)
        print(f"[run] method=AFL-LoRA ft_frac={ftv:.2f} log_csv={out_csv}")
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
