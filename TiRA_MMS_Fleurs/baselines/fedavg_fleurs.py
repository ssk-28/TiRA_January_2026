#!/usr/bin/env python3
# fedavg_lora_fleurs.py
#
# FedAvg + LoRA federated fine-tuning on local ML_SUPERB FLEURS (per-language clients).
# Keeps the same data pipeline and metrics as your HyperLoRA script:
# - per-client eval metrics: CER, WER, word-correct acc, sentence acc
# - server CSV: mean/max/p95 + tail means over clients
# - per-language (per-client) CSV each round for both fit and eval
#
# Key differences vs hypernetwork_fl_fleurs.py:
# - Replaces HyperLoRA with standard LoRA modules injected into encoder attention projections
# - FedAvg aggregates ONLY LoRA parameters (base MMS model remains frozen)
#
# Examples:
#   python fedavg_lora_fleurs.py --langs_json langs_used.json --clients_per_round 8 --rounds 50
#   python fedavg_lora_fleurs.py --langs_json /mnt/data/langs_used.json --ft_fracs 0.1,0.3
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
    """Map dataset lang_id (dir name) to an MMS tokenizer target_lang key."""
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
    """Return (kept_langs, mapping data_lang -> mms_target_lang)."""
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
# Tail helpers
# -----------------------------
def tail_mean(values: List[float], tail_frac: float, higher_is_worse: bool) -> float:
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return float("nan")
    xs.sort()
    k = max(1, int(math.ceil(tail_frac * len(xs))))
    tail = xs[-k:] if higher_is_worse else xs[:k]
    return float(sum(tail) / len(tail))


def pctl(values: List[float], q: float) -> float:
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return float("nan")
    xs.sort()
    idx = int(round((q / 100.0) * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def p95(values: List[float]) -> float:
    return pctl(values, 95.0)


def bytes_of_parameters(p: fl.common.Parameters) -> int:
    # Flower serialized payload size (no protocol overhead)
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
        if a is None or b is None:
            continue
        d = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
        tot += float(np.sum(d * d))
    return float(math.sqrt(tot))


# -----------------------------
# LoRA modules
# -----------------------------
class LoRALinear(nn.Module):
    """
    Standard LoRA for a Linear layer: y = Wx + (B(Ax)) * scaling
    Works for x shape [B,T,H] or [B,H].
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else None

        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base.in_features
        out_f = base.out_features

        self.lora_A = nn.Parameter(torch.zeros(self.r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        if self.dropout is not None:
            x = self.dropout(x)

        if x.dim() == 3:
            # x: [B,T,H], A: [r,H], B: [H,r] after transpose usage
            xA = torch.einsum("bth,rh->btr", x, self.lora_A)
            dY = torch.einsum("btr,hr->bth", xA, self.lora_B) * self.scaling
            return y0 + dY

        if x.dim() == 2:
            xA = torch.einsum("bh,rh->br", x, self.lora_A)
            dY = torch.einsum("br,hr->bh", xA, self.lora_B) * self.scaling
            return y0 + dY

        raise ValueError(f"Unsupported x.dim={x.dim()} for LoRALinear")


def inject_lora(
    model: nn.Module,
    targets=("q_proj", "k_proj", "v_proj", "out_proj"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> None:
    enc = model.wav2vec2.encoder
    for layer in enc.layers:
        attn = getattr(layer, "attention", None)
        if attn is None:
            continue
        for t in targets:
            lin = getattr(attn, t, None)
            if isinstance(lin, nn.Linear):
                setattr(attn, t, LoRALinear(lin, r=r, alpha=alpha, dropout=dropout))


# -----------------------------
# Data loading (same as before)
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
    """
    Reads language_dir/train.csv and language_dir/validation.csv (built by splits.py).
    Applies hashed fine-tune selection based on ft_frac.
    """
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


# -----------------------------
# Flower client (LoRA)
# -----------------------------
class LocalFleursLoRAClient(fl.client.NumPyClient):
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
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        seed: int,
        cache_dir: Optional[str],
        mms_target_lang: Optional[str] = None,
    ):
        self.cid = cid
        self.lang_id = lang_id
        self.lang_dir = lang_dir
        self.lang_idx = lang_idx
        self.all_langs = all_langs
        self.model_id = model_id
        self.device = torch.device(device)
        self.fp16 = fp16
        self.use_mms_adapters = use_mms_adapters
        self.batch_size = batch_size
        self.local_steps = local_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.ft_frac = ft_frac
        self.max_train_hours = max_train_hours
        self.max_valid_hours = max_valid_hours
        self.max_audio_s = max_audio_s
        self.eval_utterances = eval_utterances
        self.trainable_names = trainable_names
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.seed = int(seed)
        self.cache_dir = cache_dir
        self.mms_target_lang = mms_target_lang

        self._init()

    def _init(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)

        if self.use_mms_adapters:
            if not self.mms_target_lang:
                raise ValueError(f"Missing MMS target lang for {self.lang_id}")
            self.processor.tokenizer.set_target_lang(self.mms_target_lang)
            self.model.load_adapter(self.mms_target_lang)

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        inject_lora(self.model, r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout)
        self.model.to(self.device)

        for n, p in self.model.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
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
        self.model.to(self.device)
        set_trainable_weights(self.model, self.trainable_names, parameters)

        for m in self.model.modules():
            if isinstance(m, LoRALinear):
                m.to(self.device)

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
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach().item())
            step += 1

        dt = time.time() - t0
        if step == 0:
            print(f"[warn] client={self.cid} lang={self.lang_id}: 0 train steps, returning unchanged update.")
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
        seen_utts = 0
        num_utts = 0

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
# Strategy with tail logs + per-client logs
# -----------------------------
class LoRAFedAvg(FedAvg):
    def __init__(
        self,
        trainable_names: List[str],
        log_csv: str,
        per_client_csv: str,
        tail_fracs: List[float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trainable_names = trainable_names
        self.log_csv = log_csv
        self.per_client_csv = per_client_csv
        self.tail_fracs = [float(x) for x in tail_fracs]

        self._round_start_ts: Optional[float] = None
        self._fit_round_stats: Dict[int, Dict[str, float]] = {}

        self._payload_bytes_this_round: float = float("nan")
        self._clients_fit_configured_this_round: int = 0
        self._sent_ndarrays_this_round: Optional[List[np.ndarray]] = None

        self._last_parameters = kwargs.get("initial_parameters", None)

        os.makedirs(os.path.dirname(self.log_csv) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.per_client_csv) or ".", exist_ok=True)

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

        self._per_client_fields = [
            "round",
            "phase",  # fit or eval
            "cid",
            "lang_id",
            "lang_idx",
            "train_steps",
            "train_loss",
            "fit_time_s",
            "train_examples",
            "val_loss",
            "val_cer_mean",
            "val_wer",
            "val_word_correct_acc",
            "val_sent_acc",
            "eval_utts",
            "eval_word_tokens",
        ]
        if not os.path.exists(self.per_client_csv):
            with open(self.per_client_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._per_client_fields).writeheader()

    def configure_fit(self, server_round: int, parameters, client_manager):
        self._round_start_ts = time.time()

        cfg = super().configure_fit(server_round, parameters, client_manager)

        self._clients_fit_configured_this_round = int(len(cfg))
        if len(cfg) > 0:
            fit_ins = cfg[0][1]
            self._payload_bytes_this_round = float(bytes_of_parameters(fit_ins.parameters))
            try:
                self._sent_ndarrays_this_round = parameters_to_ndarrays(fit_ins.parameters)
            except Exception:
                self._sent_ndarrays_this_round = None
        else:
            self._payload_bytes_this_round = float("nan")
            self._sent_ndarrays_this_round = None

        return cfg

    def aggregate_fit(self, server_round, results, failures):
        clients_fit = int(len(results))
        fit_losses: List[float] = []
        delta_l2s: List[float] = []

        # per-client fit logging
        with open(self.per_client_csv, "a", newline="", encoding="utf-8") as fpc:
            wr = csv.DictWriter(fpc, fieldnames=self._per_client_fields)
            for client_proxy, fit_res in results:
                m = fit_res.metrics or {}
                if "train_loss" in m:
                    try:
                        fit_losses.append(float(m["train_loss"]))
                    except Exception:
                        pass

                if self._sent_ndarrays_this_round is not None:
                    try:
                        client_nds = parameters_to_ndarrays(fit_res.parameters)
                        d = l2_delta(client_nds, self._sent_ndarrays_this_round)
                        if not math.isnan(d):
                            delta_l2s.append(float(d))
                    except Exception:
                        pass

                row = {k: "" for k in self._per_client_fields}
                row["round"] = int(server_round)
                row["phase"] = "fit"
                row["cid"] = str(getattr(client_proxy, "cid", ""))
                row["lang_id"] = str(m.get("lang_id", ""))
                row["lang_idx"] = int(m.get("lang_idx", -1)) if m.get("lang_idx", None) is not None else -1
                row["train_steps"] = int(m.get("train_steps", 0)) if m.get("train_steps", None) is not None else 0
                row["train_loss"] = float(m.get("train_loss", float("nan"))) if m.get("train_loss", None) is not None else float("nan")
                row["fit_time_s"] = float(m.get("fit_time_s", float("nan"))) if m.get("fit_time_s", None) is not None else float("nan")
                row["train_examples"] = int(m.get("train_examples", 0)) if m.get("train_examples", None) is not None else 0
                wr.writerow(row)

        fit_mean_train_loss = float(np.mean(fit_losses)) if len(fit_losses) > 0 else float("nan")
        fit_p95_train_loss = float(np.percentile(fit_losses, 95)) if len(fit_losses) > 0 else float("nan")

        fit_mean_delta_l2 = float(np.mean(delta_l2s)) if len(delta_l2s) > 0 else float("nan")
        fit_max_delta_l2 = float(np.max(delta_l2s)) if len(delta_l2s) > 0 else float("nan")

        payload_bytes = float(self._payload_bytes_this_round) if self._payload_bytes_this_round == self._payload_bytes_this_round else float("nan")

        if payload_bytes == payload_bytes:
            round_download_bytes = float(payload_bytes * float(self._clients_fit_configured_this_round))
            round_upload_bytes = float(payload_bytes * float(clients_fit))
        else:
            round_download_bytes = float("nan")
            round_upload_bytes = float("nan")

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

        if not results:
            print(f"[warn] Round {server_round}: no fit results, keeping previous parameters")
            return (self._last_parameters, {}) if self._last_parameters is not None else None

        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            print(f"[warn] Round {server_round}: aggregate_fit returned None, keeping previous parameters")
            return (self._last_parameters, {}) if self._last_parameters is not None else None

        params_agg, metrics_agg = aggregated
        if params_agg is None or getattr(params_agg, "tensors", None) is None:
            print(f"[warn] Round {server_round}: aggregated parameters are None, keeping previous parameters")
            return (self._last_parameters, metrics_agg) if self._last_parameters is not None else None

        self._last_parameters = params_agg
        return params_agg, metrics_agg

    def aggregate_evaluate(self, server_round, results, failures):
        wall_s = float("nan") if self._round_start_ts is None else float(time.time() - self._round_start_ts)

        fit_stats = self._fit_round_stats.get(int(server_round), {})
        clients_fit = int(fit_stats.get("clients_fit", float("nan"))) if "clients_fit" in fit_stats else 0

        if not results:
            row = {k: float("nan") for k in self._csv_fields}
            row["round"] = int(server_round)
            row["wall_time_s"] = wall_s
            row["clients_fit"] = float(clients_fit) if clients_fit is not None else float("nan")
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

        # per-client eval logging
        with open(self.per_client_csv, "a", newline="", encoding="utf-8") as fpc:
            wr = csv.DictWriter(fpc, fieldnames=self._per_client_fields)
            for client_proxy, eval_res in results:
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

                row = {k: "" for k in self._per_client_fields}
                row["round"] = int(server_round)
                row["phase"] = "eval"
                row["cid"] = str(getattr(client_proxy, "cid", ""))
                row["lang_id"] = str(m.get("lang_id", ""))
                row["lang_idx"] = int(m.get("lang_idx", -1)) if m.get("lang_idx", None) is not None else -1
                row["val_loss"] = float(m.get("val_loss", float(eval_res.loss)))
                row["val_cer_mean"] = float(m.get("val_cer_mean", float("nan"))) if m.get("val_cer_mean", None) is not None else float("nan")
                row["val_wer"] = float(m.get("val_wer", float("nan"))) if m.get("val_wer", None) is not None else float("nan")
                row["val_word_correct_acc"] = float(m.get("val_word_correct_acc", float("nan"))) if m.get("val_word_correct_acc", None) is not None else float("nan")
                row["val_sent_acc"] = float(m.get("val_sent_acc", float("nan"))) if m.get("val_sent_acc", None) is not None else float("nan")
                row["eval_utts"] = int(m.get("eval_utts", 0)) if m.get("eval_utts", None) is not None else 0
                row["eval_word_tokens"] = int(m.get("eval_word_tokens", 0)) if m.get("eval_word_tokens", None) is not None else 0
                wr.writerow(row)

        mean_loss = float(sum(loss_by_client) / max(1, len(loss_by_client))) if loss_by_client else float("nan")

        mean_cer = float(sum(cer_by_client) / max(1, len(cer_by_client))) if cer_by_client else float("nan")
        max_cer = float(max(cer_by_client)) if cer_by_client else float("nan")
        p95_cer = p95(cer_by_client)

        mean_wer = float(sum(wer_by_client) / max(1, len(wer_by_client))) if wer_by_client else float("nan")
        max_wer = float(max(wer_by_client)) if wer_by_client else float("nan")
        p95_wer = p95(wer_by_client)

        mean_wacc = float(sum(wacc_by_client) / max(1, len(wacc_by_client))) if wacc_by_client else float("nan")
        min_wacc = float(min(wacc_by_client)) if wacc_by_client else float("nan")
        p05_wacc = pctl(wacc_by_client, 5.0)

        mean_sacc = float(sum(sacc_by_client) / max(1, len(sacc_by_client))) if sacc_by_client else float("nan")
        min_sacc = float(min(sacc_by_client)) if sacc_by_client else float("nan")
        p05_sacc = pctl(sacc_by_client, 5.0)

        row: Dict[str, float] = {
            "round": int(server_round),
            "wall_time_s": float(wall_s),
            "clients_fit": float(clients_fit) if clients_fit is not None else float("nan"),
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

        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)

        return mean_loss, row


# -----------------------------
# Language selection helpers
# -----------------------------
def load_langs_from_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "langs" in obj and isinstance(obj["langs"], list):
        return [str(x).strip() for x in obj["langs"] if str(x).strip()]
    raise ValueError(f"langs_json missing 'langs' list: {path}")


def _suffix_log_csv(base_csv: str, ft_frac_value: float) -> str:
    root, ext = os.path.splitext(base_csv)
    p = int(round(ft_frac_value * 100))
    return f"{root}_ft{p:02d}{ext if ext else '.csv'}"


def _suffix_per_client_csv(base_csv: str, ft_frac_value: float) -> str:
    root, ext = os.path.splitext(base_csv)
    p = int(round(ft_frac_value * 100))
    return f"{root}_per_client_ft{p:02d}{ext if ext else '.csv'}"


# -----------------------------
# Run one experiment
# -----------------------------
def run_one(args, ft_frac_value: float, log_csv_path: str, per_client_csv_path: str) -> None:
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

    # Decide how many clients (languages) to use.
    if args.num_clients is not None:
        n_clients = int(args.num_clients)
    else:
        n_clients = int(args.num_langs)

    # Priority for language list:
    # 1) --langs (explicit)
    # 2) --langs_json (if provided and exists)
    # 3) discover
    if args.langs.strip():
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    else:
        lj = args.langs_json.strip()
        if lj and os.path.isfile(lj):
            langs = load_langs_from_json(lj)
        else:
            langs = splits_mod.discover_langs(args.data_root, exclude=exclude)

    langs = [l for l in langs if l not in exclude]
    if n_clients > 0:
        langs = langs[: max(1, n_clients)]
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

    # Optional: write final languages used
    if args.write_langs_json:
        with open(args.write_langs_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_root": args.data_root,
                    "langs": ok_langs,
                    "num_clients": len(ok_langs),
                    "ft_frac": float(ft_frac_value),
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

    # Initialize global LoRA params by creating a cpu model, injecting LoRA, and extracting trainables
    tmp_proc = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    tmp_model = Wav2Vec2ForCTC.from_pretrained(args.model_id, cache_dir=args.cache_dir).to("cpu")
    if use_mms_adapters:
        # NOTE: for global init we do not load any specific adapter
        # Clients will load their own frozen adapter if enabled.
        pass
    for p in tmp_model.parameters():
        p.requires_grad = False
    inject_lora(tmp_model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    for n, p in tmp_model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
    trainable_names, init_arrays = get_trainable_weights(tmp_model)
    init_parameters = ndarrays_to_parameters(init_arrays)

    def client_fn(cid: str):
        i = int(cid)
        lang_id = all_langs[i]
        mms_target = lang_to_mms.get(lang_id, None)
        return LocalFleursLoRAClient(
            cid=cid,
            lang_id=lang_id,
            lang_dir=os.path.join(args.data_root, lang_id),
            lang_idx=i,
            all_langs=all_langs,
            mms_target_lang=mms_target,
            model_id=args.model_id,
            device=args.device,
            fp16=fp16,
            use_mms_adapters=use_mms_adapters,
            batch_size=args.batch_size,
            local_steps=local_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            ft_frac=ft_frac_use,
            max_train_hours=max_train_hours,
            max_valid_hours=max_valid_hours,
            max_audio_s=args.max_audio_s,
            eval_utterances=eval_utterances,
            trainable_names=trainable_names,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    tail_fracs = [float(x.strip()) for x in args.tail_fracs.split(",") if x.strip()]
    if not tail_fracs:
        tail_fracs = [0.10, 0.20, 0.30]

    strategy = LoRAFedAvg(
        trainable_names=trainable_names,
        log_csv=log_csv_path,
        per_client_csv=per_client_csv_path,
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
    ap.add_argument("--langs_json", type=str, default="langs_used.json", help="If --langs empty, use langs from this JSON (expects key 'langs').")
    ap.add_argument("--num_langs", type=int, default=30)
    ap.add_argument("--num_clients", type=int, default=None, help="Overrides --num_langs when set.")
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

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--clients_per_round", type=int, default=4)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ft_frac", type=float, default=0.30)
    ap.add_argument("--ft_fracs", type=str, default="", help="Optional sweep: 0.10,0.20,0.30")

    ap.add_argument("--max_train_hours", type=float, default=0.50)
    ap.add_argument("--max_valid_hours", type=float, default=0.10)
    ap.add_argument("--max_audio_s", type=float, default=16.0)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--local_steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--eval_utterances", type=int, default=128)

    ap.add_argument("--log_csv", type=str, default="runs/fleurs_fedavg_lora.csv")
    ap.add_argument("--per_client_csv", type=str, default="runs/fleurs_fedavg_lora_per_client.csv")

    ap.add_argument("--num_cpus_per_client", type=float, default=2.0)
    ap.add_argument("--num_gpus_per_client", type=float, default=1.0 if torch.cuda.is_available() else 0.0)

    ap.add_argument(
        "--unbounded_budget_baseline",
        action="store_true",
        help="Remove per-round caps: full participation, full train split, full validation eval.",
    )

    ap.add_argument("--tail_fracs", type=str, default="0.30", help="Comma-separated tail fractions, example: 0.10,0.20,0.30")

    args = ap.parse_args()

    if args.ft_fracs.strip():
        ft_list = [float(x.strip()) for x in args.ft_fracs.split(",") if x.strip()]
        if not ft_list:
            ft_list = [float(args.ft_frac)]
    else:
        ft_list = [float(args.ft_frac)]

    for k, ftv in enumerate(ft_list):
        out_csv = args.log_csv if len(ft_list) == 1 else _suffix_log_csv(args.log_csv, ftv)
        out_pc = args.per_client_csv if len(ft_list) == 1 else _suffix_per_client_csv(args.per_client_csv, ftv)
        print(f"[run] ft_frac={ftv:.2f} log_csv={out_csv} per_client_csv={out_pc}")

        run_one(args, ft_frac_value=ftv, log_csv_path=out_csv, per_client_csv_path=out_pc)

        if k + 1 < len(ft_list):
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
