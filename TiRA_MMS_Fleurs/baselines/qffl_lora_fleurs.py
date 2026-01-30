#!/usr/bin/env python3
# qffl_lora_fleurs.py
#
# Standard q-FedAvg (q-FFL solver, Algorithm 2) + LoRA adapters for MMS on FLEURS.
# - Clients: one language directory per client under --data_root
# - Trainable payload: LoRA only by default (optionally also lm_head)
# - Server update (q-FedAvg):
#     Delta_wk = L_k * (w_t - w_k)
#     Delta_k  = F_k(w_t)^q * Delta_wk
#     h_k      = q * F_k(w_t)^(q-1) * ||Delta_wk||^2 + L_k * F_k(w_t)^q
#     w_{t+1}  = w_t - server_lr * (sum Delta_k) / (sum h_k)
#
# One-command run:
#   python qffl_lora_fleurs.py
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
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

import torchaudio
from transformers import AutoProcessor, AutoConfig, Wav2Vec2ForCTC

import splits as splits_mod


# -----------------------------
# Text normalization + CER/WER
# -----------------------------
_PARENS = re.compile(r"\([^)]*\)")
_BAD = re.compile(r"[^\w\s'’]+", flags=re.UNICODE)
_WS = re.compile(r"\s+")


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
# LoRA modules (no PEFT dependency)
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear base")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1.0, float(self.r))
        self.dropout_p = float(dropout)

        for p in self.base.parameters():
            p.requires_grad = False

        dev = self.base.weight.device
        dtype = self.base.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=dev, dtype=dtype))

        nn.init.normal_(self.lora_A, std=0.01)

        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p and self.dropout_p > 0 else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        x_d = self.dropout(x)

        if x_d.dim() == 3:
            xA = torch.einsum("bth,rh->btr", x_d, self.lora_A)
            dY = torch.einsum("btr,or->bto", xA, self.lora_B) * self.scaling
            return y0 + dY

        if x_d.dim() == 2:
            xA = torch.einsum("bh,rh->br", x_d, self.lora_A)
            dY = torch.einsum("br,or->bo", xA, self.lora_B) * self.scaling
            return y0 + dY

        raise ValueError(f"Unsupported x.dim={x_d.dim()} for LoRALinear")

def inject_lora_into_wav2vec2_attention(model, r, alpha, dropout, targets=("q_proj","k_proj","v_proj","out_proj")):
    wrapped = 0
    enc = model.wav2vec2.encoder
    for layer in enc.layers:
        attn = getattr(layer, "attention", None)
        if attn is None:
            continue
        for t in targets:
            lin = getattr(attn, t, None)
            if isinstance(lin, nn.Linear):
                w = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
                w.to(lin.weight.device)
                setattr(attn, t, w)
                wrapped += 1
    return wrapped


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
    missing = [n for n in names if n not in name_to_param]
    if missing:
        raise KeyError(f"Trainable parameter(s) missing in client model: {missing[:5]} (and {max(0, len(missing)-5)} more)")
    for n, a in zip(names, arrays):
        p = name_to_param[n]
        t = torch.from_numpy(a).to(p.device)
        if p.data.shape != t.shape:
            raise ValueError(f"Shape mismatch for {n}: {tuple(p.data.shape)} vs {tuple(t.shape)}")
        p.data.copy_(t)


def payload_bytes_from_arrays(arrays: List[np.ndarray]) -> int:
    return int(sum(int(a.nbytes) for a in arrays))


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
# Flower client (LoRA)
# -----------------------------
class LocalFleursLoRAClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        lang_id: str,
        lang_dir: str,
        model_id: str,
        device: str,
        fp16: bool,
        use_mms_adapters: bool,
        mms_target_lang: Optional[str],
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
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_targets: Tuple[str, ...],
        train_lm_head: bool,
        pre_loss_batches: int,
        grad_ckpt: bool,
        trainable_names: List[str],
        seed: int,
        cache_dir: Optional[str],
    ):
        self.cid = cid
        self.lang_id = lang_id
        self.lang_dir = lang_dir
        self.model_id = model_id
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self.use_mms_adapters = bool(use_mms_adapters)
        self.mms_target_lang = mms_target_lang
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
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_targets = tuple(lora_targets)
        self.train_lm_head = bool(train_lm_head)
        self.pre_loss_batches = int(pre_loss_batches)
        self.grad_ckpt = bool(grad_ckpt)
        self.trainable_names = list(trainable_names)
        self.seed = int(seed)
        self.cache_dir = cache_dir

        self._init()

    def _init(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)

        if self.grad_ckpt:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        if self.use_mms_adapters:
            if not self.mms_target_lang:
                raise ValueError(f"Missing MMS target lang for {self.lang_id}")
            self.processor.tokenizer.set_target_lang(self.mms_target_lang)
            self.model.load_adapter(self.mms_target_lang)
            self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False

        wrapped = inject_lora_into_wav2vec2_attention(
            self.model, r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout, targets=self.lora_targets
        )
        if wrapped == 0:
            raise RuntimeError("No attention Linear modules were wrapped by LoRA. Check model structure/targets.")

        if self.train_lm_head:
            head = getattr(self.model, "lm_head", None)
            if isinstance(head, nn.Linear):
                for p in head.parameters():
                    p.requires_grad = True

        trainables = [p for _, p in trainable_named_parameters(self.model)]
        if len(trainables) == 0:
            raise RuntimeError("No trainable parameters found after LoRA injection.")

        self.optimizer = torch.optim.AdamW(
            trainables,
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
        set_trainable_weights(self.model, self.trainable_names, parameters)

    def _estimate_pre_loss(self) -> Tuple[float, List[Dict[str, torch.Tensor]], Iterable[Dict[str, torch.Tensor]]]:
        """
        Estimate F_k(w_t) before any optimizer step.
        Returns (pre_loss, buffered_batches, remaining_iterator).
        buffered_batches are the batches used for pre-loss so training can reuse them.
        """
        it = iter(self.train_loader)
        buf = []
        losses = []

        b = max(1, self.pre_loss_batches)
        self.model.eval()
        with torch.no_grad():
            for _ in range(b):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                buf.append(batch)
                batch_d = {k: v.to(self.device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=(self.fp16 and self.device.type == "cuda")):
                    out = self.model(**batch_d)
                    losses.append(float(out.loss.detach().item()))

        if len(losses) == 0:
            return 1.0, buf, it
        return float(sum(losses) / len(losses)), buf, it

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        pre_loss, pre_buf, it_rest = self._estimate_pre_loss()

        self.model.train()

        step = 0
        total_loss = 0.0
        seen_examples = 0
        t0 = time.time()

        def batch_stream():
            for b in pre_buf:
                yield b
            for b in it_rest:
                yield b

        for batch in batch_stream():
            if self.local_steps > 0 and step >= self.local_steps:
                break

            bs = int(batch["input_values"].shape[0])
            seen_examples += bs

            batch_d = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.fp16 and self.device.type == "cuda")):
                out = self.model(**batch_d)
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
            "train_steps": int(step),
            "train_examples": int(seen_examples),
            "train_loss": float(avg_loss) if math.isfinite(avg_loss) else float(pre_loss),
            "train_loss_pre": float(pre_loss),
            "fit_time_s": float(dt),
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
                batch_d = {k: v.to(self.device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=(self.fp16 and self.device.type == "cuda")):
                    out = self.model(**batch_d)
                    loss = out.loss
                total_loss += float(loss.detach().item())
                n_batches += 1

                logits = out.logits
                pred_ids = torch.argmax(logits, dim=-1)

                hyps = self.processor.batch_decode(pred_ids)
                labels = batch_d["labels"].detach().cpu().numpy()
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
# Standard q-FedAvg strategy + tail logs
# -----------------------------
class QFedAvgLoRAStrategy(FedAvg):
    """
    Standard q-FedAvg (Algorithm 2) over the trainable payload.

    Uses client-reported F_k(w_t) as train_loss_pre (estimated before any local update).
    Uses client-reported L_k as train_steps (actual number of optimizer steps executed).

    Server update:
      Delta_wk = L_k * (w_t - w_k)
      Delta_k  = F_k(w_t)^q * Delta_wk
      h_k      = q * F_k(w_t)^(q-1) * ||Delta_wk||^2 + L_k * F_k(w_t)^q
      w_{t+1}  = w_t - server_lr * (sum Delta_k) / (sum h_k)
    """
    def __init__(
        self,
        trainable_names: List[str],
        log_csv: str,
        tail_fracs: List[float],
        q: float,
        server_lr: float,
        eps: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trainable_names = list(trainable_names)
        self.log_csv = log_csv
        self.tail_fracs = [float(x) for x in tail_fracs]
        self.q = float(q)
        self.server_lr = float(server_lr)
        self.eps = float(eps)

        self._t0 = time.time()
        self._last_parameters = kwargs.get("initial_parameters", None)
        self._fit_summaries: Dict[int, Dict[str, float]] = {}

        os.makedirs(os.path.dirname(self.log_csv) or ".", exist_ok=True)

        base_fields = [
            "round",
            "wall_time_s",
            "clients_fit",
            "qffl_q",
            "qffl_server_lr",
            "fit_mean_train_loss",
            "fit_p95_train_loss",
            "fit_mean_h",
            "fit_max_h",
            "fit_sum_h",
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

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            print(f"[warn] Round {server_round}: failures={len(failures)} (fit)")

        if not results:
            print(f"[warn] Round {server_round}: no successful fit results, keeping previous parameters")
            self._fit_summaries[int(server_round)] = {
                "clients_fit": 0.0,
                "qffl_q": self.q,
                "qffl_server_lr": self.server_lr,
                "fit_mean_train_loss": float("nan"),
                "fit_p95_train_loss": float("nan"),
                "fit_mean_h": float("nan"),
                "fit_max_h": float("nan"),
                "fit_sum_h": float("nan"),
                "fit_mean_delta_l2": float("nan"),
                "fit_max_delta_l2": float("nan"),
                "payload_bytes": float("nan"),
                "round_upload_bytes": float("nan"),
                "round_download_bytes": float("nan"),
            }
            return (self._last_parameters, {}) if self._last_parameters is not None else None

        if self._last_parameters is None:
            # Should not happen if initial_parameters passed, but handle gracefully
            agg = super().aggregate_fit(server_round, results, failures)
            if agg is None:
                return None
            params_agg, _ = agg
            self._last_parameters = params_agg
            return params_agg, {}

        w_t = parameters_to_ndarrays(self._last_parameters)
        payload_bytes = payload_bytes_from_arrays(w_t)

        sum_h = 0.0
        numerator = [np.zeros_like(wi) for wi in w_t]

        losses = []
        hs = []
        delta_l2s = []

        used = 0

        for _, fit_res in results:
            if fit_res is None or getattr(fit_res, "parameters", None) is None:
                continue
            w_k = parameters_to_ndarrays(fit_res.parameters)
            if len(w_k) != len(w_t):
                continue

            m = fit_res.metrics or {}
            l_raw = m.get("train_loss_pre", None)
            if l_raw is None or (isinstance(l_raw, float) and math.isnan(l_raw)):
                l_raw = m.get("train_loss", None)

            try:
                Fk = float(l_raw)
            except Exception:
                Fk = float("nan")

            if not math.isfinite(Fk):
                # If a client gives no finite loss, skip it to avoid contaminating the round
                continue

            Lk_raw = m.get("train_steps", 1)
            try:
                Lk = float(Lk_raw)
            except Exception:
                Lk = 1.0
            Lk = max(1.0, Lk)

            # Delta_wk = Lk * (w_t - w_k)
            delta0 = [wt - wk for wt, wk in zip(w_t, w_k)]
            # norm_sq of Delta_wk
            norm_sq0 = 0.0
            for d in delta0:
                norm_sq0 += float(np.sum(d.astype(np.float64) ** 2))
            norm_sq = (Lk * Lk) * norm_sq0
            delta_l2 = float(math.sqrt(max(0.0, norm_sq)))

            Fk_eps = max(self.eps, Fk + self.eps)

            if self.q == 0.0:
                Fkq = 1.0
                h_k = Lk * 1.0  # reduces to FedAvg-like scaling with Lk
            else:
                Fkq = float(Fk_eps ** self.q)
                h_k = (self.q * float(Fk_eps ** (self.q - 1.0)) * norm_sq) + (Lk * Fkq)

            if not math.isfinite(h_k) or h_k <= 0.0:
                continue

            # Delta_k = Fk^q * Delta_wk
            for i in range(len(numerator)):
                numerator[i] += (Fkq * Lk) * delta0[i]

            sum_h += float(h_k)
            used += 1

            losses.append(float(Fk))
            hs.append(float(h_k))
            delta_l2s.append(delta_l2)

        if used == 0 or not math.isfinite(sum_h) or sum_h <= 0.0:
            print(f"[warn] Round {server_round}: no usable client updates after filtering, keeping previous parameters")
            self._fit_summaries[int(server_round)] = {
                "clients_fit": 0.0,
                "qffl_q": self.q,
                "qffl_server_lr": self.server_lr,
                "fit_mean_train_loss": float("nan"),
                "fit_p95_train_loss": float("nan"),
                "fit_mean_h": float("nan"),
                "fit_max_h": float("nan"),
                "fit_sum_h": float("nan"),
                "fit_mean_delta_l2": float("nan"),
                "fit_max_delta_l2": float("nan"),
                "payload_bytes": float(payload_bytes),
                "round_upload_bytes": float("nan"),
                "round_download_bytes": float("nan"),
            }
            return (self._last_parameters, {})

        # w_{t+1} = w_t - server_lr * (sum Delta_k)/(sum h_k)
        step = self.server_lr / sum_h
        w_new = [wt - step * num for wt, num in zip(w_t, numerator)]
        params_new = ndarrays_to_parameters(w_new)
        self._last_parameters = params_new

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        p95_loss = float(np.percentile(losses, 95.0)) if len(losses) >= 2 else (losses[0] if losses else float("nan"))
        mean_h = float(np.mean(hs)) if hs else float("nan")
        max_h = float(np.max(hs)) if hs else float("nan")
        mean_dl2 = float(np.mean(delta_l2s)) if delta_l2s else float("nan")
        max_dl2 = float(np.max(delta_l2s)) if delta_l2s else float("nan")

        clients_fit = int(used)
        round_upload_bytes = float(clients_fit * payload_bytes)
        round_download_bytes = float(clients_fit * payload_bytes)

        self._fit_summaries[int(server_round)] = {
            "clients_fit": float(clients_fit),
            "qffl_q": float(self.q),
            "qffl_server_lr": float(self.server_lr),
            "fit_mean_train_loss": float(mean_loss),
            "fit_p95_train_loss": float(p95_loss),
            "fit_mean_h": float(mean_h),
            "fit_max_h": float(max_h),
            "fit_sum_h": float(sum_h),
            "fit_mean_delta_l2": float(mean_dl2),
            "fit_max_delta_l2": float(max_dl2),
            "payload_bytes": float(payload_bytes),
            "round_upload_bytes": float(round_upload_bytes),
            "round_download_bytes": float(round_download_bytes),
        }

        return params_new, {}

    def aggregate_evaluate(self, server_round, results, failures):
        if failures:
            print(f"[warn] Round {server_round}: failures={len(failures)} (eval)")

        row = {k: float("nan") for k in self._csv_fields}
        row["round"] = int(server_round)
        row["wall_time_s"] = float(time.time() - self._t0)

        fit_sum = self._fit_summaries.get(int(server_round), None)
        if fit_sum is not None:
            for k, v in fit_sum.items():
                if k in row:
                    row[k] = float(v)
        else:
            row["clients_fit"] = 0.0
            row["qffl_q"] = float(self.q)
            row["qffl_server_lr"] = float(self.server_lr)

        if not results:
            row["clients_eval"] = 0.0
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

        row["clients_eval"] = float(len(results))
        row["mean_loss"] = float(mean_loss)

        row["mean_cer"] = float(mean_cer)
        row["max_cer"] = float(max_cer)
        row["p95_cer"] = float(p95_cer)

        row["mean_wer"] = float(mean_wer)
        row["max_wer"] = float(max_wer)
        row["p95_wer"] = float(p95_wer)

        row["mean_word_correct_acc"] = float(mean_wacc)
        row["min_word_correct_acc"] = float(min_wacc)
        row["p05_word_correct_acc"] = float(p05_wacc)

        row["mean_sent_acc"] = float(mean_sacc)
        row["min_sent_acc"] = float(min_sacc)
        row["p05_sent_acc"] = float(p05_sacc)

        for tf in self.tail_fracs:
            p = int(round(tf * 100))
            row[f"tail{p}_cer"] = tail_mean(cer_by_client, tf, higher_is_worse=True)
            row[f"tail{p}_wer"] = tail_mean(wer_by_client, tf, higher_is_worse=True)
            row[f"tail{p}_word_correct_acc"] = tail_mean(wacc_by_client, tf, higher_is_worse=False)
            row[f"tail{p}_sent_acc"] = tail_mean(sacc_by_client, tf, higher_is_worse=False)

        print(
            f"[Round {server_round}] "
            f"fitClients={int(row.get('clients_fit', 0))} evalClients={int(row.get('clients_eval', 0))} "
            f"meanCER={mean_cer:.4f} maxCER={max_cer:.4f} tail10CER={row.get('tail10_cer', float('nan')):.4f} "
            f"meanLoss={mean_loss:.4f} q={self.q:.2f}"
        )

        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)

        return mean_loss, row


# -----------------------------
# Init trainables without loading full 1B weights
# -----------------------------
def infer_wav2vec2_dims(model_id: str, cache_dir: Optional[str]) -> Tuple[int, int]:
    cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    hidden = int(getattr(cfg, "hidden_size"))
    n_layer = int(getattr(cfg, "num_hidden_layers"))
    return hidden, n_layer


def build_initial_lora_parameters(
    model_id: str,
    cache_dir: Optional[str],
    lora_r: int,
    lora_targets: Tuple[str, ...],
    seed: int,
) -> Tuple[List[str], "fl.common.Parameters", int]:
    hidden, n_layer = infer_wav2vec2_dims(model_id, cache_dir=cache_dir)
    r = int(lora_r)

    rng = np.random.RandomState(int(seed))
    name_to_arr: Dict[str, np.ndarray] = {}

    for lid in range(n_layer):
        for t in lora_targets:
            base = f"wav2vec2.encoder.layers.{lid}.attention.{t}"
            A_name = f"{base}.lora_A"
            B_name = f"{base}.lora_B"
            name_to_arr[A_name] = (rng.normal(loc=0.0, scale=0.01, size=(r, hidden))).astype(np.float32)
            name_to_arr[B_name] = np.zeros((hidden, r), dtype=np.float32)

    trainable_names = sorted(name_to_arr.keys())
    arrays = [name_to_arr[n] for n in trainable_names]
    init_parameters = ndarrays_to_parameters(arrays)
    payload_b = payload_bytes_from_arrays(arrays)
    return trainable_names, init_parameters, payload_b


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
                {"data_root": args.data_root, "langs": ok_langs, "num_clients": len(ok_langs), "ft_frac": float(ft_frac_value)},
                f,
                indent=2,
            )

    if args.build_manifests_only:
        return

    all_langs = ok_langs

    clients_per_round = int(args.clients_per_round)
    if clients_per_round > len(all_langs):
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

    lora_targets = tuple(x.strip() for x in args.lora_targets.split(",") if x.strip())
    if not lora_targets:
        lora_targets = ("q_proj", "k_proj", "v_proj", "out_proj")

    trainable_names, init_parameters, payload_b = build_initial_lora_parameters(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        lora_r=args.lora_r,
        lora_targets=lora_targets,
        seed=args.seed,
    )
    print(f"[init] LoRA payload per client: {payload_b/1024/1024:.2f} MiB")

    def client_fn(cid: str):
        i = int(cid)
        lang_id = all_langs[i]
        mms_target = lang_to_mms.get(lang_id, None)

        return LocalFleursLoRAClient(
            cid=cid,
            lang_id=lang_id,
            lang_dir=os.path.join(args.data_root, lang_id),
            model_id=args.model_id,
            device=args.device,
            fp16=fp16,
            use_mms_adapters=use_mms_adapters,
            mms_target_lang=mms_target,
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
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_targets=lora_targets,
            train_lm_head=args.train_lm_head,
            pre_loss_batches=args.pre_loss_batches,
            grad_ckpt=args.grad_ckpt,
            trainable_names=trainable_names,
            seed=args.seed + i * 17,
            cache_dir=args.cache_dir,
        )

    tail_fracs = [float(x.strip()) for x in args.tail_fracs.split(",") if x.strip()]
    if not tail_fracs:
        tail_fracs = [0.10, 0.20, 0.30]

    strategy = QFedAvgLoRAStrategy(
        trainable_names=trainable_names,
        log_csv=log_csv_path,
        tail_fracs=tail_fracs,
        q=args.qffl_q,
        server_lr=args.qffl_server_lr,
        eps=args.qffl_eps,
        fraction_fit=min(1.0, clients_per_round / max(1, len(all_langs))),
        min_fit_clients=min(clients_per_round, len(all_langs)),
        min_available_clients=len(all_langs),
        # Important: do not require all eval clients to succeed
        fraction_evaluate=1.0,
        min_evaluate_clients=1,
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

    # Safer defaults to avoid round-1 all-fail and NaN logs
    ap.add_argument("--max_train_hours", type=float, default=0.25)
    ap.add_argument("--max_valid_hours", type=float, default=0.10)
    ap.add_argument("--max_audio_s", type=float, default=8.0)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--local_steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--eval_utterances", type=int, default=64)
    ap.add_argument("--log_csv", type=str, default="runs/fleurs_qfedavg_lora.csv")

    ap.add_argument("--num_cpus_per_client", type=float, default=1.0)
    ap.add_argument("--num_gpus_per_client", type=float, default=1.0 if torch.cuda.is_available() else 0.0)

    ap.add_argument("--unbounded_budget_baseline", action="store_true")

    ap.add_argument("--tail_fracs", type=str, default="0.10,0.20,0.30")

    # LoRA args
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,out_proj")
    ap.add_argument("--train_lm_head", action="store_true")

    # Pre-loss and memory safety
    ap.add_argument("--pre_loss_batches", type=int, default=1)
    ap.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing (recommended).")
    ap.set_defaults(grad_ckpt=True)

    # q-FedAvg (q-FFL)
    ap.add_argument("--qffl_q", type=float, default=0.1)
    ap.add_argument("--qffl_server_lr", type=float, default=1)
    ap.add_argument("--qffl_eps", type=float, default=1e-8)

    args = ap.parse_args()

    if args.ft_fracs.strip():
        ft_list = [float(x.strip()) for x in args.ft_fracs.split(",") if x.strip()]
        if not ft_list:
            ft_list = [float(args.ft_frac)]
    else:
        ft_list = [float(args.ft_frac)]

    for k, ftv in enumerate(ft_list):
        out_csv = args.log_csv if len(ft_list) == 1 else _suffix_log_csv(args.log_csv, ftv)
        print(f"[run] ft_frac={ftv:.2f} log_csv={out_csv} q={args.qffl_q} lr={args.qffl_server_lr}")
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
