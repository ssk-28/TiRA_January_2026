#!/usr/bin/env python3
# hypernetwork_fl_fleurs.py
#
# HyperLoRA federated fine-tuning on local ML_SUPERB FLEURS.
# Uses splits.py to build and validate per-language manifests.
#
# Usage:
#   python hypernetwork_fl_fleurs.py
#   python hypernetwork_fl_fleurs.py --data_root ML_SUPERB/fleurs --num_langs 8
#   python hypernetwork_fl_fleurs.py --build_manifests_only --force_rebuild_manifests
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

def resolve_mms_target_lang(tokenizer, lang_id: str) -> Optional[str]:
    """Map dataset lang_id (dir name) to an MMS tokenizer target_lang key."""
    opts = set(tokenizer.vocab.keys())

    # Direct hit
    if lang_id in opts:
        return lang_id

    # Prefix hit like "cmn" -> "cmn-script_simplified"
    pref = [k for k in opts if k.startswith(lang_id + "-")]
    if len(pref) == 1:
        return pref[0]
    if len(pref) > 1:
        return sorted(pref)[0]

    # Common alias map for ML-SUPERB/FLEURS style ids -> MMS ids
    alias = {
        "aze": "azj-script_latin",      # Azerbaijani
        "cmn": "cmn-script_simplified", # Mandarin
        "yue": "yue-script_traditional",
        "srp": "srp-script_latin",
        "urd": "urd-script_arabic",
        "uzb": "uzb-script_latin",
        "nep": "npi",                   # Nepali
        "fil": "tgl",                   # Filipino/Tagalog
        "msa": "zlm",                   # Malay
        "swa": "swh",                   # Swahili
        "ori": "ory",                   # Odia
    }
    if lang_id in alias and alias[lang_id] in opts:
        return alias[lang_id]

    return None


def filter_langs_supported_by_mms(model_id: str, cache_dir: Optional[str], data_langs: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return (kept_langs, mapping data_lang->mms_target)."""
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


def cvar(values: List[float], alpha: float = 0.9) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = max(1, int(math.ceil((1.0 - alpha) * len(xs))))
    tail = xs[-k:]
    return float(sum(tail) / len(tail))

def _edit_counts(ref_tokens, hyp_tokens):
    # Returns (S, D, I, C, N)
    # DP over tokens, also reconstruct counts
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
    wer = (S + D + I) / N
    word_correct_acc = C / N
    sent_acc = 1.0 if (S + D + I) == 0 else 0.0
    return wer, word_correct_acc, sent_acc, (S, D, I, C, N)

def pctl(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = int(round((q / 100.0) * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


class HyperContext:
    _lang_idx: Optional[torch.LongTensor] = None

    @classmethod
    def set_lang_idx(cls, lang_idx: torch.LongTensor) -> None:
        cls._lang_idx = lang_idx

    @classmethod
    def get_lang_idx(cls) -> torch.LongTensor:
        if cls._lang_idx is None:
            raise RuntimeError("HyperContext lang_idx not set")
        return cls._lang_idx


class HyperLoRA(nn.Module):
    def __init__(
        self,
        n_lang: int,
        n_layer: int,
        hidden_size: int,
        r: int = 8,
        alpha: float = 16.0,
        code_dim: int = 128,
        lang_emb_dim: int = 64,
        layer_emb_dim: int = 16,
    ):
        super().__init__()
        self.n_lang = n_lang
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)

        self.lang_emb = nn.Embedding(n_lang, lang_emb_dim)
        self.layer_emb = nn.Embedding(n_layer, layer_emb_dim)

        in_dim = lang_emb_dim + layer_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim),
            nn.ReLU(),
        )
        self.to_A = nn.Linear(code_dim, r * hidden_size, bias=False)
        self.to_B = nn.Linear(code_dim, hidden_size * r, bias=False)

        nn.init.normal_(self.to_A.weight, std=0.01)
        nn.init.normal_(self.to_B.weight, std=0.01)

    def gen(self, layer_id: int, lang_idx: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        le = self.lang_emb(lang_idx)
        lid = torch.full_like(lang_idx, fill_value=layer_id)
        pe = self.layer_emb(lid)
        code = self.mlp(torch.cat([le, pe], dim=-1))
        A = self.to_A(code).view(-1, self.r, self.hidden_size) / math.sqrt(self.hidden_size)
        B = self.to_B(code).view(-1, self.hidden_size, self.r)
        return A, B


class HyperLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, hyper: HyperLoRA, layer_id: int):
        super().__init__()
        self.base = base
        self.hyper = hyper
        self.layer_id = layer_id
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        lang_idx = HyperContext.get_lang_idx()

        if x.dim() == 3:
            B = x.size(0)
            if lang_idx.numel() == 1:
                lang_idx = lang_idx.expand(B)
            A, Bm = self.hyper.gen(self.layer_id, lang_idx)
            xA = torch.einsum("bth,brh->btr", x, A)
            dY = torch.einsum("btr,bhr->bth", xA, Bm) * self.hyper.scaling
            return y0 + dY

        if x.dim() == 2:
            B = x.size(0)
            if lang_idx.numel() == 1:
                lang_idx = lang_idx.expand(B)
            A, Bm = self.hyper.gen(self.layer_id, lang_idx)
            xA = torch.einsum("bh,brh->br", x, A)
            dY = torch.einsum("br,bhr->bh", xA, Bm) * self.hyper.scaling
            return y0 + dY

        raise ValueError(f"Unsupported x.dim={x.dim()} for HyperLoRALinear")


def _find_hidden_size_and_n_layers(model: nn.Module) -> Tuple[int, int]:
    enc = model.wav2vec2.encoder
    n_layer = len(enc.layers)
    attn = enc.layers[0].attention
    for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        lin = getattr(attn, name, None)
        if isinstance(lin, nn.Linear):
            return lin.in_features, n_layer
    raise RuntimeError("Could not infer hidden size")


def inject_hyperlora(model: nn.Module, hyper: HyperLoRA, targets=("q_proj", "k_proj", "v_proj", "out_proj")) -> None:
    enc = model.wav2vec2.encoder
    for lid, layer in enumerate(enc.layers):
        attn = getattr(layer, "attention", None)
        if attn is None:
            continue
        for t in targets:
            lin = getattr(attn, t, None)
            if isinstance(lin, nn.Linear) and lin.in_features == lin.out_features == hyper.hidden_size:
                setattr(attn, t, HyperLoRALinear(lin, hyper=hyper, layer_id=lid))


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
    Applies hashed ft split. If ft bucket is empty, falls back to full split to avoid 0 examples.
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


class LocalFleursHyperClient(fl.client.NumPyClient):
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
        seed: int,
        cache_dir: Optional[str],
        mms_target_lang: Optional[str]= None,

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
        self.seed = seed
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

        for p in self.model.parameters():
            p.requires_grad = False

        hidden, n_layer = _find_hidden_size_and_n_layers(self.model)
        self.hyper = HyperLoRA(n_lang=len(self.all_langs), n_layer=n_layer, hidden_size=hidden).to(self.device)
        inject_hyperlora(self.model, self.hyper)
        for p in self.hyper.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            [p for p in self.hyper.parameters() if p.requires_grad],
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
        _, arrays = get_trainable_weights(self.hyper)
        return arrays

    def set_parameters(self, parameters):
        set_trainable_weights(self.hyper, self.trainable_names, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        HyperContext.set_lang_idx(torch.tensor([self.lang_idx], device=self.device, dtype=torch.long))

        self.model.train()
        self.hyper.train()

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
                torch.nn.utils.clip_grad_norm_(self.hyper.parameters(), self.max_grad_norm)

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
            "lang_idx": self.lang_idx,
            "train_steps": step,
            "train_loss": avg_loss,
            "fit_time_s": dt,
            "train_examples": int(seen_examples),
        }
        return new_params, num_ex, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        HyperContext.set_lang_idx(torch.tensor([self.lang_idx], device=self.device, dtype=torch.long))

        self.model.eval()
        self.hyper.eval()

        total_loss = 0.0
        n_batches = 0
        cer_list: List[float] = []
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

                for h, r in zip(hyps, refs):
                    h = normalize_transcript(h.replace("|", " ").lower())
                    r = normalize_transcript(r.replace("|", " ").lower())
                    cer_list.append(cer(h, r))
                    seen_utts += 1
                    if self.eval_utterances and seen_utts >= self.eval_utterances:
                        break
                if self.eval_utterances and seen_utts >= self.eval_utterances:
                    break

        avg_loss = total_loss / max(1, n_batches)
        mean_cer = float(sum(cer_list) / max(1, len(cer_list)))
        max_cer = float(max(cer_list)) if cer_list else float("nan")
        
        tot_S = tot_D = tot_I = tot_C = tot_N = 0
        sent_ok = 0
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
                    hyp = normalize_transcript(hyp_raw.replace("|", " ").lower())
                    ref = normalize_transcript(ref_raw.replace("|", " ").lower())

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
                    if self.eval_utterances and seen_utts >= self.eval_utterances:
                        break

                if self.eval_utterances and seen_utts >= self.eval_utterances:
                    break

        avg_loss = total_loss / max(1, n_batches)
        mean_cer = float(sum(cer_list) / max(1, len(cer_list)))
        max_cer = float(max(cer_list)) if cer_list else float("nan")

        corpus_wer = (tot_S + tot_D + tot_I) / max(1, tot_N)
        corpus_word_correct_acc = tot_C / max(1, tot_N)
        corpus_sent_acc = sent_ok / max(1, num_utts)



        metrics = {
            "lang_id": self.lang_id,
            "lang_idx": self.lang_idx,
            "val_loss": avg_loss,
            "val_cer_mean": mean_cer,
            "val_cer_max": max_cer,
            "eval_utts": int(seen_utts),
            
        }
        metrics.update({
            "val_wer": float(corpus_wer),
            "val_word_correct_acc": float(corpus_word_correct_acc),
            "val_sent_acc": float(corpus_sent_acc),
        })

        
        return float(avg_loss), int(seen_utts), metrics


class HyperFedAvg(FedAvg):
    def __init__(self, trainable_names: List[str], log_csv: str, cvar_alpha: float = 0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable_names = trainable_names
        self.log_csv = log_csv
        self.cvar_alpha = cvar_alpha
        self._t0 = time.time()

        # Keep the last non-None parameters so we can continue if a round aggregates to None
        self._last_parameters = kwargs.get("initial_parameters", None)

        os.makedirs(os.path.dirname(self.log_csv) or ".", exist_ok=True)
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["round", "mean_cer", "max_cer", "cvar90_cer", "p95_cer", "mean_loss", "clients_eval", "wall_time_s"],
                )
                w.writeheader()

    def aggregate_fit(self, server_round, results, failures):
        # If no results, keep previous global parameters
        if not results:
            print(f"[warn] Round {server_round}: no fit results, keeping previous parameters")
            return (self._last_parameters, {}) if self._last_parameters is not None else None

        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            print(f"[warn] Round {server_round}: super().aggregate_fit returned None, keeping previous parameters")
            return (self._last_parameters, {}) if self._last_parameters is not None else None

        params_agg, metrics_agg = aggregated

        # This is the case you hit: params_agg is None
        if params_agg is None or getattr(params_agg, "tensors", None) is None:
            print(f"[warn] Round {server_round}: aggregated parameters are None, keeping previous parameters")
            return (self._last_parameters, metrics_agg) if self._last_parameters is not None else None

        # Convert to ndarrays
        nds = parameters_to_ndarrays(params_agg)

        # Apply the lang_emb row update trick only if present
        name_to_i = {n: i for i, n in enumerate(self.trainable_names)}
        key = "lang_emb.weight"
        if key in name_to_i:
            li = name_to_i[key]
            global_lang_emb = nds[li].copy()

            # Update only the row corresponding to each client's language
            for _, fit_res in results:
                if fit_res is None or getattr(fit_res, "parameters", None) is None:
                    continue
                try:
                    client_params = parameters_to_ndarrays(fit_res.parameters)
                except Exception:
                    continue

                m = fit_res.metrics or {}
                lang_idx = m.get("lang_idx", None)
                if lang_idx is None:
                    continue
                lang_idx = int(lang_idx)

                if li < len(client_params):
                    client_lang_emb = client_params[li]
                    if 0 <= lang_idx < global_lang_emb.shape[0]:
                        global_lang_emb[lang_idx] = client_lang_emb[lang_idx]

            nds[li] = global_lang_emb
            params_fixed = ndarrays_to_parameters(nds)
        else:
            params_fixed = params_agg

        # Save last good parameters
        self._last_parameters = params_fixed

        return params_fixed, metrics_agg

    def aggregate_evaluate(self, server_round, results, failures):
        # Flower expects a tuple (loss, metrics). Never return None.
        if not results:
            row = {
                "round": int(server_round),
                "clients_eval": 0,
                "mean_cer": float("nan"),
                "max_cer": float("nan"),
                "cvar90_cer": float("nan"),
                "p95_cer": float("nan"),
                "mean_loss": float("nan"),
                "wall_time_s": float(time.time() - self._t0),
            }
            print(f"[warn] Round {server_round}: no eval results (failures={len(failures)}).")
            # Optional: still log the row so your CSV has consistent rounds
            with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writerow(row)
            return float("nan"), row

        cer_by_client = []
        loss_by_client = []

        for _, eval_res in results:
            loss_by_client.append(float(eval_res.loss))
            m = eval_res.metrics or {}
            if "val_cer_mean" in m:
                cer_by_client.append(float(m["val_cer_mean"]))

        mean_cer = float(sum(cer_by_client) / max(1, len(cer_by_client)))
        max_cer = float(max(cer_by_client)) if cer_by_client else float("nan")
        cvar90 = cvar(cer_by_client, alpha=self.cvar_alpha)
        p95v = pctl(cer_by_client, 95.0)
        mean_loss = float(sum(loss_by_client) / max(1, len(loss_by_client))) if loss_by_client else float("nan")

        wall = time.time() - self._t0
        row = {
            "round": int(server_round),
            "mean_cer": mean_cer,
            "max_cer": max_cer,
            "cvar90_cer": cvar90,
            "p95_cer": p95v,
            "mean_loss": mean_loss,
            "clients_eval": int(len(cer_by_client)),
            "wall_time_s": float(wall),
        }

        print(
            f"[Round {server_round}] meanCER={mean_cer:.4f} maxCER={max_cer:.4f} "
            f"CVaR{int(self.cvar_alpha*100)}={cvar90:.4f} p95={p95v:.4f} meanLoss={mean_loss:.4f}"
        )

        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writerow(row)

        # Return the aggregated loss and metrics
        loss_aggregated = mean_loss
        return loss_aggregated, row



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="../ML_SUPERB/fleurs")
    ap.add_argument("--langs", type=str, default="")
    ap.add_argument("--num_langs", type=int, default=8)
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

    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--clients_per_round", type=int, default=4)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ft_frac", type=float, default=0.30)

    ap.add_argument("--max_train_hours", type=float, default=0.50)
    ap.add_argument("--max_valid_hours", type=float, default=0.10)
    ap.add_argument("--max_audio_s", type=float, default=16.0)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--local_steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--eval_utterances", type=int, default=128)
    ap.add_argument("--log_csv", type=str, default="runs/fleurs_local_hyperlora.csv")

    ap.add_argument("--num_cpus_per_client", type=float, default=2.0)
    ap.add_argument("--num_gpus_per_client", type=float, default=1.0 if torch.cuda.is_available() else 0.0)
    ap.add_argument("--unbounded_budget_baseline", action="store_true",
               help="Remove per-round caps: full participation, full train split, full validation eval.")
    ap.add_argument("--cvar_alpha", type=float, default=0.7)

    args = ap.parse_args()

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

    if args.langs.strip():
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    else:
        discovered = splits_mod.discover_langs(args.data_root, exclude=exclude)
        langs = discovered[: max(1, args.num_langs)]
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
            json.dump({"data_root": args.data_root, "langs": ok_langs}, f, indent=2)

    if args.build_manifests_only:
        return

    if not ok_langs:
        raise RuntimeError("No usable languages after manifest build")

    all_langs = ok_langs

    if args.unbounded_budget_baseline:
        args.clients_per_round = len(all_langs)   # full participation
        args.ft_frac = 1.0                        # use all train rows for fine-tuning
        args.max_train_hours = 0.0                # 0 means no cap in your iterable
        args.max_valid_hours = 0.0
        args.eval_utterances = 0                  # 0 means evaluate full validation
        args.local_steps = 0                      # 0 means no step cap (see patch below)


    tmp_model = Wav2Vec2ForCTC.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    hidden, n_layer = _find_hidden_size_and_n_layers(tmp_model)
    tmp_hyper = HyperLoRA(n_lang=len(all_langs), n_layer=n_layer, hidden_size=hidden).to("cpu")
    for p in tmp_hyper.parameters():
        p.requires_grad = True
    trainable_names, init_arrays = get_trainable_weights(tmp_hyper)
    init_parameters = ndarrays_to_parameters(init_arrays)

    def client_fn(cid: str):
        i = int(cid)
        lang_id = all_langs[i]
        mms_target = lang_to_mms.get(lang_id, None)

        return LocalFleursHyperClient(
            cid=cid,
            lang_id=lang_id,
            lang_dir=os.path.join(args.data_root, lang_id),
            lang_idx=i,
            all_langs=all_langs,
            mms_target_lang=mms_target,   # <-- add this
            model_id=args.model_id,
            device=args.device,
            fp16=fp16,
            use_mms_adapters=use_mms_adapters,
            batch_size=args.batch_size,
            local_steps=args.local_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            ft_frac=args.ft_frac,
            max_train_hours=args.max_train_hours,
            max_valid_hours=args.max_valid_hours,
            max_audio_s=args.max_audio_s,
            eval_utterances=args.eval_utterances,
            trainable_names=trainable_names,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )


    strategy = HyperFedAvg(
        trainable_names=trainable_names,
        log_csv=args.log_csv,
        fraction_fit=min(1.0, args.clients_per_round / max(1, len(all_langs))),
        fraction_evaluate=1.0,
        min_fit_clients=min(args.clients_per_round, len(all_langs)),
        min_available_clients=len(all_langs),
        min_evaluate_clients=len(all_langs),
        initial_parameters=init_parameters,
        cvar_alpha=args.cvar_alpha,

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


if __name__ == "__main__":
    main()
