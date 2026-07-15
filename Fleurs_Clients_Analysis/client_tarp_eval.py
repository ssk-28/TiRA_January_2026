from __future__ import annotations

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
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import torchaudio

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitIns, EvaluateIns
from flwr.server.strategy import FedAvg

from transformers import AutoProcessor, Wav2Vec2ForCTC

import splits as splits_mod


###############################################################################
# Text normalization + CER/WER
###############################################################################

_PARENS = re.compile(r"\([^)]*\)")
_BAD = re.compile(r"[^\w\s'’]+", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_transcript(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
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


###############################################################################
# MMS target lang map
###############################################################################

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
    kept: List[str] = []
    mapping: Dict[str, str] = {}
    for l in data_langs:
        t = resolve_mms_target_lang(proc.tokenizer, l)
        if t is None:
            print(f"[drop] lang {l}: no MMS target_lang mapping", flush=True)
            continue
        kept.append(l)
        mapping[l] = t
    return kept, mapping


###############################################################################
# Numeric helpers
###############################################################################

def bytes_of_parameters(p: fl.common.Parameters) -> int:
    if p is None or getattr(p, "tensors", None) is None:
        return 0
    return int(sum(len(t) for t in p.tensors))


def l2_delta(nds_a: List[np.ndarray], nds_b: List[np.ndarray]) -> float:
    if nds_a is None or nds_b is None or len(nds_a) != len(nds_b):
        return float("nan")
    tot = 0.0
    for a, b in zip(nds_a, nds_b):
        d = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
        tot += float(np.sum(d * d))
    return float(math.sqrt(tot))


def _finite_list(values: List[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


def pctl(values: List[float], q: float) -> float:
    xs = _finite_list(values)
    if not xs:
        return float("nan")
    xs.sort()
    idx = int(round((q / 100.0) * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return float(xs[idx])


def tail_mean(values: List[float], tail_frac: float, higher_is_worse: bool) -> float:
    xs = _finite_list(values)
    if not xs:
        return float("nan")
    xs.sort()
    k = max(1, int(math.ceil(tail_frac * len(xs))))
    tail = xs[-k:] if higher_is_worse else xs[:k]
    return float(sum(tail) / len(tail))


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb + eps))


def _avg_ndarrays(weighted: List[Tuple[float, np.ndarray]]) -> np.ndarray:
    sw = sum(w for w, _ in weighted)
    if sw <= 0:
        return weighted[0][1].copy()
    out = np.zeros_like(weighted[0][1], dtype=np.float32)
    for w, a in weighted:
        out += (float(w) / float(sw)) * a.astype(np.float32)
    return out


def _flatten_params(deltas: List[np.ndarray], idxs: List[int]) -> np.ndarray:
    if not idxs:
        return np.zeros((1,), dtype=np.float32)
    parts = []
    for i in idxs:
        parts.append(deltas[i].reshape(-1).astype(np.float32))
    return np.concatenate(parts, axis=0)


###############################################################################
# Strict CSV logging
###############################################################################

class StrictCSVLogger:
    def __init__(self, path: str, fieldnames: List[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._ensure_header()

    def _read_header(self) -> Optional[List[str]]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                return next(r, None)
        except Exception:
            return None

    def _ensure_header(self) -> None:
        if os.path.exists(self.path):
            hdr = self._read_header()
            if hdr != self.fieldnames:
                root, ext = os.path.splitext(self.path)
                ts = time.strftime("%Y%m%d_%H%M%S")
                new_path = f"{root}_restarted_{ts}{ext if ext else '.csv'}"
                print(f"[warn] CSV header mismatch. Writing to: {new_path}", flush=True)
                self.path = new_path

        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(self.fieldnames)

    def write_row(self, row: Dict[str, Any]) -> None:
        out = []
        for k in self.fieldnames:
            v = row.get(k, float("nan"))
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            out.append(v)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(out)
            f.flush()


###############################################################################
# HyperLoRA modules (dual-head)
###############################################################################

class HyperContext:
    _lang_idx: Optional[torch.LongTensor] = None
    _use_tail_head: bool = False

    @classmethod
    def set_lang_idx(cls, lang_idx: torch.LongTensor) -> None:
        cls._lang_idx = lang_idx

    @classmethod
    def get_lang_idx(cls) -> torch.LongTensor:
        if cls._lang_idx is None:
            raise RuntimeError("HyperContext lang_idx not set")
        return cls._lang_idx

    @classmethod
    def set_use_tail_head(cls, use_tail_head: bool) -> None:
        cls._use_tail_head = bool(use_tail_head)

    @classmethod
    def use_tail_head(cls) -> bool:
        return bool(cls._use_tail_head)


class DualHeadHyperLoRA(nn.Module):
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
        tail_r: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_lang = n_lang
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.r_shared = int(r)
        self.r_tail = int(tail_r) if tail_r is not None else int(r)
        self.alpha = float(alpha)

        self.scaling_shared = self.alpha / float(self.r_shared)
        self.scaling_tail = self.alpha / float(self.r_tail)

        self.lang_emb = nn.Embedding(n_lang, lang_emb_dim)
        self.layer_emb = nn.Embedding(n_layer, layer_emb_dim)

        in_dim = lang_emb_dim + layer_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim),
            nn.ReLU(),
        )

        self.shared_to_A = nn.Linear(code_dim, self.r_shared * hidden_size, bias=False)
        self.shared_to_B = nn.Linear(code_dim, hidden_size * self.r_shared, bias=False)

        self.tail_to_A = nn.Linear(code_dim, self.r_tail * hidden_size, bias=False)
        self.tail_to_B = nn.Linear(code_dim, hidden_size * self.r_tail, bias=False)

        nn.init.normal_(self.shared_to_A.weight, std=0.01)
        nn.init.normal_(self.shared_to_B.weight, std=0.01)
        nn.init.normal_(self.tail_to_A.weight, std=0.01)
        nn.init.normal_(self.tail_to_B.weight, std=0.01)

    def gen(self, layer_id: int, lang_idx: torch.LongTensor, head: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
        le = self.lang_emb(lang_idx)
        lid = torch.full_like(lang_idx, fill_value=layer_id)
        pe = self.layer_emb(lid)
        code = self.mlp(torch.cat([le, pe], dim=-1))

        if head == "tail":
            A = self.tail_to_A(code).view(-1, self.r_tail, self.hidden_size) / math.sqrt(self.hidden_size)
            B = self.tail_to_B(code).view(-1, self.hidden_size, self.r_tail)
            return A, B, self.scaling_tail

        A = self.shared_to_A(code).view(-1, self.r_shared, self.hidden_size) / math.sqrt(self.hidden_size)
        B = self.shared_to_B(code).view(-1, self.hidden_size, self.r_shared)
        return A, B, self.scaling_shared


class HyperLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, hyper: DualHeadHyperLoRA, layer_id: int) -> None:
        super().__init__()
        self.base = base
        self.hyper = hyper
        self.layer_id = layer_id
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        lang_idx = HyperContext.get_lang_idx()
        head = "tail" if HyperContext.use_tail_head() else "shared"

        if x.dim() == 3:
            bsz = x.size(0)
            if lang_idx.numel() == 1:
                lang_idx = lang_idx.expand(bsz)
            A, Bm, sc = self.hyper.gen(self.layer_id, lang_idx, head=head)
            xA = torch.einsum("bth,brh->btr", x, A)
            dY = torch.einsum("btr,bhr->bth", xA, Bm) * sc
            return y0 + dY

        if x.dim() == 2:
            bsz = x.size(0)
            if lang_idx.numel() == 1:
                lang_idx = lang_idx.expand(bsz)
            A, Bm, sc = self.hyper.gen(self.layer_id, lang_idx, head=head)
            xA = torch.einsum("bh,brh->br", x, A)
            dY = torch.einsum("br,bhr->bh", xA, Bm) * sc
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


def inject_hyperlora(model: nn.Module, hyper: DualHeadHyperLoRA, targets=("q_proj", "k_proj", "v_proj", "out_proj")) -> None:
    enc = model.wav2vec2.encoder
    for lid, layer in enumerate(enc.layers):
        attn = getattr(layer, "attention", None)
        if attn is None:
            continue
        for t in targets:
            lin = getattr(attn, t, None)
            if isinstance(lin, nn.Linear) and lin.in_features == lin.out_features == hyper.hidden_size:
                setattr(attn, t, HyperLoRALinear(lin, hyper=hyper, layer_id=lid))


###############################################################################
# Data loading
###############################################################################

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

    if max_audio_s and max_audio_s > 0:
        max_len = int(max_audio_s * sr)
        if wav.numel() > max_len:
            wav = wav[:max_len]

    if wav.numel() == 0:
        raise ValueError("empty after truncation")

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
    ) -> None:
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
            key = (p, uid)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"path": p, "text": tx, "uid": uid})

        rng = random.Random(self.seed + (0 if split == "train" else 17))
        rng.shuffle(rows)

        def is_ft_row(rr):
            return _ft_bucket(self.lang_id, rr["path"], rr["uid"], self.seed) < self.ft_frac

        if split == "validation":
            filtered = rows
        else:
            if self.want_ft:
                filtered = [rr for rr in rows if is_ft_row(rr)]
                if not filtered:
                    print(f"[warn] {self.lang_id} split={self.split}: ft bucket empty, using full split", flush=True)
                    filtered = rows
            else:
                filtered = [rr for rr in rows if not is_ft_row(rr)]
                if not filtered:
                    filtered = rows

        if not filtered:
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


###############################################################################
# Collator + CTC filter
###############################################################################

class CTCCollator:
    def __init__(self, processor, pad_to_multiple_of: Optional[int] = None) -> None:
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        audios = [b["audio"] for b in batch]
        srs = [b["sr"] for b in batch]
        if len(set(srs)) != 1:
            raise ValueError("mixed sampling rates in batch")
        texts = [b["text"] for b in batch]

        inputs = self.processor.feature_extractor(
            audios,
            sampling_rate=srs[0],
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        tok = self.processor.tokenizer(texts, return_tensors="pt", padding=True)
        labels = tok.input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        out = {
            "input_values": inputs["input_values"],
            "attention_mask": inputs.get("attention_mask", None),
            "labels": labels,
        }
        if out["attention_mask"] is None:
            out["attention_mask"] = torch.ones_like(out["input_values"], dtype=torch.long)
        return out


def filter_ctc_batch(model: Wav2Vec2ForCTC, batch: Dict[str, torch.Tensor]) -> Tuple[Optional[Dict[str, torch.Tensor]], int, int]:
    attn = batch["attention_mask"]
    labels = batch["labels"]

    with torch.no_grad():
        input_lengths = model._get_feat_extract_output_lengths(attn.sum(-1))

    label_lengths = (labels != -100).sum(-1)
    keep = label_lengths <= input_lengths

    kept = int(keep.sum().item())
    total = int(keep.numel())

    if kept == 0:
        return None, kept, total

    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor) and v.size(0) == keep.size(0):
            batch[k] = v[keep]
    return batch, kept, total


###############################################################################
# Trainable weights helpers
###############################################################################

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


###############################################################################
# Flower client
###############################################################################

class LocalFleursTARPClient(fl.client.NumPyClient):
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
        mms_target_lang: Optional[str] = None,
    ) -> None:
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

        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)
        self.model.config.ctc_zero_infinity = True

        if self.use_mms_adapters:
            if not self.mms_target_lang:
                raise ValueError(f"Missing MMS target lang for {self.lang_id}")
            self.processor.tokenizer.set_target_lang(self.mms_target_lang)
            self.model.load_adapter(self.mms_target_lang)

        for p in self.model.parameters():
            p.requires_grad = False

        hidden, n_layer = _find_hidden_size_and_n_layers(self.model)
        self.hyper = DualHeadHyperLoRA(
            n_lang=len(self.all_langs),
            n_layer=n_layer,
            hidden_size=hidden,
            r=8,
            tail_r=8,
        ).to(self.device)

        inject_hyperlora(self.model, self.hyper)
        for p in self.hyper.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            [p for p in self.hyper.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.fp16 and self.device.type == "cuda"))

        self._make_loaders(max_audio_s=self.max_audio_s)

    def _make_loaders(self, max_audio_s: float) -> None:
        train_ds = LocalFleursIterable(
            lang_id=self.lang_id,
            lang_dir=self.lang_dir,
            split="train",
            ft_frac=self.ft_frac,
            want_ft=True,
            max_hours=self.max_train_hours,
            max_audio_s=max_audio_s,
            seed=self.seed,
        )
        valid_ds = LocalFleursIterable(
            lang_id=self.lang_id,
            lang_dir=self.lang_dir,
            split="validation",
            ft_frac=1.0,
            want_ft=True,
            max_hours=self.max_valid_hours,
            max_audio_s=max_audio_s,
            seed=self.seed + 17,
        )
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, collate_fn=CTCCollator(self.processor))
        self.valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, collate_fn=CTCCollator(self.processor))
        self._loader_max_audio_s = float(max_audio_s)

    def get_parameters(self, config):
        _, arrays = get_trainable_weights(self.hyper)
        return arrays

    def set_parameters(self, parameters):
        set_trainable_weights(self.hyper, self.trainable_names, parameters)

    def _fit_one_pass(self, parameters, use_tail_head: bool) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.set_parameters(parameters)

        HyperContext.set_lang_idx(torch.tensor([self.lang_idx], device=self.device, dtype=torch.long))
        HyperContext.set_use_tail_head(bool(use_tail_head))

        self.model.train()
        self.hyper.train()

        step = 0
        total_loss = 0.0
        seen_examples = 0

        skip_ctc_batches = 0
        skip_nonfinite = 0
        ctc_kept_total = 0
        ctc_total_total = 0

        t0 = time.time()

        for batch in self.train_loader:
            if self.local_steps > 0 and step >= self.local_steps:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            batch, kept, total = filter_ctc_batch(self.model, batch)
            ctc_kept_total += kept
            ctc_total_total += total
            if batch is None:
                skip_ctc_batches += 1
                continue

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.fp16 and self.device.type == "cuda")):
                out = self.model(**batch)
                loss = out.loss

            if loss is None or (not torch.isfinite(loss)):
                skip_nonfinite += 1
                continue

            self.scaler.scale(loss).backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.hyper.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = int(batch["input_values"].shape[0])
            seen_examples += bs
            total_loss += float(loss.detach().item())
            step += 1

        dt = time.time() - t0
        avg_loss = (total_loss / step) if step > 0 else float("nan")
        effective = 1 if step > 0 else 0
        new_params = self.get_parameters(config={}) if step > 0 else parameters
        num_ex = max(1, int(seen_examples))

        metrics = {
            "lang_id": self.lang_id,
            "lang_idx": int(self.lang_idx),
            "train_steps": int(step),
            "train_examples": int(seen_examples),
            "effective_fit": int(effective),
            "train_loss": float(avg_loss),
            "fit_time_s": float(dt),
            "use_tail_head": int(bool(use_tail_head)),
            "skip_ctc_batches": int(skip_ctc_batches),
            "skip_nonfinite": int(skip_nonfinite),
            "ctc_kept_total": int(ctc_kept_total),
            "ctc_total_total": int(ctc_total_total),
            "max_audio_s_used": float(self._loader_max_audio_s),
        }
        return new_params, num_ex, metrics

    def fit(self, parameters, config):
        use_tail = bool(int(config.get("use_tail_head", 0)))

        new_params, num_ex, metrics = self._fit_one_pass(parameters, use_tail_head=use_tail)

        if int(metrics.get("train_steps", 0)) == 0 and float(self._loader_max_audio_s) > 0.0:
            ctc_total = int(metrics.get("ctc_total_total", 0))
            ctc_kept = int(metrics.get("ctc_kept_total", 0))
            if ctc_total > 0 and ctc_kept == 0:
                self._make_loaders(max_audio_s=0.0)
                new_params, num_ex, metrics2 = self._fit_one_pass(parameters, use_tail_head=use_tail)
                metrics2["fallback_reran"] = 1
                return new_params, num_ex, metrics2

        metrics["fallback_reran"] = 0
        return new_params, num_ex, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        use_tail = bool(int(config.get("use_tail_head", 0)))

        HyperContext.set_lang_idx(torch.tensor([self.lang_idx], device=self.device, dtype=torch.long))
        HyperContext.set_use_tail_head(bool(use_tail))

        self.model.eval()
        self.hyper.eval()

        total_loss = 0.0
        n_batches = 0

        skip_ctc_batches = 0
        skip_nonfinite = 0
        ctc_kept_total = 0
        ctc_total_total = 0

        cer_list: List[float] = []
        tot_S = tot_D = tot_I = tot_C = tot_N = 0
        sent_ok = 0
        seen_utts = 0

        with torch.inference_mode():
            for batch in self.valid_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                batch, kept, total = filter_ctc_batch(self.model, batch)
                ctc_kept_total += kept
                ctc_total_total += total
                if batch is None:
                    skip_ctc_batches += 1
                    continue

                out = self.model(**batch)
                loss = out.loss
                if loss is None or (not torch.isfinite(loss)):
                    skip_nonfinite += 1
                    continue

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
                    wv, wacc, sacc, (S, D, I, C, N) = wer_and_acc(ref, hyp)
                    tot_S += S
                    tot_D += D
                    tot_I += I
                    tot_C += C
                    tot_N += N
                    sent_ok += int(sacc)
                    seen_utts += 1
                    if self.eval_utterances and self.eval_utterances > 0 and seen_utts >= self.eval_utterances:
                        break

                if self.eval_utterances and self.eval_utterances > 0 and seen_utts >= self.eval_utterances:
                    break

        avg_loss = (total_loss / n_batches) if n_batches > 0 else float("nan")
        mean_cer = float(np.mean(cer_list)) if cer_list else float("nan")
        max_cer = float(np.max(cer_list)) if cer_list else float("nan")

        corpus_wer = (tot_S + tot_D + tot_I) / max(1, tot_N)
        corpus_word_correct_acc = tot_C / max(1, tot_N)
        corpus_sent_acc = sent_ok / max(1, seen_utts)

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
            "use_tail_head": int(bool(use_tail)),
            "skip_ctc_batches": int(skip_ctc_batches),
            "skip_nonfinite": int(skip_nonfinite),
            "ctc_kept_total": int(ctc_kept_total),
            "ctc_total_total": int(ctc_total_total),
            "max_audio_s_used": float(self._loader_max_audio_s),
        }
        return float(avg_loss), int(max(1, seen_utts)), metrics


###############################################################################
# Strategy: TARP aggregation and logging (round + per-language)
###############################################################################

class TARPFedAvg(FedAvg):
    def __init__(
        self,
        trainable_names: List[str],
        all_langs: List[str],
        log_csv: str,
        log_csv_per_lang: str,
        tail_fracs: List[float],
        route_tail_frac: float,
        route_cos_thr: float,
        route_warmup_rounds: int,
        route_risk_beta: float,
        route_risk_metric: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainable_names = list(trainable_names)
        self.all_langs = list(all_langs)

        self.tail_fracs = [float(x) for x in tail_fracs]
        self.route_tail_frac = float(route_tail_frac)
        self.route_cos_thr = float(route_cos_thr)
        self.route_warmup_rounds = int(route_warmup_rounds)
        self.route_risk_beta = float(route_risk_beta)
        self.route_risk_metric = str(route_risk_metric)

        self._risk_ema: Dict[int, float] = {}
        self._tail_lang_idxs: List[int] = []
        self._round_start_ts: Optional[float] = None

        self._sent_ndarrays_this_round: Optional[List[np.ndarray]] = None
        self._payload_bytes_this_round: float = float("nan")
        self._clients_fit_configured_this_round: int = 0

        self._fit_round_stats: Dict[int, Dict[str, float]] = {}
        self._per_lang_fit_round: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._round_cos_stats: Dict[str, float] = {}

        self._last_parameters = kwargs.get("initial_parameters", None)

        self._name_to_i = {n: i for i, n in enumerate(self.trainable_names)}
        self._idx_shared_head = [i for n, i in self._name_to_i.items() if n.startswith("shared_to_")]
        self._idx_tail_head = [i for n, i in self._name_to_i.items() if n.startswith("tail_to_")]
        self._idx_lang_emb = [self._name_to_i["lang_emb.weight"]] if "lang_emb.weight" in self._name_to_i else []

        base_fields = [
            "round", "wall_time_s",
            "clients_fit_configured", "clients_fit_received",
            "clients_fit_effective", "clients_fit_noop",
            "fit_sum_train_steps", "fit_sum_train_examples",
            "fit_sum_skip_ctc_batches", "fit_sum_skip_nonfinite",
            "fit_mean_train_loss", "fit_p95_train_loss",
            "fit_mean_delta_l2", "fit_max_delta_l2",
            "payload_bytes", "round_download_bytes", "round_upload_bytes",
            "clients_eval",
            "mean_loss", "mean_cer", "max_cer", "p95_cer",
            "mean_wer", "max_wer", "p95_wer",
            "mean_word_correct_acc", "min_word_correct_acc", "p05_word_correct_acc",
            "mean_sent_acc", "min_sent_acc", "p05_sent_acc",
            "route_tail_frac", "route_cos_thr", "route_risk_beta", "route_risk_metric_code",
            "tail_risk_ema_mean", "tail_risk_ema_max", "tail_set_size",
            "merged_tail_into_shared", "kept_tail_in_tailhead", "mean_tail_cos", "min_tail_cos",
        ]
        tail_fields = []
        for tf in self.tail_fracs:
            p = int(round(tf * 100))
            tail_fields += [f"tail{p}_cer", f"tail{p}_wer", f"tail{p}_word_correct_acc", f"tail{p}_sent_acc"]

        self._csv_fields = base_fields + tail_fields
        self._logger = StrictCSVLogger(log_csv, self._csv_fields)

        per_lang_fields = [
            "round",
            "lang_id",
            "lang_idx",
            "used_tail_head",
            "in_tail_set_next",
            "risk_raw",
            "risk_ema",
            "val_loss",
            "val_cer_mean",
            "val_cer_max",
            "val_wer",
            "val_word_correct_acc",
            "val_sent_acc",
            "eval_utts",
            "eval_word_tokens",
            "skip_ctc_batches_eval",
            "skip_nonfinite_eval",
            "ctc_kept_total_eval",
            "ctc_total_total_eval",
            "max_audio_s_used_eval",
            "train_loss",
            "train_steps",
            "train_examples",
            "fit_time_s",
            "effective_fit",
            "skip_ctc_batches_fit",
            "skip_nonfinite_fit",
            "ctc_kept_total_fit",
            "ctc_total_total_fit",
            "max_audio_s_used_fit",
            "delta_l2",
        ]
        self._per_lang_logger = StrictCSVLogger(log_csv_per_lang, per_lang_fields)
        self._per_lang_fields = per_lang_fields

    def _cfg_use_tail_head(self, server_round: int, lang_idx: int) -> bool:
        if server_round <= self.route_warmup_rounds:
            return False
        return int(lang_idx) in set(self._tail_lang_idxs)

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start_ts = time.time()
        cfgs = super().configure_fit(server_round, parameters, client_manager)

        self._clients_fit_configured_this_round = int(len(cfgs))
        self._payload_bytes_this_round = float(bytes_of_parameters(parameters))
        try:
            self._sent_ndarrays_this_round = parameters_to_ndarrays(parameters)
        except Exception:
            self._sent_ndarrays_this_round = None

        out = []
        for client, fit_ins in cfgs:
            cid_str = getattr(client, "cid", None)
            use_tail = False
            try:
                if cid_str is not None:
                    use_tail = self._cfg_use_tail_head(server_round, int(cid_str))
            except Exception:
                use_tail = False
            new_cfg = dict(fit_ins.config)
            new_cfg["use_tail_head"] = int(use_tail)
            out.append((client, FitIns(fit_ins.parameters, new_cfg)))
        return out

    def configure_evaluate(self, server_round, parameters, client_manager):
        cfgs = super().configure_evaluate(server_round, parameters, client_manager)
        out = []
        for client, ev_ins in cfgs:
            cid_str = getattr(client, "cid", None)
            use_tail = False
            try:
                if cid_str is not None:
                    use_tail = self._cfg_use_tail_head(server_round, int(cid_str))
            except Exception:
                use_tail = False
            new_cfg = dict(ev_ins.config)
            new_cfg["use_tail_head"] = int(use_tail)
            out.append((client, EvaluateIns(ev_ins.parameters, new_cfg)))
        return out

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            print(f"[warn] Round {server_round}: no fit results", flush=True)
            return (self._last_parameters, {}) if self._last_parameters is not None else (None, {})

        def safe_float(x, default=float("nan")):
            try:
                return float(x)
            except Exception:
                return default

        def get_weight_effective(fit_res) -> Tuple[float, bool]:
            m = fit_res.metrics or {}
            eff_flag = m.get("effective_fit", None)
            if eff_flag is not None:
                effective = int(safe_float(eff_flag, 0.0)) == 1
            else:
                train_steps = int(safe_float(m.get("train_steps", 0), 0.0))
                train_examples = int(safe_float(m.get("train_examples", 0), 0.0))
                effective = (train_steps > 0) or (train_examples > 0)

            w = safe_float(getattr(fit_res, "num_examples", float("nan")), float("nan"))
            if not np.isfinite(w):
                w = float(safe_float(m.get("train_examples", 0), 0.0))

            if not effective:
                return 0.0, False
            if w < 0.0 or (not np.isfinite(w)):
                w = 0.0
            return float(w), True

        def weighted_avg_param(group, j, fallback):
            weighted = [(w, nds[j]) for (w, nds, _) in group if w > 0.0]
            sw = sum(w for w, _ in weighted)
            if sw <= 0.0:
                return fallback.copy()
            return _avg_ndarrays(weighted)

        clients_fit_received = int(len(results))
        eff_count = 0
        noop_count = 0

        fit_losses: List[float] = []
        delta_l2s: List[float] = []

        sum_train_steps = 0
        sum_train_examples = 0
        sum_skip_ctc_batches = 0
        sum_skip_nonfinite = 0

        sent_nds = self._sent_ndarrays_this_round
        payload_bytes = float(self._payload_bytes_this_round) if np.isfinite(self._payload_bytes_this_round) else float("nan")

        self._per_lang_fit_round[int(server_round)] = {}

        for _, fit_res in results:
            m = fit_res.metrics or {}
            lang_idx = int(safe_float(m.get("lang_idx", -1), -1.0))

            w, effective = get_weight_effective(fit_res)
            if effective:
                eff_count += 1
            else:
                noop_count += 1

            sum_train_steps += int(safe_float(m.get("train_steps", 0), 0.0))
            sum_train_examples += int(safe_float(m.get("train_examples", 0), 0.0))
            sum_skip_ctc_batches += int(safe_float(m.get("skip_ctc_batches", 0), 0.0))
            sum_skip_nonfinite += int(safe_float(m.get("skip_nonfinite", 0), 0.0))

            tl = safe_float(m.get("train_loss", float("nan")), float("nan"))
            if np.isfinite(tl):
                fit_losses.append(float(tl))

            dlt = float("nan")
            if sent_nds is not None and w > 0.0:
                try:
                    client_nds = parameters_to_ndarrays(fit_res.parameters)
                    d = l2_delta(client_nds, sent_nds)
                    if np.isfinite(d):
                        delta_l2s.append(float(d))
                        dlt = float(d)
                except Exception:
                    pass

            if lang_idx >= 0:
                self._per_lang_fit_round[int(server_round)][lang_idx] = {
                    "train_loss": safe_float(m.get("train_loss", float("nan"))),
                    "train_steps": int(safe_float(m.get("train_steps", 0), 0.0)),
                    "train_examples": int(safe_float(m.get("train_examples", 0), 0.0)),
                    "fit_time_s": safe_float(m.get("fit_time_s", float("nan"))),
                    "effective_fit": int(safe_float(m.get("effective_fit", 0), 0.0)),
                    "use_tail_head": int(safe_float(m.get("use_tail_head", 0), 0.0)),
                    "skip_ctc_batches_fit": int(safe_float(m.get("skip_ctc_batches", 0), 0.0)),
                    "skip_nonfinite_fit": int(safe_float(m.get("skip_nonfinite", 0), 0.0)),
                    "ctc_kept_total_fit": int(safe_float(m.get("ctc_kept_total", 0), 0.0)),
                    "ctc_total_total_fit": int(safe_float(m.get("ctc_total_total", 0), 0.0)),
                    "max_audio_s_used_fit": safe_float(m.get("max_audio_s_used", float("nan"))),
                    "delta_l2": dlt,
                }

        fit_mean_train_loss = float(np.mean(fit_losses)) if fit_losses else float("nan")
        fit_p95_train_loss = float(np.percentile(fit_losses, 95)) if fit_losses else float("nan")
        fit_mean_delta_l2 = float(np.mean(delta_l2s)) if delta_l2s else float("nan")
        fit_max_delta_l2 = float(np.max(delta_l2s)) if delta_l2s else float("nan")

        if np.isfinite(payload_bytes):
            round_download_bytes = float(payload_bytes * float(self._clients_fit_configured_this_round))
            round_upload_bytes = float(payload_bytes * float(clients_fit_received))
        else:
            round_download_bytes = float("nan")
            round_upload_bytes = float("nan")

        self._fit_round_stats[int(server_round)] = {
            "clients_fit_configured": float(self._clients_fit_configured_this_round),
            "clients_fit_received": float(clients_fit_received),
            "clients_fit_effective": float(eff_count),
            "clients_fit_noop": float(noop_count),
            "fit_sum_train_steps": float(sum_train_steps),
            "fit_sum_train_examples": float(sum_train_examples),
            "fit_sum_skip_ctc_batches": float(sum_skip_ctc_batches),
            "fit_sum_skip_nonfinite": float(sum_skip_nonfinite),
            "fit_mean_train_loss": float(fit_mean_train_loss),
            "fit_p95_train_loss": float(fit_p95_train_loss),
            "fit_mean_delta_l2": float(fit_mean_delta_l2),
            "fit_max_delta_l2": float(fit_max_delta_l2),
            "payload_bytes": float(payload_bytes),
            "round_upload_bytes": float(round_upload_bytes),
            "round_download_bytes": float(round_download_bytes),
        }

        if self._last_parameters is not None:
            global_nds = parameters_to_ndarrays(self._last_parameters)
        else:
            global_nds = parameters_to_ndarrays(results[0][1].parameters)

        shared_group = []
        tail_group = []

        for _, fit_res in results:
            m = fit_res.metrics or {}
            lang_idx = int(safe_float(m.get("lang_idx", -1), -1.0))
            w, effective = get_weight_effective(fit_res)
            if (not effective) or (w <= 0.0):
                continue
            nds = parameters_to_ndarrays(fit_res.parameters)
            used_tail = int(safe_float(m.get("use_tail_head", 0), 0.0)) == 1
            if used_tail:
                tail_group.append((w, nds, lang_idx))
            else:
                shared_group.append((w, nds, lang_idx))

        if not shared_group and not tail_group:
            return (self._last_parameters, {}) if self._last_parameters is not None else (ndarrays_to_parameters(global_nds), {})

        ref_vec = None
        if self._idx_shared_head and shared_group:
            deltas = []
            for w, nds, _ in shared_group:
                d_all = [nds[i] - global_nds[i] for i in range(len(nds))]
                dv = _flatten_params(d_all, self._idx_shared_head)
                if np.isfinite(dv).all():
                    deltas.append((w, dv))
            ref_vec = _avg_ndarrays(deltas) if deltas else None

        merged_tail = []
        kept_tail = []
        tail_cos = []

        if ref_vec is None:
            kept_tail = tail_group[:]
            tail_cos = [0.0 for _ in tail_group]
        else:
            for w, nds, lang_idx in tail_group:
                d_all = [nds[i] - global_nds[i] for i in range(len(nds))]
                dv_tail = _flatten_params(d_all, self._idx_tail_head)
                c = cosine_sim(dv_tail, ref_vec)
                tail_cos.append(c)
                if c >= self.route_cos_thr:
                    merged_tail.append((w, nds, lang_idx))
                else:
                    kept_tail.append((w, nds, lang_idx))

        self._round_cos_stats = {
            "merged_tail_into_shared": float(len(merged_tail)),
            "kept_tail_in_tailhead": float(len(kept_tail)),
            "mean_tail_cos": float(sum(tail_cos) / max(1, len(tail_cos))) if tail_cos else float("nan"),
            "min_tail_cos": float(min(tail_cos)) if tail_cos else float("nan"),
        }

        n_params = len(global_nds)
        new_nds = [a.copy() for a in global_nds]

        everyone = shared_group + tail_group
        non_head_idxs = [i for i in range(n_params) if (i not in self._idx_shared_head and i not in self._idx_tail_head)]

        if everyone:
            for j in non_head_idxs:
                new_nds[j] = weighted_avg_param(everyone, j, fallback=global_nds[j])

        if self._idx_shared_head:
            group_shared_plus = shared_group + merged_tail
            for j in self._idx_shared_head:
                new_nds[j] = weighted_avg_param(group_shared_plus, j, fallback=global_nds[j])

        if self._idx_tail_head:
            if kept_tail:
                for j in self._idx_tail_head:
                    new_nds[j] = weighted_avg_param(kept_tail, j, fallback=global_nds[j])
            else:
                for j in self._idx_tail_head:
                    new_nds[j] = global_nds[j].copy()

        if self._idx_lang_emb:
            li = self._idx_lang_emb[0]
            lang_emb = new_nds[li].copy()
            for _, fit_res in results:
                m = fit_res.metrics or {}
                lang_idx = int(safe_float(m.get("lang_idx", -1), -1.0))
                w, effective = get_weight_effective(fit_res)
                if (not effective) or (w <= 0.0):
                    continue
                try:
                    cnds = parameters_to_ndarrays(fit_res.parameters)
                except Exception:
                    continue
                if 0 <= lang_idx < lang_emb.shape[0]:
                    lang_emb[lang_idx] = cnds[li][lang_idx]
            new_nds[li] = lang_emb

        params_fixed = ndarrays_to_parameters(new_nds)
        self._last_parameters = params_fixed
        return params_fixed, {}

    def aggregate_evaluate(self, server_round: int, results, failures):
        wall_s = float(time.time() - self._round_start_ts) if self._round_start_ts is not None else float("nan")
        fit_stats = self._fit_round_stats.get(int(server_round), {})

        cer_by_client: List[float] = []
        wer_by_client: List[float] = []
        wacc_by_client: List[float] = []
        sacc_by_client: List[float] = []
        loss_by_client: List[float] = []

        cer_map: Dict[int, float] = {}
        loss_map: Dict[int, float] = {}

        eval_rows: Dict[int, Dict[str, Any]] = {}

        if results:
            for _, eval_res in results:
                m = eval_res.metrics or {}
                lang_idx = int(m.get("lang_idx", -1))
                lang_id = str(m.get("lang_id", self.all_langs[lang_idx] if 0 <= lang_idx < len(self.all_langs) else ""))
                used_tail = int(m.get("use_tail_head", 0))

                lv = float(eval_res.loss) if eval_res.loss is not None else float("nan")
                if np.isfinite(lv):
                    loss_by_client.append(lv)
                    if lang_idx >= 0:
                        loss_map[lang_idx] = lv

                vcer = float(m.get("val_cer_mean", float("nan")))
                if np.isfinite(vcer):
                    cer_by_client.append(vcer)
                    if lang_idx >= 0:
                        cer_map[lang_idx] = vcer

                vwer = float(m.get("val_wer", float("nan")))
                vwacc = float(m.get("val_word_correct_acc", float("nan")))
                vsacc = float(m.get("val_sent_acc", float("nan")))
                if np.isfinite(vwer):
                    wer_by_client.append(vwer)
                if np.isfinite(vwacc):
                    wacc_by_client.append(vwacc)
                if np.isfinite(vsacc):
                    sacc_by_client.append(vsacc)

                eval_rows[lang_idx] = {
                    "round": int(server_round),
                    "lang_id": lang_id,
                    "lang_idx": int(lang_idx),
                    "used_tail_head": int(used_tail),
                    "val_loss": float(m.get("val_loss", lv)),
                    "val_cer_mean": float(m.get("val_cer_mean", float("nan"))),
                    "val_cer_max": float(m.get("val_cer_max", float("nan"))),
                    "val_wer": float(m.get("val_wer", float("nan"))),
                    "val_word_correct_acc": float(m.get("val_word_correct_acc", float("nan"))),
                    "val_sent_acc": float(m.get("val_sent_acc", float("nan"))),
                    "eval_utts": int(m.get("eval_utts", 0)),
                    "eval_word_tokens": int(m.get("eval_word_tokens", 0)),
                    "skip_ctc_batches_eval": int(m.get("skip_ctc_batches", 0)),
                    "skip_nonfinite_eval": int(m.get("skip_nonfinite", 0)),
                    "ctc_kept_total_eval": int(m.get("ctc_kept_total", 0)),
                    "ctc_total_total_eval": int(m.get("ctc_total_total", 0)),
                    "max_audio_s_used_eval": float(m.get("max_audio_s_used", float("nan"))),
                }

        mean_loss = float(np.mean(loss_by_client)) if loss_by_client else float("nan")
        mean_cer = float(np.mean(cer_by_client)) if cer_by_client else float("nan")
        max_cer = float(np.max(cer_by_client)) if cer_by_client else float("nan")
        p95_cer = pctl(cer_by_client, 95.0)

        mean_wer = float(np.mean(wer_by_client)) if wer_by_client else float("nan")
        max_wer = float(np.max(wer_by_client)) if wer_by_client else float("nan")
        p95_wer = pctl(wer_by_client, 95.0)

        mean_wacc = float(np.mean(wacc_by_client)) if wacc_by_client else float("nan")
        min_wacc = float(np.min(wacc_by_client)) if wacc_by_client else float("nan")
        p05_wacc = pctl(wacc_by_client, 5.0)

        mean_sacc = float(np.mean(sacc_by_client)) if sacc_by_client else float("nan")
        min_sacc = float(np.min(sacc_by_client)) if sacc_by_client else float("nan")
        p05_sacc = pctl(sacc_by_client, 5.0)

        risk_map: Dict[int, float] = {}
        if self.route_risk_metric == "loss":
            for idx, v in loss_map.items():
                if idx >= 0 and np.isfinite(v):
                    risk_map[idx] = v
        else:
            for idx, v in cer_map.items():
                if idx >= 0 and np.isfinite(v):
                    risk_map[idx] = v

        beta = float(self.route_risk_beta)
        for idx, r in risk_map.items():
            prev = self._risk_ema.get(idx, r)
            self._risk_ema[idx] = beta * float(prev) + (1.0 - beta) * float(r)

        if self._risk_ema and self.route_tail_frac > 0.0:
            items = [(idx, v) for idx, v in self._risk_ema.items() if idx >= 0 and np.isfinite(v)]
            items.sort(key=lambda x: x[1], reverse=True)
            k = max(1, int(math.ceil(self.route_tail_frac * len(items))))
            self._tail_lang_idxs = [idx for idx, _ in items[:k]]
            tail_vals = [v for _, v in items[:k]]
            tail_ema_mean = float(np.mean(tail_vals)) if tail_vals else float("nan")
            tail_ema_max = float(np.max(tail_vals)) if tail_vals else float("nan")
        else:
            self._tail_lang_idxs = []
            tail_ema_mean = float("nan")
            tail_ema_max = float("nan")

        row: Dict[str, Any] = {k: float("nan") for k in self._csv_fields}
        row["round"] = int(server_round)
        row["wall_time_s"] = float(wall_s)
        row["clients_eval"] = int(len(results) if results else 0)

        row["mean_loss"] = mean_loss
        row["mean_cer"] = mean_cer
        row["max_cer"] = max_cer
        row["p95_cer"] = p95_cer

        row["mean_wer"] = mean_wer
        row["max_wer"] = max_wer
        row["p95_wer"] = p95_wer

        row["mean_word_correct_acc"] = mean_wacc
        row["min_word_correct_acc"] = min_wacc
        row["p05_word_correct_acc"] = p05_wacc

        row["mean_sent_acc"] = mean_sacc
        row["min_sent_acc"] = min_sacc
        row["p05_sent_acc"] = p05_sacc

        row["route_tail_frac"] = float(self.route_tail_frac)
        row["route_cos_thr"] = float(self.route_cos_thr)
        row["route_risk_beta"] = float(self.route_risk_beta)
        row["route_risk_metric_code"] = float(0.0 if self.route_risk_metric == "cer" else 1.0)
        row["tail_risk_ema_mean"] = float(tail_ema_mean)
        row["tail_risk_ema_max"] = float(tail_ema_max)
        row["tail_set_size"] = float(len(self._tail_lang_idxs))

        row["merged_tail_into_shared"] = float(self._round_cos_stats.get("merged_tail_into_shared", 0.0))
        row["kept_tail_in_tailhead"] = float(self._round_cos_stats.get("kept_tail_in_tailhead", 0.0))
        row["mean_tail_cos"] = float(self._round_cos_stats.get("mean_tail_cos", float("nan")))
        row["min_tail_cos"] = float(self._round_cos_stats.get("min_tail_cos", float("nan")))

        for k2, v2 in fit_stats.items():
            row[k2] = float(v2)

        for tf in self.tail_fracs:
            p = int(round(tf * 100))
            row[f"tail{p}_cer"] = tail_mean(cer_by_client, tf, higher_is_worse=True)
            row[f"tail{p}_wer"] = tail_mean(wer_by_client, tf, higher_is_worse=True)
            row[f"tail{p}_word_correct_acc"] = tail_mean(wacc_by_client, tf, higher_is_worse=False)
            row[f"tail{p}_sent_acc"] = tail_mean(sacc_by_client, tf, higher_is_worse=False)

        self._logger.write_row(row)

        fit_map = self._per_lang_fit_round.get(int(server_round), {})
        tail_set = set(int(x) for x in self._tail_lang_idxs)

        for lang_idx, er in eval_rows.items():
            rr = {k: float("nan") for k in self._per_lang_fields}
            rr["round"] = int(server_round)
            rr["lang_id"] = er.get("lang_id", "")
            rr["lang_idx"] = int(lang_idx)
            rr["used_tail_head"] = int(er.get("used_tail_head", 0))
            rr["in_tail_set_next"] = int(lang_idx in tail_set)

            if self.route_risk_metric == "loss":
                rr["risk_raw"] = float(er.get("val_loss", float("nan")))
            else:
                rr["risk_raw"] = float(er.get("val_cer_mean", float("nan")))
            rr["risk_ema"] = float(self._risk_ema.get(lang_idx, float("nan")))

            rr["val_loss"] = float(er.get("val_loss", float("nan")))
            rr["val_cer_mean"] = float(er.get("val_cer_mean", float("nan")))
            rr["val_cer_max"] = float(er.get("val_cer_max", float("nan")))
            rr["val_wer"] = float(er.get("val_wer", float("nan")))
            rr["val_word_correct_acc"] = float(er.get("val_word_correct_acc", float("nan")))
            rr["val_sent_acc"] = float(er.get("val_sent_acc", float("nan")))
            rr["eval_utts"] = int(er.get("eval_utts", 0))
            rr["eval_word_tokens"] = int(er.get("eval_word_tokens", 0))

            rr["skip_ctc_batches_eval"] = int(er.get("skip_ctc_batches_eval", 0))
            rr["skip_nonfinite_eval"] = int(er.get("skip_nonfinite_eval", 0))
            rr["ctc_kept_total_eval"] = int(er.get("ctc_kept_total_eval", 0))
            rr["ctc_total_total_eval"] = int(er.get("ctc_total_total_eval", 0))
            rr["max_audio_s_used_eval"] = float(er.get("max_audio_s_used_eval", float("nan")))

            fr = fit_map.get(lang_idx, {})
            rr["train_loss"] = float(fr.get("train_loss", float("nan")))
            rr["train_steps"] = int(fr.get("train_steps", 0))
            rr["train_examples"] = int(fr.get("train_examples", 0))
            rr["fit_time_s"] = float(fr.get("fit_time_s", float("nan")))
            rr["effective_fit"] = int(fr.get("effective_fit", 0))

            rr["skip_ctc_batches_fit"] = int(fr.get("skip_ctc_batches_fit", 0))
            rr["skip_nonfinite_fit"] = int(fr.get("skip_nonfinite_fit", 0))
            rr["ctc_kept_total_fit"] = int(fr.get("ctc_kept_total_fit", 0))
            rr["ctc_total_total_fit"] = int(fr.get("ctc_total_total_fit", 0))
            rr["max_audio_s_used_fit"] = float(fr.get("max_audio_s_used_fit", float("nan")))
            rr["delta_l2"] = float(fr.get("delta_l2", float("nan")))

            self._per_lang_logger.write_row(rr)

        merged_i = int(row["merged_tail_into_shared"]) if np.isfinite(row["merged_tail_into_shared"]) else 0
        tailset_n = int(row["tail_set_size"]) if np.isfinite(row["tail_set_size"]) else 0
        print(
            f"[Round {server_round}] meanCER={mean_cer if np.isfinite(mean_cer) else float('nan'):.4f} "
            f"meanLoss={mean_loss if np.isfinite(mean_loss) else float('nan'):.4f} "
            f"tailSet={tailset_n} merged={merged_i}",
            flush=True,
        )

        ret_metrics = {}
        for k, v in row.items():
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                ret_metrics[k] = float(v)
        loss_out = float(mean_loss) if np.isfinite(mean_loss) else None
        return loss_out, ret_metrics


###############################################################################
# Run one experiment
###############################################################################

def _suffix_log_csv(base_csv: str, ft_frac_value: float) -> str:
    root, ext = os.path.splitext(base_csv)
    p = int(round(ft_frac_value * 100))
    return f"{root}_ft{p:02d}{ext if ext else '.csv'}"


def _default_per_lang_csv(log_csv: str) -> str:
    root, ext = os.path.splitext(log_csv)
    return f"{root}_perlang{ext if ext else '.csv'}"


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
        fp16 = str(args.device).startswith("cuda")

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
                    "route_tail_frac": float(args.route_tail_frac),
                    "route_cos_thr": float(args.route_cos_thr),
                },
                f,
                indent=2,
            )

    if args.build_manifests_only:
        return

    all_langs = ok_langs

    clients_per_round = int(args.clients_per_round)
    if clients_per_round > len(all_langs):
        print(f"[warn] clients_per_round={clients_per_round} > num_clients={len(all_langs)}; clamping", flush=True)
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

    tmp_model = Wav2Vec2ForCTC.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    tmp_model.config.ctc_zero_infinity = True
    hidden, n_layer = _find_hidden_size_and_n_layers(tmp_model)
    tmp_hyper = DualHeadHyperLoRA(n_lang=len(all_langs), n_layer=n_layer, hidden_size=hidden).to("cpu")
    for p in tmp_hyper.parameters():
        p.requires_grad = True
    trainable_names, init_arrays = get_trainable_weights(tmp_hyper)
    init_parameters = ndarrays_to_parameters(init_arrays)

    def client_fn(cid: str):
        i = int(cid)
        lang_id = all_langs[i]
        mms_target = lang_to_mms.get(lang_id, None)
        return LocalFleursTARPClient(
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
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    tail_fracs = [float(x.strip()) for x in args.tail_fracs.split(",") if x.strip()]
    if not tail_fracs:
        tail_fracs = [0.10, 0.20, 0.30]

    per_lang_csv = args.log_csv_per_lang.strip() if args.log_csv_per_lang.strip() else _default_per_lang_csv(log_csv_path)

    strategy = TARPFedAvg(
        trainable_names=trainable_names,
        all_langs=all_langs,
        log_csv=log_csv_path,
        log_csv_per_lang=per_lang_csv,
        tail_fracs=tail_fracs,
        route_tail_frac=float(args.route_tail_frac),
        route_cos_thr=float(args.route_cos_thr),
        route_warmup_rounds=int(args.route_warmup_rounds),
        route_risk_beta=float(args.route_risk_beta),
        route_risk_metric=str(args.route_risk_metric),
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


###############################################################################
# Main
###############################################################################

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

    ap.add_argument("--log_csv", type=str, default="runs/tarp_fleurs.csv")
    ap.add_argument("--log_csv_per_lang", type=str, default="", help="Per-language metrics CSV. Default: <log_csv>_perlang.csv")

    ap.add_argument("--num_cpus_per_client", type=float, default=2.0)
    ap.add_argument("--num_gpus_per_client", type=float, default=1.0 if torch.cuda.is_available() else 0.0)

    ap.add_argument("--unbounded_budget_baseline", action="store_true")

    ap.add_argument("--tail_fracs", type=str, default="0.10,0.20,0.30")

    ap.add_argument("--route_tail_frac", type=float, default=0.20)
    ap.add_argument("--route_cos_thr", type=float, default=0.0)
    ap.add_argument("--route_warmup_rounds", type=int, default=10)
    ap.add_argument("--route_risk_beta", type=float, default=0.90)
    ap.add_argument("--route_risk_metric", type=str, default="loss", choices=["cer", "loss"])

    args = ap.parse_args()

    if args.ft_fracs.strip():
        ft_list = [float(x.strip()) for x in args.ft_fracs.split(",") if x.strip()]
        if not ft_list:
            ft_list = [float(args.ft_frac)]
    else:
        ft_list = [float(args.ft_frac)]

    for k, ftv in enumerate(ft_list):
        out_csv = args.log_csv if len(ft_list) == 1 else _suffix_log_csv(args.log_csv, ftv)
        print(f"[run] method=TARP ft_frac={ftv:.2f} log_csv={out_csv}", flush=True)
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
