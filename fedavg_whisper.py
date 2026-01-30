#!/usr/bin/env python3
# fedavg_lora_whisper_fleurs.py
#
# FedAvg + LoRA on Whisper (encoder-side only) for local FLEURS manifests.
# - Whisper seq2seq loss (cross-entropy), no CTC
# - LoRA injected into Whisper encoder self-attn projections
# - Only LoRA params are trained and federated
# - Mel features padded/truncated to 3000 frames (30s) to satisfy Whisper forward

from __future__ import annotations

import os
import re
import csv
import time
import math
import json
import random
import argparse
import unicodedata
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitIns, EvaluateIns
from flwr.server.strategy import FedAvg

import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import splits as splits_mod


# ----------------------------------------------------------------------
# Text normalization + CER
# ----------------------------------------------------------------------

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
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]

def cer(hyp: str, ref: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(hyp, ref) / len(ref)

def pctl(xs: List[float], q: float) -> float:
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return float("nan")
    return float(np.percentile(xs, q))

def tail_mean(xs: List[float], frac: float, higher_is_worse: bool) -> float:
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return float("nan")
    xs.sort(reverse=higher_is_worse)
    k = max(1, int(math.ceil(float(frac) * len(xs))))
    return float(np.mean(xs[:k]))


# ----------------------------------------------------------------------
# Audio + dataset
# ----------------------------------------------------------------------

def _load_audio_16k(path: str, max_audio_s: float) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0)
    if max_audio_s and max_audio_s > 0:
        wav = wav[: int(16000 * float(max_audio_s))]
    return wav.numpy()

class LocalFleursIterable(IterableDataset):
    def __init__(self, lang_dir: str, split: str, max_hours: float, max_audio_s: float):
        self.rows: List[Tuple[str, str]] = []
        with open(os.path.join(lang_dir, f"{split}.csv"), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = os.path.join(lang_dir, r["path"])
                if os.path.isfile(p):
                    self.rows.append((p, normalize_transcript(r["text"])))
        self.max_hours = float(max_hours)
        self.max_audio_s = float(max_audio_s)

    def __iter__(self):
        total = 0.0
        for p, t in self.rows:
            wav = _load_audio_16k(p, self.max_audio_s)
            dur = float(len(wav)) / 16000.0
            total += dur
            yield {"audio": wav, "text": t}
            if self.max_hours > 0 and total >= self.max_hours * 3600.0:
                break

class WhisperCollatorPad3000:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
        self.num_frames = 3000  # Whisper expects 80 x 3000

    def __call__(self, batch):
        audios = [b["audio"] for b in batch]
        texts = [b["text"] for b in batch]

        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        feats = inputs["input_features"]  # (B, 80, T)
        T = feats.shape[-1]
        if T < self.num_frames:
            pad = torch.zeros((feats.shape[0], feats.shape[1], self.num_frames - T), dtype=feats.dtype)
            feats = torch.cat([feats, pad], dim=-1)
        elif T > self.num_frames:
            feats = feats[..., : self.num_frames]
        inputs["input_features"] = feats

        labels = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


# ----------------------------------------------------------------------
# LoRA
# ----------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r) if self.r > 0 else 1.0
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else None

        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base.in_features
        out_f = base.out_features

        # A: (r, in), B: (out, r)
        self.A = nn.Parameter(torch.zeros(self.r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, self.r))

        # init: A small random, B zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        y0 = self.base(x)
        if self.r <= 0:
            return y0
        x_in = self.dropout(x) if self.dropout is not None else x
        # (..., in) -> (..., r) -> (..., out)
        z = F.linear(x_in, self.A)          # weight (r, in)
        dz = F.linear(z, self.B) * self.scaling  # weight (out, r)
        return y0 + dz

def inject_lora_whisper_encoder(model: WhisperForConditionalGeneration, r: int, alpha: float, dropout: float) -> List[Tuple[str, nn.Parameter]]:
    enc = model.model.encoder
    trainables: List[Tuple[str, nn.Parameter]] = []
    for lid, layer in enumerate(enc.layers):
        attn = layer.self_attn
        for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            lin = getattr(attn, pname)
            wrapped = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
            setattr(attn, pname, wrapped)
            trainables.append((f"enc.layers.{lid}.self_attn.{pname}.A", wrapped.A))
            trainables.append((f"enc.layers.{lid}.self_attn.{pname}.B", wrapped.B))
    return trainables

def whisper_hidden_and_layers(model):
    enc = model.model.encoder
    hidden = enc.layers[0].self_attn.q_proj.in_features
    return hidden, len(enc.layers)


# ----------------------------------------------------------------------
# Flower client
# ----------------------------------------------------------------------

class LocalFleursFedAvgLoRAClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        lang_dir: str,
        lang_idx: int,
        all_langs: List[str],
        model_id: str,
        device: str,
        fp16: bool,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        max_train_hours: float,
        max_valid_hours: float,
        max_audio_s: float,
        batch_size: int,
        local_steps: int,
        lr: float,
        weight_decay: float,
        max_grad_norm: float,
    ):
        self.cid = cid
        self.lang_idx = int(lang_idx)
        self.lang_token = all_langs[self.lang_idx]

        self.device = torch.device(device)
        self.fp16 = bool(fp16) and (self.device.type == "cuda")
        self.local_steps = int(local_steps)
        self.max_grad_norm = float(max_grad_norm)

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)

        # freeze full backbone
        for p in self.model.parameters():
            p.requires_grad = False

        # inject LoRA (encoder-only)
        named_trainables = inject_lora_whisper_encoder(
            self.model, r=int(lora_r), alpha=float(lora_alpha), dropout=float(lora_dropout)
        )
        self.trainable_names = [n for (n, _) in named_trainables]
        self.trainable_params = [p for (_, p) in named_trainables]

        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        self.train_loader = DataLoader(
            LocalFleursIterable(lang_dir, "train", max_hours=max_train_hours, max_audio_s=max_audio_s),
            batch_size=int(batch_size),
            collate_fn=WhisperCollatorPad3000(self.processor),
        )

        self.valid_loader = DataLoader(
            LocalFleursIterable(lang_dir, "validation", max_hours=max_valid_hours, max_audio_s=max_audio_s),
            batch_size=int(batch_size),
            collate_fn=WhisperCollatorPad3000(self.processor),
        )

        # forced decoding ids (best effort)
        self.forced_decoder_ids = None
        try:
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=self.lang_token,
                task="transcribe",
            )
        except Exception:
            self.forced_decoder_ids = None

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.trainable_params]

    def set_parameters(self, params):
        for p, v in zip(self.trainable_params, params):
            p.data.copy_(torch.from_numpy(v).to(p.device))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        step = 0
        total_loss = 0.0

        for batch in self.train_loader:
            if self.local_steps > 0 and step >= self.local_steps:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.fp16):
                out = self.model(**batch)
                loss = out.loss

            self.scaler.scale(loss).backward()

            if self.max_grad_norm and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            step += 1

        avg_loss = total_loss / max(1, step)

        return (
            self.get_parameters({}),
            step,  # using steps as weight (consistent with your other sims)
            {"lang_idx": self.lang_idx, "train_loss": float(avg_loss)},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        loss_only = bool(config.get("loss_only", False))
        max_batches = int(config.get("eval_max_batches", 2))
        gen_max_new_tokens = int(config.get("gen_max_new_tokens", 128))

        losses: List[float] = []
        cer_vals: List[float] = []

        with torch.inference_mode():
            for bi, batch in enumerate(self.valid_loader):
                if max_batches > 0 and bi >= max_batches:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    out = self.model(**batch)
                losses.append(float(out.loss.item()))

                if not loss_only:
                    gen = self.model.generate(
                        batch["input_features"],
                        forced_decoder_ids=self.forced_decoder_ids,
                        num_beams=1,
                        do_sample=False,
                        max_new_tokens=gen_max_new_tokens,
                        return_dict_in_generate=False,
                        output_scores=False,
                    )
                    hyps = self.processor.batch_decode(gen, skip_special_tokens=True)

                    labels = batch["labels"].clone()
                    labels[labels == -100] = self.processor.tokenizer.pad_token_id
                    refs = self.processor.batch_decode(labels, skip_special_tokens=True)

                    for h, r in zip(hyps, refs):
                        cer_vals.append(cer(normalize_transcript(h), normalize_transcript(r)))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_cer = float(np.mean(cer_vals)) if cer_vals else float("nan")

        return mean_loss, 1, {
            "lang_idx": self.lang_idx,
            "val_loss": mean_loss,
            "val_cer_mean": mean_cer,
        }


# ----------------------------------------------------------------------
# Strategy with eval throttling + CSV logging
# ----------------------------------------------------------------------

class FedAvgLoRAStrategy(FedAvg):
    def __init__(
        self,
        log_csv: str,
        tail_fracs: List[float],
        eval_every: int,
        cer_every: int,
        eval_max_batches: int,
        gen_max_new_tokens: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.log_csv = log_csv
        self.tail_fracs = [float(x) for x in tail_fracs]
        self.eval_every = int(eval_every)
        self.cer_every = int(cer_every)
        self.eval_max_batches = int(eval_max_batches)
        self.gen_max_new_tokens = int(gen_max_new_tokens)

        self._round_start_ts: Optional[float] = None

        os.makedirs(os.path.dirname(self.log_csv) or ".", exist_ok=True)
        self._csv_fields = ["round", "wall_time_s", "clients_fit", "clients_eval", "mean_loss", "mean_cer", "max_cer", "p95_cer"]
        for tf in self.tail_fracs:
            p = int(round(tf * 100))
            self._csv_fields.append(f"tail{p}_cer")

        if not os.path.exists(self.log_csv):
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start_ts = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.eval_every > 1 and (server_round % self.eval_every != 0):
            return []

        cfgs = super().configure_evaluate(server_round, parameters, client_manager)

        do_cer = (self.cer_every <= 1) or (server_round % self.cer_every == 0)
        out = []
        for client, ev_ins in cfgs:
            new_cfg = dict(ev_ins.config)
            new_cfg["loss_only"] = int(not do_cer)
            new_cfg["eval_max_batches"] = int(self.eval_max_batches)
            new_cfg["gen_max_new_tokens"] = int(self.gen_max_new_tokens)
            out.append((client, EvaluateIns(ev_ins.parameters, new_cfg)))
        return out

    def aggregate_evaluate(self, server_round, results, failures):
        wall_s = float(time.time() - self._round_start_ts) if self._round_start_ts is not None else float("nan")

        if not results:
            row = {k: float("nan") for k in self._csv_fields}
            row["round"] = int(server_round)
            row["wall_time_s"] = wall_s
            row["clients_fit"] = float("nan")
            row["clients_eval"] = 0
            with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)
            return float("nan"), row

        losses: List[float] = []
        cers: List[float] = []

        for _, ev in results:
            losses.append(float(ev.loss))
            m = ev.metrics or {}
            if "val_cer_mean" in m:
                v = float(m["val_cer_mean"])
                if v == v:
                    cers.append(v)

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_cer = float(np.mean(cers)) if cers else float("nan")
        max_cer = float(max(cers)) if cers else float("nan")
        p95_cer = pctl(cers, 95.0)

        row: Dict[str, float] = {
            "round": int(server_round),
            "wall_time_s": wall_s,
            "clients_fit": float("nan"),
            "clients_eval": float(len(results)),
            "mean_loss": mean_loss,
            "mean_cer": mean_cer,
            "max_cer": max_cer,
            "p95_cer": p95_cer,
        }

        for tf in self.tail_fracs:
            p = int(round(tf * 100))
            row[f"tail{p}_cer"] = tail_mean(cers, tf, higher_is_worse=True)

        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)

        print(f"[Round {server_round}] meanCER={mean_cer:.4f} maxCER={max_cer:.4f} tail10={row.get('tail10_cer', float('nan')):.4f}")
        return mean_loss, row


# ----------------------------------------------------------------------
# Run one experiment
# ----------------------------------------------------------------------

def _suffix_log_csv(base_csv: str, tag: str) -> str:
    root, ext = os.path.splitext(base_csv)
    return f"{root}_{tag}{ext if ext else '.csv'}"

def run_one(args, log_csv_path: str) -> None:
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

    n_clients = int(args.num_clients) if args.num_clients is not None else int(args.num_langs)

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

    if not ok_langs:
        raise RuntimeError("No usable languages after manifest build")

    if args.write_langs_json:
        with open(args.write_langs_json, "w", encoding="utf-8") as f:
            json.dump(
                {"data_root": args.data_root, "langs": ok_langs, "num_clients": len(ok_langs)},
                f,
                indent=2,
            )

    if args.build_manifests_only:
        return

    all_langs = ok_langs

    clients_per_round = int(args.clients_per_round)
    if clients_per_round > len(all_langs):
        clients_per_round = len(all_langs)

    # Init LoRA params from a dummy model to get deterministic shapes/order
    tmp_model = WhisperForConditionalGeneration.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    for p in tmp_model.parameters():
        p.requires_grad = False
    named_trainables = inject_lora_whisper_encoder(tmp_model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    init_arrays = [p.detach().cpu().numpy() for _, p in named_trainables]
    init_parameters = ndarrays_to_parameters(init_arrays)

    def client_fn(cid: str):
        i = int(cid)
        return LocalFleursFedAvgLoRAClient(
            cid=cid,
            lang_dir=os.path.join(args.data_root, all_langs[i]),
            lang_idx=i,
            all_langs=all_langs,
            model_id=args.model_id,
            device=args.device,
            fp16=fp16,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_train_hours=args.max_train_hours,
            max_valid_hours=args.max_valid_hours,
            max_audio_s=args.max_audio_s,
            batch_size=args.batch_size,
            local_steps=args.local_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )

    tail_fracs = [float(x.strip()) for x in args.tail_fracs.split(",") if x.strip()] or [0.10, 0.20, 0.30]

    eval_clients = min(int(args.eval_clients_per_round), len(all_langs))
    strategy = FedAvgLoRAStrategy(
        log_csv=log_csv_path,
        tail_fracs=tail_fracs,
        eval_every=int(args.eval_every),
        cer_every=int(args.cer_every),
        eval_max_batches=int(args.eval_max_batches),
        gen_max_new_tokens=int(args.gen_max_new_tokens),
        fraction_fit=min(1.0, clients_per_round / max(1, len(all_langs))),
        min_fit_clients=min(clients_per_round, len(all_langs)),
        min_available_clients=len(all_langs),
        fraction_evaluate=min(1.0, eval_clients / max(1, len(all_langs))),
        min_evaluate_clients=eval_clients,
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


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

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

    ap.add_argument("--model_id", type=str, default="openai/whisper-small")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--clients_per_round", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_train_hours", type=float, default=0.50)
    ap.add_argument("--max_valid_hours", type=float, default=0.10)
    ap.add_argument("--max_audio_s", type=float, default=16.0)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--local_steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    # Evaluation throttling
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--cer_every", type=int, default=10)  # full CER decode every N eval rounds
    ap.add_argument("--eval_clients_per_round", type=int, default=4)
    ap.add_argument("--eval_max_batches", type=int, default=2)
    ap.add_argument("--gen_max_new_tokens", type=int, default=128)

    ap.add_argument("--log_csv", type=str, default="runs/fedavg_lora_fleurs_whisper.csv")
    ap.add_argument("--tail_fracs", type=str, default="0.10,0.20,0.30")

    ap.add_argument("--num_cpus_per_client", type=float, default=2.0)
    ap.add_argument("--num_gpus_per_client", type=float, default=1.0 if torch.cuda.is_available() else 0.0)

    args = ap.parse_args()

    print(f"[run] method=FedAvg+LoRA model={args.model_id} log_csv={args.log_csv}")
    run_one(args, log_csv_path=args.log_csv)

if __name__ == "__main__":
    main()
