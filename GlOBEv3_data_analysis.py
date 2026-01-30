#!/usr/bin/env python3
# globe_v3_stream_stats.py

import argparse
import time
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None

from datasets import load_dataset


def norm_str(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and np.isnan(x):
        return "NA"
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"none", "nan"}:
            return "NA"
        return s
    return str(x)


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


class RunningStats:
    # Welford mean/var + min/max
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = None
        self.max = None

    def update(self, x: float):
        self.n += 1
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return float(np.sqrt(self.m2 / (self.n - 1)))


class Reservoir:
    # uniform reservoir sampling for approximate quantiles
    def __init__(self, k: int, seed: int = 0):
        self.k = k
        self.rng = random.Random(seed)
        self.buf = []
        self.seen = 0

    def add(self, x: float):
        self.seen += 1
        if len(self.buf) < self.k:
            self.buf.append(x)
            return
        j = self.rng.randint(1, self.seen)
        if j <= self.k:
            self.buf[j - 1] = x

    def values(self):
        return self.buf


def print_schema(ds_obj, split: str):
    # Works for both IterableDataset (streaming) and Dataset
    feats = getattr(ds_obj, "features", None)
    if feats is None:
        print("[schema] features: NA (not exposed)")
        return
    cols = list(feats.keys())
    print("\n[schema] split:", split)
    print("[schema] num_columns:", len(cols))
    print("[schema] columns:")
    for c in cols:
        print("  -", c)
    print("\n[schema] feature dtypes:")
    for c in cols:
        try:
            print("  -", c, ":", feats[c])
        except Exception:
            print("  -", c, ": <unprintable>")


def get_audio_info(ex: Dict[str, Any]) -> Dict[str, Any]:
    a = ex.get("audio")
    if isinstance(a, dict):
        return {
            "audio_path": a.get("path"),
            "audio_sr": a.get("sampling_rate"),
            "audio_len": None if a.get("array") is None else len(a.get("array")),
        }
    return {"audio_path": None, "audio_sr": None, "audio_len": None}


def torch_quantiles(vals, use_cuda: bool):
    if torch is None:
        return None
    if len(vals) == 0:
        return None
    t = torch.tensor(vals, dtype=torch.float32)
    if use_cuda and torch.cuda.is_available():
        t = t.cuda(non_blocking=True)
    qs = torch.tensor([0.05, 0.50, 0.95], dtype=torch.float32, device=t.device)
    out = torch.quantile(t, qs).detach().cpu().numpy().tolist()
    return {"p05": out[0], "p50": out[1], "p95": out[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="MushanW/GLOBE_V3")
    ap.add_argument("--split", type=str, default="train", choices=["train", "dev", "test"])
    ap.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended).")
    ap.add_argument("--max_rows", type=int, default=200000, help="Stop after this many rows (0 = no limit).")
    ap.add_argument("--progress_every", type=int, default=5000, help="Print progress every N rows.")
    ap.add_argument("--topk", type=int, default=15, help="Top-K categories to show for each field.")
    ap.add_argument("--reservoir", type=int, default=100000, help="Reservoir sample size for quantiles.")
    ap.add_argument("--use_cuda", action="store_true", help="Use CUDA for quantile computation if available.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading:", args.dataset, "split:", args.split, "streaming:", args.streaming)
    d = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    print_schema(d, args.split)

    # Peek first example to confirm keys
    it = iter(d)
    first = next(it)
    print("\n[first_example] keys:", list(first.keys()))
    aud = get_audio_info(first)
    print("[first_example] audio_info:", {k: aud[k] for k in aud})

    # Put first back into a small buffer so we do not lose it
    buffered = [first]

    # Categorical counters
    cat_fields = [
        "predicted_accent",
        "predicted_gender",
        "predicted_age",
        "common_voice_accents",
        "common_voice_gender",
        "common_voice_age",
    ]
    counters = {f: Counter() for f in cat_fields}
    missing = {f: 0 for f in cat_fields}

    # Numeric stats
    num_fields = ["snr", "utmos", "cer", "wer"]
    stats = {f: RunningStats() for f in num_fields}
    reservoirs = {f: Reservoir(args.reservoir, seed=args.seed + i) for i, f in enumerate(num_fields)}

    # Column existence map (so you see what is actually present as it streams)
    seen_cols = Counter()

    start = time.time()
    n = 0

    def handle(ex: Dict[str, Any]):
        nonlocal n
        n += 1
        for k in ex.keys():
            seen_cols[k] += 1

        # categorical
        for f in cat_fields:
            if f in ex:
                v = norm_str(ex.get(f))
                counters[f][v] += 1
                if v == "NA":
                    missing[f] += 1
            else:
                # field not present at all in this example
                pass

        # numeric
        for f in num_fields:
            if f in ex:
                v = to_float(ex.get(f))
                if v is not None:
                    stats[f].update(v)
                    reservoirs[f].add(v)

        # audio sanity (do not force decode array)
        _ = ex.get("audio")

    # process buffered first row
    for ex in buffered:
        handle(ex)

    # continue streaming
    while True:
        if args.max_rows and n >= args.max_rows:
            break
        try:
            ex = next(it)
        except StopIteration:
            break
        handle(ex)

        if args.progress_every > 0 and (n % args.progress_every == 0):
            elapsed = time.time() - start
            rps = n / max(elapsed, 1e-9)
            print(f"[progress] rows={n} elapsed_s={elapsed:.1f} rows_per_s={rps:.1f}")

    elapsed = time.time() - start
    print("\nDone.")
    print("Total rows processed:", n)
    print("Elapsed seconds:", round(elapsed, 2))
    print("Throughput rows/s:", round(n / max(elapsed, 1e-9), 2))

    # Columns observed
    print("\n[observed_columns] count:", len(seen_cols))
    print("[observed_columns] top 50 by presence:")
    for k, v in seen_cols.most_common(50):
        print("  -", k, ":", v)

    # Categorical summaries
    for f in cat_fields:
        if sum(counters[f].values()) == 0:
            print(f"\n[{f}] not observed in processed rows.")
            continue
        print(f"\n[{f}] total_seen={sum(counters[f].values())} missing_NA={missing[f]}")
        for k, v in counters[f].most_common(args.topk):
            print("  ", k, ":", v)

    # Numeric summaries + approximate quantiles from reservoir
    use_cuda = bool(args.use_cuda and (torch is not None) and torch.cuda.is_available())
    if args.use_cuda and torch is None:
        print("\n[cuda] torch not installed, cannot use CUDA.")
    if args.use_cuda and torch is not None and not torch.cuda.is_available():
        print("\n[cuda] torch is installed but CUDA is not available on this machine.")

    for f in num_fields:
        s = stats[f]
        if s.n == 0:
            print(f"\n[{f}] no numeric values observed.")
            continue
        q = torch_quantiles(reservoirs[f].values(), use_cuda=use_cuda)
        print(f"\n[{f}] n={s.n} mean={s.mean:.6f} std={s.std():.6f} min={s.min} max={s.max}")
        if q is None:
            # fallback: numpy quantiles on CPU
            vals = reservoirs[f].values()
            if len(vals) > 0:
                p05, p50, p95 = np.quantile(np.array(vals, dtype=np.float32), [0.05, 0.5, 0.95]).tolist()
                print(f"[{f}] approx_quantiles_reservoir p05={p05:.6f} p50={p50:.6f} p95={p95:.6f} (CPU)")
        else:
            print(f"[{f}] approx_quantiles_reservoir p05={q['p05']:.6f} p50={q['p50']:.6f} p95={q['p95']:.6f} ({'CUDA' if use_cuda else 'CPU torch'})")


if __name__ == "__main__":
    main()
