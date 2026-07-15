#!/usr/bin/env python3
# client_atlas_fleurs.py
#
# Client Atlas for local FLEURS-like partitions where each client = language.
#
# Inputs expected (per language directory):
#   <data_root>/<lang>/train.csv
#   <data_root>/<lang>/validation.csv   (optional; not used by default)
#
# Each CSV row should include at least: path, text (uid optional).
#
# Outputs:
#   <out_dir>/client_atlas_fleurs.png
#   <out_dir>/client_atlas_fleurs_table.csv
#   <out_dir>/capacity_rank.csv
#
# Heatmap blocks:
#   A) audio duration bins (fraction of utts)
#   B) transcript length bins (fraction of utts, word count)
#   C) script composition (fraction of letter chars)

from __future__ import annotations

import os
import re
import csv
import json
import math
import argparse
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

try:
    import torchaudio
except Exception as e:
    torchaudio = None


# -------------------------
# Buckets
# -------------------------
DUR_BINS = [2, 4, 6, 8, 12, 16, 24, 60]  # seconds, last is catch-all upper
DUR_LABELS = ["<2s", "2-4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-60+"]

WORD_BINS = [3, 6, 10, 15, 25, 40, 80, 10_000]
WORD_LABELS = ["<=3", "4-6", "7-10", "11-15", "16-25", "26-40", "41-80", "80+"]

# Scripts are auto-discovered; we keep top K and merge rest into "other"
DEFAULT_MAX_SCRIPTS = 8


# -------------------------
# Helpers
# -------------------------
_WS = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.strip()
    s = _WS.sub(" ", s)
    return s

def bucket_idx(x: float, edges: List[float]) -> int:
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges) - 1

def safe_frac_row(mat: np.ndarray) -> np.ndarray:
    denom = np.maximum(mat.sum(axis=1, keepdims=True), 1e-12)
    return mat / denom

def gini(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= 1e-12:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (idx * x).sum() / (n * s)) - ((n + 1.0) / n))

def discover_langs(data_root: str, exclude: set) -> List[str]:
    out = []
    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if os.path.isdir(p) and name not in exclude:
            out.append(name)
    return out

def read_langs_from_json(fp: str) -> List[str]:
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    langs = obj.get("langs", [])
    return [str(x).strip() for x in langs if str(x).strip()]

def iter_rows(csv_path: str, lang_dir: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            p = (r.get("path") or "").strip()
            t = normalize_text(r.get("text") or "")
            if not p or not t:
                continue
            if not os.path.isabs(p):
                p = os.path.join(lang_dir, p)
            if not os.path.isfile(p):
                continue
            yield p, t

def duration_s_fast(path: str) -> Optional[float]:
    if torchaudio is None:
        return None
    try:
        info = torchaudio.info(path)
        if getattr(info, "num_frames", None) is None or getattr(info, "sample_rate", None) is None:
            return None
        if info.sample_rate <= 0:
            return None
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        return None

def duration_s_fallback(path: str) -> Optional[float]:
    if torchaudio is None:
        return None
    try:
        wav, sr = torchaudio.load(path)
        if sr <= 0 or wav.numel() == 0:
            return None
        n = wav.shape[-1]
        return float(n) / float(sr)
    except Exception:
        return None

def get_duration_s(path: str) -> Optional[float]:
    d = duration_s_fast(path)
    if d is not None and math.isfinite(d) and d > 0:
        return d
    d = duration_s_fallback(path)
    if d is not None and math.isfinite(d) and d > 0:
        return d
    return None

def script_of_char(ch: str) -> Optional[str]:
    # Use Unicode character name as a lightweight script proxy.
    # Only consider letters to avoid punctuation noise.
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    if not name:
        return None
    # Common script tags found in names
    for key in [
        "LATIN", "CYRILLIC", "ARABIC", "GREEK", "HEBREW",
        "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM",
        "THAI", "LAO", "HANGUL", "HIRAGANA", "KATAKANA", "CJK UNIFIED IDEOGRAPH",
        "GEORGIAN", "ARMENIAN", "ETHIOPIC"
    ]:
        if key in name:
            if key == "CJK UNIFIED IDEOGRAPH":
                return "HAN"
            return key
    return "OTHER"

def summarize_tail(df: pd.DataFrame, col: str, tail_frac: float) -> List[str]:
    n = len(df)
    k = max(1, int(math.ceil(tail_frac * n)))
    tail = df.sort_values(col, ascending=True).head(k)
    return tail["lang"].tolist()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../ML_SUPERB/fleurs", help="Root containing per-language folders")
    ap.add_argument("--langs_json", type=str, default="", help="Optional langs_used.json to lock language list")
    ap.add_argument("--langs", type=str, default="", help="Optional comma list to lock language list")
    ap.add_argument("--exclude_langs", type=str, default="", help="Comma list to drop")
    ap.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="Which split CSV to analyze")
    ap.add_argument("--out_dir", type=str, default="", help="Default: <data_root>/analysis_figs")
    ap.add_argument("--max_rows_per_lang", type=int, default=0, help="Optional cap to speed up (0 = all)")
    ap.add_argument("--max_scripts", type=int, default=DEFAULT_MAX_SCRIPTS)
    ap.add_argument("--cmap", type=str, default="Blues")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--tail_fracs", type=str, default="0.10,0.20", help="Comma list for tail sets")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    out_dir = args.out_dir.strip() or os.path.join(data_root, "analysis_figs")
    os.makedirs(out_dir, exist_ok=True)

    exclude = {x.strip() for x in args.exclude_langs.split(",") if x.strip()}

    if args.langs_json.strip():
        langs = read_langs_from_json(args.langs_json.strip())
    elif args.langs.strip():
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    else:
        langs = discover_langs(data_root, exclude)

    langs = [l for l in langs if l and l not in exclude]
    if not langs:
        raise RuntimeError("No languages selected")

    # Pass 1: compute per-language stats and gather global script frequencies
    rows = []
    per_lang_script_counts: Dict[str, Dict[str, int]] = {}
    global_script_counts = defaultdict(int)

    for lang in langs:
        lang_dir = os.path.join(data_root, lang)
        csv_path = os.path.join(lang_dir, f"{args.split}.csv")
        if not os.path.isfile(csv_path):
            # allow validation.csv naming
            if args.split == "validation":
                csv_path = os.path.join(lang_dir, "validation.csv")
        if not os.path.isfile(csv_path):
            print(f"[skip] missing {args.split} CSV for {lang}: {csv_path}")
            continue

        dur_bins = np.zeros(len(DUR_BINS), dtype=float)
        word_bins = np.zeros(len(WORD_BINS), dtype=float)
        script_counts = defaultdict(int)

        utt = 0
        total_sec = 0.0
        dur_list = []
        word_list = []

        for i, (path, text) in enumerate(iter_rows(csv_path, lang_dir)):
            if args.max_rows_per_lang and i >= args.max_rows_per_lang:
                break

            d = get_duration_s(path)
            if d is None:
                continue

            w = len(text.split())
            utt += 1
            total_sec += float(d)
            dur_list.append(float(d))
            word_list.append(float(w))

            dur_bins[bucket_idx(d, DUR_BINS)] += 1.0
            word_bins[bucket_idx(w, WORD_BINS)] += 1.0

            # Script composition: count letter chars by script
            for ch in text:
                sc = script_of_char(ch)
                if sc is None:
                    continue
                script_counts[sc] += 1
                global_script_counts[sc] += 1

        if utt == 0:
            print(f"[skip] no usable rows for {lang}")
            continue

        per_lang_script_counts[lang] = dict(script_counts)

        dur_arr = np.asarray(dur_list, dtype=float)
        w_arr = np.asarray(word_list, dtype=float)

        def pct(x, q):
            if x.size == 0:
                return float("nan")
            return float(np.percentile(x, q))

        rows.append({
            "lang": lang,
            "utts": int(utt),
            "train_minutes": float(total_sec / 60.0),
            "mean_dur_s": float(dur_arr.mean()) if dur_arr.size else float("nan"),
            "p95_dur_s": pct(dur_arr, 95),
            "mean_words": float(w_arr.mean()) if w_arr.size else float("nan"),
            "p95_words": pct(w_arr, 95),
            **{f"durbin__{DUR_LABELS[j]}": float(dur_bins[j]) for j in range(len(DUR_LABELS))},
            **{f"wbin__{WORD_LABELS[j]}": float(word_bins[j]) for j in range(len(WORD_LABELS))},
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No languages were parsed. Check data_root and CSVs.")

    # Choose top scripts globally, merge rest into OTHER
    scripts_sorted = sorted(global_script_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_scripts = [k for (k, _) in scripts_sorted if k != "OTHER"][: max(1, int(args.max_scripts))]
    if "OTHER" not in top_scripts:
        top_scripts.append("OTHER")

    # Add per-language script columns
    for sc in top_scripts:
        df[f"script__{sc}"] = 0.0
    for idx, r in df.iterrows():
        lang = r["lang"]
        scounts = per_lang_script_counts.get(lang, {})
        other = 0
        for sc, c in scounts.items():
            if sc in top_scripts:
                df.at[idx, f"script__{sc}"] += float(c)
            else:
                other += int(c)
        df.at[idx, "script__OTHER"] += float(other)

    # Sort by utterances high to low 
    df = df.sort_values(["utts", "train_minutes"], ascending=[False, False]).reset_index(drop=True)

    # Save tables
    table_path = os.path.join(out_dir, "client_atlas_fleurs_table.csv")
    df.to_csv(table_path, index=False)

    rank_path = os.path.join(out_dir, "capacity_rank.csv")
    df[["lang", "utts", "train_minutes", "mean_dur_s", "mean_words"]].to_csv(rank_path, index=False)

    # Tail summary printed to console
    ut = df["utts"].to_numpy(dtype=float)
    mins = df["train_minutes"].to_numpy(dtype=float)
    print(f"[capacity] n_lang={len(df)} gini_utts={gini(ut):.4f} gini_minutes={gini(mins):.4f}")
    tail_fracs = [float(x.strip()) for x in args.tail_fracs.split(",") if x.strip()]
    for tf in tail_fracs:
        tlangs = summarize_tail(df, "utts", tf)
        print(f"[tail_utts] frac={tf:.2f} langs={tlangs}")

    # Build matrices for plotting (fractions)
    y = np.arange(len(df))
    cap = df["utts"].to_numpy(dtype=float)

    Dmat = df[[f"durbin__{lab}" for lab in DUR_LABELS]].to_numpy(dtype=float)
    Wmat = df[[f"wbin__{lab}" for lab in WORD_LABELS]].to_numpy(dtype=float)
    Smat = df[[f"script__{sc}" for sc in top_scripts]].to_numpy(dtype=float)

    Dp = safe_frac_row(Dmat)
    Wp = safe_frac_row(Wmat)
    Sp = safe_frac_row(Smat)

    # Plot
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    w_bar, w_dur, w_words, w_script = 1.2, 2.3, 2.3, max(2.0, 0.35 * len(top_scripts))
    widths = [w_bar, w_dur, w_words, w_script]

    fig_h = max(6.0, 0.28 * len(df))
    fig_w = sum(widths) + 2.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(1, len(widths), width_ratios=widths, wspace=0.25)

    # Capacity bar
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.barh(y, cap)
    ax0.invert_yaxis()
    ax0.set_xlabel("train utterances")
    ax0.set_yticks(y)
    ax0.set_yticklabels(df["lang"].tolist())
    ax0.set_title("capacity")

    def heat(ax, mat, col_labels, title):
        im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                       cmap=args.cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.tick_params(axis="both", which="both", length=0)
        return im

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = heat(ax1, Dp, DUR_LABELS, "audio duration")

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = heat(ax2, Wp, WORD_LABELS, "transcript length")

    ax3 = fig.add_subplot(gs[0, 3])
    im3 = heat(ax3, Sp, top_scripts, "script")

    # One shared colorbar
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im1, cax=cax)
    cb.set_label("fraction")

    fig.suptitle(f"Client Atlas (FLEURS {args.split} split)", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.96])

    out_png = os.path.join(out_dir, "client_atlas_fleurs.png")
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)

    print("[ok] wrote:", out_png)
    print("[ok] wrote:", table_path)
    print("[ok] wrote:", rank_path)


if __name__ == "__main__":
    main()
