#!/usr/bin/env python3
# analyze_fleurs_noniid_v2.py
#
# Readability-focused non-IID analysis for ML_SUPERB FLEURS per-language splits.
#
# Uses existing per-language manifests:
#   <data_root>/<lang>/{train.csv,validation.csv,test.csv}
# Columns: path,text,uid,duration_sec
#
# Improvements vs v1:
#   - Bar plots: show only top-K and bottom-K languages (sorted), horizontal bars.
#   - Heatmap: optional subset selection and label thinning; optional clustering reorder if scipy is available.
#   - PCA: annotate only informative points (top-K by divergence + axis extremes).
#
# Example:
#   python analyze_fleurs_noniid_v2.py --data_root ML_SUPERB/fleurs --split train --out_dir noniid_out
#   python analyze_fleurs_noniid_v2.py --langs_json langs_used.json --split train --heatmap_max_langs 50 --pca_annotate_k 20
#

from __future__ import annotations

import json
import argparse
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_float(x: object) -> float:
    try:
        if x is None:
            return float("nan")
        s = str(x).strip()
        if s == "":
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


@lru_cache(maxsize=65536)
def _char_script(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except Exception:
        return "OTHER"

    for key in [
        "LATIN", "CYRILLIC", "ARABIC", "DEVANAGARI", "BENGALI", "GURMUKHI",
        "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM",
        "SINHALA", "THAI", "LAO", "MYANMAR", "GEORGIAN", "ARMENIAN",
        "HEBREW", "GREEK", "HANGUL", "HIRAGANA", "KATAKANA", "CJK UNIFIED",
        "HAN", "ETHIOPIC"
    ]:
        if key in name:
            if key in ["CJK UNIFIED", "HAN"]:
                return "HAN"
            return key

    cat = unicodedata.category(ch)
    if cat.startswith("N"):
        return "NUMBER"
    if cat.startswith("P"):
        return "PUNCT"
    if cat.startswith("S"):
        return "SYMBOL"
    return "OTHER"


def _discover_langs(data_root: Path) -> List[str]:
    langs = []
    if not data_root.is_dir():
        return langs
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and (p / "train.csv").is_file():
            langs.append(p.name)
    return langs


def _load_langs(args: argparse.Namespace) -> List[str]:
    if args.langs_json:
        with open(args.langs_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        langs = obj.get("langs", [])
        if not isinstance(langs, list) or not langs:
            raise ValueError(f"langs_json has no 'langs' list: {args.langs_json}")
        return [str(x) for x in langs]

    if args.langs:
        return [x.strip() for x in args.langs.split(",") if x.strip()]

    return _discover_langs(Path(args.data_root))


def _read_split_csv(lang_dir: Path, split: str) -> pd.DataFrame:
    csv_path = lang_dir / f"{split}.csv"
    if not csv_path.is_file():
        return pd.DataFrame(columns=["path", "text", "uid", "duration_sec"])
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError(f"Missing 'text' column in {csv_path}")
    if "duration_sec" not in df.columns:
        df["duration_sec"] = np.nan
    return df


def _tokenize(text: str) -> List[str]:
    return [t for t in (text or "").split() if t]


def _char_counts(text: str) -> Counter:
    c = Counter()
    for ch in (text or ""):
        if ch.isspace():
            continue
        c[ch] += 1
    return c


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = np.clip(x, 0.0, None)
    if np.allclose(x.sum(), 0.0):
        return 0.0
    xs = np.sort(x)
    n = xs.size
    cum = np.cumsum(xs)
    return float((n + 1.0 - 2.0 * np.sum(cum / cum[-1])) / n)


def _pca_2d(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :2] * S[:2]


def _try_cluster_order(D: np.ndarray) -> Optional[np.ndarray]:
    # Returns an index order for the heatmap.
    # Uses scipy if available; otherwise returns None.
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except Exception:
        return None

    # Convert full matrix to condensed distance.
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    return order


def _select_lang_subset(
    stats_df: pd.DataFrame,
    heatmap_max_langs: int,
    strategy: str,
    seed: int = 0,
) -> List[str]:
    langs = stats_df["lang"].tolist()
    if heatmap_max_langs <= 0 or len(langs) <= heatmap_max_langs:
        return langs

    rng = np.random.default_rng(seed)

    if strategy == "divergent":
        # Most informative for non-IID: choose high avg_jsd languages.
        sub = stats_df.sort_values("avg_jsd_to_others", ascending=False).head(heatmap_max_langs)
        return sub["lang"].tolist()

    if strategy == "hours":
        sub = stats_df.sort_values("total_hours", ascending=False).head(heatmap_max_langs)
        return sub["lang"].tolist()

    if strategy == "mixed":
        k = heatmap_max_langs
        a = max(1, k // 3)
        b = max(1, k // 3)
        c = k - a - b

        top_div = stats_df.sort_values("avg_jsd_to_others", ascending=False).head(a)["lang"].tolist()
        top_hrs = stats_df.sort_values("total_hours", ascending=False).head(b)["lang"].tolist()

        remaining = [l for l in langs if (l not in set(top_div)) and (l not in set(top_hrs))]
        rng.shuffle(remaining)
        rand = remaining[:c]
        out = []
        for l in top_div + top_hrs + rand:
            if l not in out:
                out.append(l)
        return out[:k]

    # random
    rng.shuffle(langs)
    return langs[:heatmap_max_langs]


def _plot_top_bottom_barh(
    stats_df: pd.DataFrame,
    col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    top_k: int,
    bottom_k: int,
    fig_dpi: int,
) -> None:
    df = stats_df[["lang", col]].copy()
    df = df[np.isfinite(df[col].to_numpy(dtype=np.float64))]
    df = df.sort_values(col, ascending=False)

    top = df.head(max(0, top_k)).copy() if top_k > 0 else df.head(0)
    bot = df.tail(max(0, bottom_k)).copy() if bottom_k > 0 else df.head(0)

    shown = pd.concat([top, bot], axis=0)
    if shown.empty:
        return

    shown = shown.drop_duplicates(subset=["lang"], keep="first")
    shown = shown.sort_values(col, ascending=True)  # barh looks better small->large

    plt.figure(figsize=(8.2, max(4.0, 0.25 * len(shown))))
    plt.barh(shown["lang"], shown[col])
    plt.xlabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=fig_dpi)
    plt.close()


def _plot_heatmap(
    D: np.ndarray,
    langs: List[str],
    out_path: Path,
    fig_dpi: int,
    cluster: bool,
    max_tick_labels: int,
) -> None:
    n = len(langs)
    order = None
    if cluster and n >= 3:
        order = _try_cluster_order(D)
    if order is None:
        order = np.arange(n)

    D2 = D[np.ix_(order, order)]
    langs2 = [langs[i] for i in order.tolist()]

    # Thin tick labels to keep it readable.
    tick_every = max(1, int(math.ceil(n / float(max(1, max_tick_labels)))))

    plt.figure(figsize=(8.5, 7.6))
    plt.imshow(D2, aspect="auto", interpolation="nearest")
    plt.colorbar(label="JSD (character distribution)")

    xt = np.arange(0, n, tick_every)
    plt.xticks(ticks=xt, labels=[langs2[i] for i in xt], rotation=90, fontsize=7)
    plt.yticks(ticks=xt, labels=[langs2[i] for i in xt], fontsize=7)

    plt.title(f"Pairwise JSD heatmap (n={n}, labels every {tick_every})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=fig_dpi)
    plt.close()


def _plot_pca(
    Z: np.ndarray,
    langs: List[str],
    stats_df: pd.DataFrame,
    out_path: Path,
    fig_dpi: int,
    annotate_k: int,
    annotate_extremes: int,
) -> None:
    plt.figure(figsize=(7.2, 6.0))
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of languages by char distribution")

    # Annotation strategy:
    # - top-K by avg_jsd_to_others (most non-IID)
    # - plus extremes along PC1 and PC2
    annotate = set()

    if "avg_jsd_to_others" in stats_df.columns and annotate_k > 0:
        tmp = stats_df.set_index("lang").loc[langs]
        top = tmp["avg_jsd_to_others"].sort_values(ascending=False).head(annotate_k).index.tolist()
        annotate.update(top)

    if annotate_extremes > 0:
        idx_pc1 = np.argsort(Z[:, 0])
        idx_pc2 = np.argsort(Z[:, 1])
        for arr in [idx_pc1, idx_pc2]:
            for i in arr[:annotate_extremes]:
                annotate.add(langs[int(i)])
            for i in arr[-annotate_extremes:]:
                annotate.add(langs[int(i)])

    # Draw only selected labels.
    for i, lang in enumerate(langs):
        if lang in annotate:
            plt.text(Z[i, 0], Z[i, 1], lang, fontsize=8)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=fig_dpi)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="ML_SUPERB/fleurs")
    ap.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    ap.add_argument("--langs", type=str, default="")
    ap.add_argument("--langs_json", type=str, default="")
    ap.add_argument("--max_langs", type=int, default=0, help="0 means no limit")
    ap.add_argument("--sample_rows", type=int, default=0, help="0 means use all rows per language")
    ap.add_argument("--top_k_chars", type=int, default=300, help="char vocab size used for divergence/PCA")
    ap.add_argument("--out_dir", type=str, default="noniid_out")
    ap.add_argument("--fig_dpi", type=int, default=160)

    # Readability controls
    ap.add_argument("--bar_top_k", type=int, default=20, help="show top-K languages in bar plots")
    ap.add_argument("--bar_bottom_k", type=int, default=20, help="show bottom-K languages in bar plots")
    ap.add_argument("--heatmap_max_langs", type=int, default=60, help="0 means all; otherwise subset size for heatmap")
    ap.add_argument("--heatmap_subset", type=str, default="mixed",
                    choices=["mixed", "divergent", "hours", "random"],
                    help="how to choose languages for the heatmap subset")
    ap.add_argument("--heatmap_cluster", action="store_true", help="cluster-reorder heatmap if scipy is available")
    ap.add_argument("--heatmap_max_tick_labels", type=int, default=35, help="max labels per axis in heatmap")

    ap.add_argument("--pca_annotate_k", type=int, default=20, help="annotate top-K by avg divergence")
    ap.add_argument("--pca_annotate_extremes", type=int, default=6, help="annotate extremes on PC axes")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = _load_langs(args)
    if args.max_langs and args.max_langs > 0:
        langs = langs[: args.max_langs]
    if not langs:
        raise RuntimeError("No languages found. Check --data_root or --langs_json or --langs.")

    lang_rows: List[Dict[str, object]] = []
    lang_char_counters: Dict[str, Counter] = {}
    lang_dur_samples: Dict[str, np.ndarray] = {}

    global_char = Counter()

    for lang in langs:
        lang_dir = data_root / lang
        df = _read_split_csv(lang_dir, args.split)
        if df.empty:
            continue

        if args.sample_rows and args.sample_rows > 0 and len(df) > args.sample_rows:
            df = df.sample(n=args.sample_rows, random_state=0).reset_index(drop=True)

        n_utt = int(len(df))
        durs = np.array([_safe_float(x) for x in df["duration_sec"].values], dtype=np.float64)
        durs = durs[np.isfinite(durs)]
        total_sec = float(durs.sum()) if durs.size else float("nan")

        texts = df["text"].astype(str).fillna("").tolist()
        token_lens = []
        char_counter = Counter()
        script_counter = Counter()

        for t in texts:
            token_lens.append(len(_tokenize(t)))

            cc = _char_counts(t)
            char_counter.update(cc)
            for ch, ct in cc.items():
                script_counter[_char_script(ch)] += int(ct)

        token_lens = np.asarray(token_lens, dtype=np.float64)

        tot_chars = sum(char_counter.values())
        if tot_chars > 0:
            probs = np.array([v for v in char_counter.values()], dtype=np.float64) / float(tot_chars)
            char_entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        else:
            char_entropy = float("nan")

        lang_char_counters[lang] = char_counter
        global_char.update(char_counter)
        lang_dur_samples[lang] = durs

        row = {
            "lang": lang,
            "split": args.split,
            "n_utt": n_utt,
            "total_hours": (total_sec / 3600.0) if np.isfinite(total_sec) else float("nan"),
            "mean_dur_sec": float(durs.mean()) if durs.size else float("nan"),
            "p95_dur_sec": float(np.percentile(durs, 95)) if durs.size else float("nan"),
            "mean_tokens": float(token_lens.mean()) if token_lens.size else float("nan"),
            "p95_tokens": float(np.percentile(token_lens, 95)) if token_lens.size else float("nan"),
            "uniq_chars": int(len(char_counter)),
            "char_entropy": char_entropy,
        }

        s_tot = sum(script_counter.values())
        if s_tot > 0:
            for k, v in script_counter.items():
                row[f"script_{k}_frac"] = float(v) / float(s_tot)

        lang_rows.append(row)

    if not lang_rows:
        raise RuntimeError("No usable languages found (empty CSVs).")

    stats_df = pd.DataFrame(lang_rows).sort_values("lang").reset_index(drop=True)
    gini_hours = _gini(stats_df["total_hours"].to_numpy(dtype=np.float64))
    gini_utts = _gini(stats_df["n_utt"].to_numpy(dtype=np.float64))

    # Build vocab
    top_k = max(10, int(args.top_k_chars))
    vocab = [ch for ch, _ in global_char.most_common(top_k)]
    lang_list = stats_df["lang"].tolist()
    V = len(vocab)
    X = np.zeros((len(lang_list), V), dtype=np.float64)

    for i, lang in enumerate(lang_list):
        c = lang_char_counters.get(lang, Counter())
        for j, ch in enumerate(vocab):
            X[i, j] = float(c.get(ch, 0))

    # Pairwise JSD
    n = len(lang_list)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _js_divergence(X[i], X[j])
            D[i, j] = d
            D[j, i] = d

    stats_df["avg_jsd_to_others"] = D.mean(axis=1)

    # Save tables
    out_stats = out_dir / f"lang_stats_{args.split}.csv"
    stats_df.to_csv(out_stats, index=False)

    D_df = pd.DataFrame(D, index=lang_list, columns=lang_list)
    out_jsd = out_dir / f"pairwise_jsd_{args.split}.csv"
    D_df.to_csv(out_jsd)

    print(f"[noniid] split={args.split} langs={len(stats_df)}")
    print(f"[noniid] gini(total_hours)={gini_hours:.4f} gini(n_utt)={gini_utts:.4f}")
    print(f"[noniid] wrote {out_stats}")
    print(f"[noniid] wrote {out_jsd}")

    # Bar plots: show only top and bottom languages
    _plot_top_bottom_barh(
        stats_df,
        col="total_hours",
        title=f"Data imbalance by language (hours) [{args.split}] (Gini={gini_hours:.3f})",
        ylabel="Total hours",
        out_path=out_dir / f"hours_top_bottom_{args.split}.png",
        top_k=args.bar_top_k,
        bottom_k=args.bar_bottom_k,
        fig_dpi=args.fig_dpi,
    )

    _plot_top_bottom_barh(
        stats_df,
        col="n_utt",
        title=f"Utterance imbalance by language [{args.split}] (Gini={gini_utts:.3f})",
        ylabel="Utterances",
        out_path=out_dir / f"utts_top_bottom_{args.split}.png",
        top_k=args.bar_top_k,
        bottom_k=args.bar_bottom_k,
        fig_dpi=args.fig_dpi,
    )

    _plot_top_bottom_barh(
        stats_df,
        col="avg_jsd_to_others",
        title=f"Most non-IID languages (avg JSD to others) [{args.split}]",
        ylabel="Avg JSD",
        out_path=out_dir / f"avg_jsd_top_bottom_{args.split}.png",
        top_k=args.bar_top_k,
        bottom_k=args.bar_bottom_k,
        fig_dpi=args.fig_dpi,
    )

    # Heatmap subset selection + label thinning
    subset_langs = _select_lang_subset(stats_df, args.heatmap_max_langs, args.heatmap_subset, seed=0)
    idx = [lang_list.index(l) for l in subset_langs if l in lang_list]
    D_sub = D[np.ix_(idx, idx)]
    _plot_heatmap(
        D_sub,
        subset_langs,
        out_path=out_dir / f"jsd_heatmap_{args.split}_n{subset_langs.__len__()}.png",
        fig_dpi=args.fig_dpi,
        cluster=args.heatmap_cluster,
        max_tick_labels=args.heatmap_max_tick_labels,
    )

    # JSD histogram remains readable, keep it
    vals = D[np.triu_indices(n, k=1)]
    plt.figure(figsize=(6.2, 4.0))
    plt.hist(vals, bins=30)
    plt.xlabel("Pairwise JSD")
    plt.ylabel("Count")
    plt.title(f"Distribution of language divergence (JSD) [{args.split}]")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"jsd_hist_{args.split}.png", dpi=args.fig_dpi)
    plt.close()

    # PCA: annotate only informative points
    Z = _pca_2d(X / (X.sum(axis=1, keepdims=True) + 1e-12))
    _plot_pca(
        Z=Z,
        langs=lang_list,
        stats_df=stats_df,
        out_path=out_dir / f"pca_chars_{args.split}.png",
        fig_dpi=args.fig_dpi,
        annotate_k=args.pca_annotate_k,
        annotate_extremes=args.pca_annotate_extremes,
    )

    print(f"[noniid] plots saved to: {out_dir.resolve()}")
    print("[noniid] recommended paper-friendly plots:")
    print("  - hours_top_bottom_<split>.png for imbalance")
    print("  - avg_jsd_top_bottom_<split>.png for non-IID severity")
    print("  - jsd_heatmap_<split>_nK.png with K around 40 to 70 for readable blocks")
    print("  - pca_chars_<split>.png with limited annotations")


if __name__ == "__main__":
    main()
