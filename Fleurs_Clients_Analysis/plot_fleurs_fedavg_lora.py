#!/usr/bin/env python3
"""Plot FedAvg+LoRA per-language results (FLEURS) from a per-client CSV.

Outputs (in --out_dir):
1) Heatmap snapshot for a chosen round:
   - rows: languages (filtered to --langs_json)
   - cols: metrics
   - colors: per-metric z-score of "worse-ness" within used languages
   - numbers: raw values

2) Tail curves across rounds:
   - mean and tail (worst-k or best-k depending on metric direction)

Example:
  python plot_fleurs_fedavg_lora.py \
       --csv fleurs_fedavg_lora_per_client.csv \
    --langs_json langs_used.json \
    --out_dir fedavg_lora_plots

Notes:
- For color consistency, accuracy metrics are transformed to worse-ness as (1 - acc)
  before z-scoring. Raw numbers remain the original accuracy.
"""

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def tail_mean(vals: List[float], tail_frac: float, higher_is_worse: bool) -> float:
    xs = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return float("nan")
    xs.sort()
    k = max(1, int(math.ceil(tail_frac * len(xs))))
    tail = xs[-k:] if higher_is_worse else xs[:k]
    return float(np.mean(tail))


def pretty_metric_name(m: str) -> str:
    return {
        "eval_utts": "utts",
        "val_cer_mean": "CER",
        "val_wer": "WER",
        "val_word_correct_acc": "word-acc",
        "val_sent_acc": "sent-acc",
        "val_loss": "loss",
    }.get(m, m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Per-client CSV with columns: round, phase, lang_id, val_*")
    ap.add_argument("--langs_json", type=str, required=True, help="JSON with key 'langs' listing used languages")
    ap.add_argument("--out_dir", type=str, default="fedavg_lora_plots")
    ap.add_argument("--phase", type=str, default="eval", choices=["eval", "fit"])
    ap.add_argument("--round", type=str, default="last", help="Round number, or 'last'")
    ap.add_argument("--tail_frac", type=float, default=0.20, help="Tail fraction for curves (example: 0.20 means worst 20%)")
    ap.add_argument("--cmap", type=str, default="PuOr", help="Diverging colormap for z-scores")
    ap.add_argument("--zclip", type=float, default=2.0, help="Clip z-scores to [-zclip, +zclip]")
    ap.add_argument(
        "--metrics",
        type=str,
        default="eval_utts,val_cer_mean,val_wer,val_word_correct_acc,val_sent_acc,val_loss",
        help="Comma-separated metric columns to plot",
    )
    ap.add_argument("--sort_by", type=str, default="eval_utts", help="Metric to sort rows by (descending)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    with open(args.langs_json, "r", encoding="utf-8") as f:
        langs_obj = json.load(f)
    used_langs = langs_obj.get("langs", [])
    used_langs = [str(x).strip() for x in used_langs if str(x).strip()]

    phase_df = df[df["phase"].astype(str).str.lower() == args.phase.lower()].copy()
    phase_df = phase_df[phase_df["lang_id"].astype(str).isin(used_langs)].copy()
    if phase_df.empty:
        raise RuntimeError("No rows after filtering to phase and langs_json")

    if args.round.lower() == "last":
        chosen_round = int(phase_df["round"].max())
    else:
        chosen_round = int(args.round)

    snap = phase_df[phase_df["round"] == chosen_round].copy()
    if snap.empty:
        raise RuntimeError(f"No rows for round={chosen_round}")

    # One row per lang_id
    snap = snap.sort_values(["lang_id", "round"]).groupby("lang_id", as_index=False).tail(1)
    snap = snap.set_index("lang_id")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    missing = [m for m in metrics if m not in snap.columns]
    if missing:
        raise ValueError(f"Missing metric columns in CSV: {missing}")

    keep_langs = [l for l in used_langs if l in set(snap.index)]
    raw = snap.loc[keep_langs, metrics].apply(pd.to_numeric, errors="coerce")

    # Transform to worse-ness for z-score colors (raw remains unchanged)
    worse = raw.copy()
    if "val_word_correct_acc" in worse.columns:
        worse["val_word_correct_acc"] = 1.0 - worse["val_word_correct_acc"]
    if "val_sent_acc" in worse.columns:
        worse["val_sent_acc"] = 1.0 - worse["val_sent_acc"]

    # z-score per metric within used languages
    z = (worse - worse.mean(axis=0)) / worse.std(axis=0, ddof=0)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z = z.clip(-float(args.zclip), float(args.zclip))

    # Sort by requested metric
    sort_key = args.sort_by
    if sort_key not in raw.columns:
        raise ValueError(f"--sort_by={sort_key} not in metrics list: {metrics}")
    raw_sorted = raw.sort_values(sort_key, ascending=False)
    z_sorted = z.loc[raw_sorted.index]

    # Save snapshot table
    snap_out = os.path.join(args.out_dir, f"lang_snapshot_round{chosen_round}.csv")
    raw_sorted.reset_index().to_csv(snap_out, index=False)

    # Heatmap
    col_labels = [pretty_metric_name(m) for m in metrics]
    row_labels = list(raw_sorted.index)

    fig_h = max(5.0, 0.35 * len(row_labels))
    fig_w = 9.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    im = ax.imshow(
        z_sorted.values,
        aspect="auto",
        interpolation="nearest",
        cmap=args.cmap,
        vmin=-float(args.zclip),
        vmax=float(args.zclip),
    )
    ax.set_title(
        f"FedAvg+LoRA per-language snapshot (round={chosen_round})\n"
        f"colors=z-score of worse-ness, numbers=raw"
    )
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate raw values
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            metric = metrics[j]
            val = raw_sorted.iloc[i, j]
            if pd.isna(val):
                txt = ""
            else:
                if metric == "eval_utts":
                    txt = f"{int(round(val))}"
                else:
                    txt = f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}"
            bg = float(z_sorted.values[i, j])
            color = "white" if abs(bg) > 1.0 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("z-score (higher=worse within used langs)")

    plt.tight_layout()
    heat_path = os.path.join(args.out_dir, f"lang_heatmap_round{chosen_round}.png")
    plt.savefig(heat_path, dpi=200)
    plt.close(fig)

    # Tail curves table
    tail_frac = float(args.tail_frac)
    rounds = sorted(phase_df["round"].unique())
    curve_rows: List[Dict[str, float]] = []

    for r in rounds:
        rr = phase_df[phase_df["round"] == r]
        rr = rr[rr["lang_id"].astype(str).isin(used_langs)]

        cer_vals = rr["val_cer_mean"].astype(float).tolist() if "val_cer_mean" in rr.columns else []
        wer_vals = rr["val_wer"].astype(float).tolist() if "val_wer" in rr.columns else []
        wacc_vals = rr["val_word_correct_acc"].astype(float).tolist() if "val_word_correct_acc" in rr.columns else []
        sacc_vals = rr["val_sent_acc"].astype(float).tolist() if "val_sent_acc" in rr.columns else []

        curve_rows.append(
            {
                "round": int(r),
                "mean_CER": float(np.mean(cer_vals)) if cer_vals else float("nan"),
                "tail_CER": tail_mean(cer_vals, tail_frac=tail_frac, higher_is_worse=True),
                "mean_WER": float(np.mean(wer_vals)) if wer_vals else float("nan"),
                "tail_WER": tail_mean(wer_vals, tail_frac=tail_frac, higher_is_worse=True),
                "mean_word_acc": float(np.mean(wacc_vals)) if wacc_vals else float("nan"),
                "tail_word_acc": tail_mean(wacc_vals, tail_frac=tail_frac, higher_is_worse=False),
                "mean_sent_acc": float(np.mean(sacc_vals)) if sacc_vals else float("nan"),
                "tail_sent_acc": tail_mean(sacc_vals, tail_frac=tail_frac, higher_is_worse=False),
            }
        )

    curve_df = pd.DataFrame(curve_rows)
    curve_out = os.path.join(args.out_dir, f"tail_curves_tail{int(round(tail_frac*100))}.csv")
    curve_df.to_csv(curve_out, index=False)

    # CER curve
    if "mean_CER" in curve_df.columns and curve_df["mean_CER"].notna().any():
        fig = plt.figure(figsize=(9.0, 5.0))
        ax = plt.gca()
        ax.plot(curve_df["round"], curve_df["mean_CER"], label="mean CER")
        ax.plot(curve_df["round"], curve_df["tail_CER"], label=f"tail{int(round(tail_frac*100))} CER")
        ax.set_xlabel("round")
        ax.set_ylabel("CER")
        ax.set_title("FedAvg+LoRA CER over rounds (used langs)")
        ax.legend()
        plt.tight_layout()
        cer_path = os.path.join(args.out_dir, f"cer_curve_tail{int(round(tail_frac*100))}.png")
        plt.savefig(cer_path, dpi=200)
        plt.close(fig)

    # WER curve
    if "mean_WER" in curve_df.columns and curve_df["mean_WER"].notna().any():
        fig = plt.figure(figsize=(9.0, 5.0))
        ax = plt.gca()
        ax.plot(curve_df["round"], curve_df["mean_WER"], label="mean WER")
        ax.plot(curve_df["round"], curve_df["tail_WER"], label=f"tail{int(round(tail_frac*100))} WER")
        ax.set_xlabel("round")
        ax.set_ylabel("WER")
        ax.set_title("FedAvg+LoRA WER over rounds (used langs)")
        ax.legend()
        plt.tight_layout()
        wer_path = os.path.join(args.out_dir, f"wer_curve_tail{int(round(tail_frac*100))}.png")
        plt.savefig(wer_path, dpi=200)
        plt.close(fig)

    print("[ok] wrote heatmap:", heat_path)
    print("[ok] wrote snapshot table:", snap_out)
    print("[ok] wrote tail curve table:", curve_out)


if __name__ == "__main__":
    main()
