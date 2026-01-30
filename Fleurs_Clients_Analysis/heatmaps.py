#!/usr/bin/env python3
# plot_fleurs_metric_heatmap.py
#
# Colors = per-column z-score across the SELECTED languages only.
# Numbers = raw values.
#
# Requires:
#   client_atlas_fleurs_table.csv (the big table that may include many langs)
#   used_langs.json (the langs you actually trained on)
#
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


def load_used_langs(json_path: str):
    if not json_path or not os.path.isfile(json_path):
        raise FileNotFoundError(f"langs_json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Accept either a list or a dict with several possible keys
    if isinstance(obj, list):
        langs = obj
    elif isinstance(obj, dict):
        for k in ["langs", "used_langs", "languages", "ok_langs", "all_langs"]:
            if k in obj and isinstance(obj[k], list):
                langs = obj[k]
                break
        else:
            raise ValueError(
                f"Could not find language list in {json_path}. "
                f"Expected keys like langs/used_langs/languages/ok_langs."
            )
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}: {type(obj)}")

    langs = [str(x).strip() for x in langs if str(x).strip()]
    if not langs:
        raise ValueError(f"No languages found in {json_path}")
    return langs


def zscore_cols(mat: np.ndarray) -> np.ndarray:
    z = np.zeros_like(mat, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j].astype(float)
        m = float(np.nanmean(col))
        s = float(np.nanstd(col))
        if not np.isfinite(s) or s < 1e-12:
            z[:, j] = 0.0
        else:
            z[:, j] = (col - m) / s
    return z


def fmt_val(col_name: str, v: float) -> str:
    if not np.isfinite(v):
        return ""
    if col_name == "utts":
        return f"{int(round(v))}"
    if col_name == "train_minutes":
        return f"{v:.1f}"
    if col_name in {"mean_dur_s", "p95_dur_s"}:
        return f"{v:.2f}"
    if col_name in {"mean_words", "p95_words"}:
        return f"{v:.1f}"
    return f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_csv", type=str, default="client_atlas_fleurs_table.csv")
    ap.add_argument("--langs_json", type=str, default="langs_used.json")
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--z_clip", type=float, default=2.0)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.table_csv)
    if "lang" not in df.columns:
        raise ValueError("table_csv must contain a 'lang' column")

    used_langs = load_used_langs(args.langs_json)
    df = df[df["lang"].isin(set(used_langs))].copy()

    if df.empty:
        raise RuntimeError(
            "After filtering by used_langs.json, no rows remain. "
            "Check that your lang ids match between the JSON and the CSV."
        )

    # Sort by capacity
    df = df.sort_values(["utts", "train_minutes"], ascending=[False, False]).reset_index(drop=True)

    metric_cols = [
        ("utts", "utts"),
        ("train_minutes", "min"),
        ("mean_dur_s", "mean_s"),
        ("p95_dur_s", "p95_s"),
        ("mean_words", "mean_w"),
        ("p95_words", "p95_w"),
    ]

    missing = [c for c, _ in metric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in table_csv: {missing}")

    raw = df[[c for c, _ in metric_cols]].to_numpy(dtype=float)

    # IMPORTANT: z-scores computed over the SELECTED languages only
    Z = zscore_cols(raw)
    z_clip = float(args.z_clip)
    Zc = np.clip(Z, -z_clip, z_clip)

    langs = df["lang"].tolist()
    y = np.arange(len(langs))
    x = np.arange(len(metric_cols))

    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig_h = max(5.0, 0.32 * len(langs))
    fig_w = 7.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)

    cmap = plt.get_cmap("PuOr_r")
    norm = colors.TwoSlopeNorm(vmin=-z_clip, vcenter=0.0, vmax=z_clip)

    im = ax.imshow(Zc, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    ax.set_yticks(y)
    ax.set_yticklabels(langs)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metric_cols])
    ax.tick_params(axis="both", which="both", length=0)

    for i in range(raw.shape[0]):
        for j, (col_name, _) in enumerate(metric_cols):
            txt = fmt_val(col_name, raw[i, j])
            bg = Zc[i, j]
            txt_color = "white" if abs(bg) > 1.0 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("z-score (within selected languages from FLEURS)")

    ax.set_title(" Clients data distribution summary heatmap")
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, "client_atlas_fleurs_metric_heatmap.png")
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    out_table = os.path.join(args.out_dir, "client_atlas_fleurs_metric_heatmap_table.csv")
    df_out = df[["lang"] + [c for c, _ in metric_cols]].copy()
    for j, (c, _) in enumerate(metric_cols):
        df_out[f"z__{c}"] = Z[:, j]
    df_out.to_csv(out_table, index=False)

    print("[ok] wrote:", out_png)
    print("[ok] wrote:", out_table)


if __name__ == "__main__":
    main()
