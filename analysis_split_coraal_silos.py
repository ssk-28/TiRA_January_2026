#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * (cumx / cumx[-1]).sum()) / n)


def parse_fracs(s: str) -> list[float]:
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
            if 0.0 < v < 1.0:
                out.append(v)
        except Exception:
            pass
    return sorted(set(out))


def load_clients(out_dir: str) -> pd.DataFrame:
    pat = os.path.join(out_dir, "clients", "client_*_*.csv")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No client CSVs found under {pat}")

    dfs = []
    for fp in files:
        base = os.path.basename(fp)
        # client_000_train.csv
        parts = base.replace(".csv", "").split("_")
        cid = int(parts[1])
        split = parts[2]
        df = pd.read_csv(fp)
        df["client"] = cid
        df["split"] = split
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    if "duration_s" in all_df.columns:
        all_df["duration_s"] = pd.to_numeric(all_df["duration_s"], errors="coerce").fillna(0.0)
    else:
        all_df["duration_s"] = 0.0
    return all_df


def savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def heatmap(
    matrix: np.ndarray,
    row_labels,
    col_labels,
    title: str,
    out_path: str,
    log1p: bool = False,
    annotate: bool = False,
    annot_fmt: str = "d",
):
    data = np.log1p(matrix) if log1p else matrix
    plt.figure(figsize=(max(7, 0.25 * len(col_labels) + 4), max(6, 0.22 * len(row_labels) + 2)))
    plt.imshow(data, aspect="auto")
    plt.title(title)
    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize=8)
    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right", fontsize=9)
    plt.colorbar()

    if annotate:
        # annotate original values (not log1p)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isfinite(v) and v != 0:
                    if annot_fmt == "d":
                        txt = f"{int(v)}"
                    else:
                        txt = format(float(v), annot_fmt)
                    plt.text(j, i, txt, ha="center", va="center", fontsize=7)

    savefig(out_path)


def rank_bar_with_cum(
    values: np.ndarray,
    title: str,
    ylabel: str,
    out_path: str,
    descending: bool = True,
    tail_fracs: list[float] | None = None,
):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    x = np.maximum(x, 0.0)
    x_sorted = np.sort(x)[::-1] if descending else np.sort(x)
    n = x_sorted.size
    ranks = np.arange(1, n + 1)

    plt.figure(figsize=(9, 4))
    plt.bar(ranks, x_sorted)
    plt.title(title)
    plt.xlabel("Rank (sorted, client id ignored)")
    plt.ylabel(ylabel)

    if tail_fracs:
        # tail is on the right if descending (small values), on the left if ascending
        for rho in tail_fracs:
            if descending:
                k = int(np.floor((1.0 - rho) * n))
                k = max(1, min(n, k))
                plt.axvline(k, linestyle="--")
            else:
                k = int(np.ceil(rho * n))
                k = max(1, min(n, k))
                plt.axvline(k, linestyle="--")

    savefig(out_path)

    # cumulative share plot
    total = float(x_sorted.sum())
    if total > 0:
        cum_share = np.cumsum(x_sorted) / total
        plt.figure(figsize=(9, 4))
        plt.plot(ranks, cum_share)
        plt.ylim(0.0, 1.0)
        plt.title(title + " (cumulative share)")
        plt.xlabel("Rank (sorted, client id ignored)")
        plt.ylabel("Cumulative fraction of total")

        if tail_fracs:
            for rho in tail_fracs:
                if descending:
                    k = int(np.floor((1.0 - rho) * n))
                    k = max(1, min(n, k))
                    plt.axvline(k, linestyle="--")
                else:
                    k = int(np.ceil(rho * n))
                    k = max(1, min(n, k))
                    plt.axvline(k, linestyle="--")

        savefig(out_path.replace(".png", "_cumshare.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="output/partitions_coraal", help="Partition output dir")
    ap.add_argument("--top_sources", type=int, default=12, help="How many sources to show in heatmaps")
    ap.add_argument("--top_races", type=int, default=10, help="How many race_ethnicity values to show")
    ap.add_argument("--tail_fracs", type=str, default="0.10,0.20,0.30", help="Tail fractions for rank plots markers")
    ap.add_argument("--accent_col", type=str, default="accent", help="Column name to use as accent (default: accent)")
    ap.add_argument("--top_accents", type=int, default=12, help="How many accents to keep (by total train minutes)")
    ap.add_argument("--annotate_small_heatmaps", action="store_true", help="Annotate counts for small heatmaps")
    args = ap.parse_args()

    out_dir = args.out_dir
    fig_dir = os.path.join(out_dir, "analysis_figs")
    os.makedirs(fig_dir, exist_ok=True)

    df = load_clients(out_dir)

    # Required fields
    for c in ["client", "split", "duration_s"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c}. Columns: {list(df.columns)}")

    # Summary per client per split
    per = (
        df.groupby(["client", "split"])
        .agg(n_segments=("duration_s", "size"), minutes=("duration_s", lambda x: float(x.sum() / 60.0)))
        .reset_index()
    )

    # Pivot: clients x splits (minutes)
    clients = sorted(per["client"].unique().tolist())
    splits = ["train", "dev", "test"]
    mat_split = np.zeros((len(clients), len(splits)), dtype=float)
    for i, cid in enumerate(clients):
        for j, sp in enumerate(splits):
            v = per[(per.client == cid) & (per.split == sp)]["minutes"]
            mat_split[i, j] = float(v.iloc[0]) if len(v) else 0.0

    # Plot 1: heatmap clients x split (minutes)
    heatmap(
        mat_split,
        row_labels=[f"{c:02d}" for c in clients],
        col_labels=splits,
        title="CORAAL partition: minutes per client per split",
        out_path=os.path.join(fig_dir, "heatmap_client_by_split_minutes.png"),
        log1p=True,
    )

    # Plot 2: client minutes distribution (train) by client id
    train_minutes = mat_split[:, splits.index("train")]
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(clients)), train_minutes)
    plt.title("Train minutes per client")
    plt.xlabel("Client id")
    plt.ylabel("Minutes")
    savefig(os.path.join(fig_dir, "bar_train_minutes_per_client.png"))

    # Plot 2b: sorted (rank) train minutes to show tail shape
    tail_fracs = parse_fracs(args.tail_fracs)
    rank_bar_with_cum(
        train_minutes,
        title="Train minutes per client (sorted)",
        ylabel="Minutes",
        out_path=os.path.join(fig_dir, "rank_train_minutes_sorted.png"),
        descending=True,
        tail_fracs=tail_fracs,
    )

    # Plot 3: train segments per client (by id)
    per_train = per[per["split"] == "train"].set_index("client")
    segs = np.array([float(per_train.loc[c, "n_segments"]) if c in per_train.index else 0.0 for c in clients])
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(clients)), segs)
    plt.title("Train segments per client")
    plt.xlabel("Client id")
    plt.ylabel("Segments")
    savefig(os.path.join(fig_dir, "bar_train_segments_per_client.png"))

    # Plot 3b: sorted (rank) train segments
    rank_bar_with_cum(
        segs,
        title="Train segments per client (sorted)",
        ylabel="Segments",
        out_path=os.path.join(fig_dir, "rank_train_segments_sorted.png"),
        descending=True,
        tail_fracs=tail_fracs,
    )

    # Concentration metrics
    g_train = gini(train_minutes)
    g_all = gini(mat_split.sum(axis=1))
    with open(os.path.join(fig_dir, "inequality_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Gini(train_minutes) = {g_train:.4f}\n")
        f.write(f"Gini(total_minutes_over_splits) = {g_all:.4f}\n")

    # Plot 4: clients x source (minutes, top sources) on train
    if "source" in df.columns:
        dtrain = df[df["split"] == "train"].copy()
        dtrain["source"] = dtrain["source"].fillna("unknown").astype(str)

        src_minutes = dtrain.groupby("source")["duration_s"].sum().sort_values(ascending=False) / 60.0
        top_sources = src_minutes.head(args.top_sources).index.tolist()
        dtrain2 = dtrain[dtrain["source"].isin(top_sources)]

        pivot = dtrain2.groupby(["client", "source"])["duration_s"].sum().unstack(fill_value=0.0) / 60.0
        pivot = pivot.reindex(index=clients, columns=top_sources, fill_value=0.0)
        heatmap(
            pivot.values,
            row_labels=[f"{c:02d}" for c in clients],
            col_labels=top_sources,
            title="Train: minutes per client by source (top sources)",
            out_path=os.path.join(fig_dir, "heatmap_client_by_source_minutes.png"),
            log1p=True,
        )

    # Plot 5: clients x gender (minutes) on train
    if "gender" in df.columns:
        dtrain = df[df["split"] == "train"].copy()
        dtrain["gender"] = dtrain["gender"].fillna("unknown").astype(str)
        genders = ["male", "female", "other", "unknown"]
        pivot = dtrain.groupby(["client", "gender"])["duration_s"].sum().unstack(fill_value=0.0) / 60.0
        pivot = pivot.reindex(index=clients, columns=genders, fill_value=0.0)
        heatmap(
            pivot.values,
            row_labels=[f"{c:02d}" for c in clients],
            col_labels=genders,
            title="Train: minutes per client by gender",
            out_path=os.path.join(fig_dir, "heatmap_client_by_gender_minutes.png"),
            log1p=True,
        )

    # Plot 6: clients x race_ethnicity (minutes, top) on train
    if "race_ethnicity" in df.columns:
        dtrain = df[df["split"] == "train"].copy()
        dtrain["race_ethnicity"] = dtrain["race_ethnicity"].fillna("unknown").astype(str)
        race_minutes = dtrain.groupby("race_ethnicity")["duration_s"].sum().sort_values(ascending=False) / 60.0
        top_races = race_minutes.head(args.top_races).index.tolist()
        dtrain2 = dtrain[dtrain["race_ethnicity"].isin(top_races)]
        pivot = dtrain2.groupby(["client", "race_ethnicity"])["duration_s"].sum().unstack(fill_value=0.0) / 60.0
        pivot = pivot.reindex(index=clients, columns=top_races, fill_value=0.0)
        heatmap(
            pivot.values,
            row_labels=[f"{c:02d}" for c in clients],
            col_labels=top_races,
            title="Train: minutes per client by race_ethnicity (top)",
            out_path=os.path.join(fig_dir, "heatmap_client_by_race_minutes.png"),
            log1p=True,
        )

    # Plot 7: Gender (y) x Accent (x) heatmap where each block represents clients
    # Method: assign each client to a single (gender, accent) by max train minutes, then count clients per cell.
    accent_col = args.accent_col
    if ("gender" in df.columns) and (accent_col in df.columns):
        dtrain = df[df["split"] == "train"].copy()
        dtrain["gender"] = dtrain["gender"].fillna("unknown").astype(str)
        dtrain[accent_col] = dtrain[accent_col].fillna("unknown").astype(str)

        # keep top accents by total minutes for readability (everything else -> other)
        acc_total = dtrain.groupby(accent_col)["duration_s"].sum().sort_values(ascending=False)
        top_acc = acc_total.head(args.top_accents).index.tolist()
        dtrain.loc[~dtrain[accent_col].isin(top_acc), accent_col] = "other"

        # minutes per client x (gender, accent)
        ca = (
            dtrain.groupby(["client", "gender", accent_col])["duration_s"]
            .sum()
            .reset_index()
        )

        # pick dominant (gender, accent) per client by max duration
        ca = ca.sort_values(["client", "duration_s"], ascending=[True, False])
        dominant = ca.groupby("client").head(1).copy()

        # build axes
        genders = sorted(dominant["gender"].unique().tolist())
        accents = sorted(dominant[accent_col].unique().tolist())

        # matrix: count clients
        mat_cnt = np.zeros((len(genders), len(accents)), dtype=int)
        mat_min = np.zeros((len(genders), len(accents)), dtype=float)  # minutes sum of dominant assignment

        g2i = {g: i for i, g in enumerate(genders)}
        a2i = {a: j for j, a in enumerate(accents)}

        for _, r in dominant.iterrows():
            i = g2i.get(r["gender"], None)
            j = a2i.get(r[accent_col], None)
            if i is None or j is None:
                continue
            mat_cnt[i, j] += 1
            mat_min[i, j] += float(r["duration_s"]) / 60.0

        annotate = args.annotate_small_heatmaps and (len(genders) * len(accents) <= 400)

        heatmap(
            mat_cnt,
            row_labels=genders,
            col_labels=accents,
            title=f"Train: client blocks by gender (y) x {accent_col} (x) [count]",
            out_path=os.path.join(fig_dir, f"heatmap_gender_by_{accent_col}_clientcount.png"),
            log1p=False,
            annotate=annotate,
            annot_fmt="d",
        )

        # optional: minutes for those dominant assignments (can show if you want intensity by data mass too)
        heatmap(
            mat_min,
            row_labels=genders,
            col_labels=accents,
            title=f"Train: client blocks by gender (y) x {accent_col} (x) [dominant minutes]",
            out_path=os.path.join(fig_dir, f"heatmap_gender_by_{accent_col}_dominant_minutes.png"),
            log1p=True,
            annotate=False,
        )

    # Save a compact CSV summary
    per_out = per.sort_values(["split", "client"]).copy()
    per_out.to_csv(os.path.join(fig_dir, "client_split_summary.csv"), index=False)

    print("[ok] wrote figures and summaries to:", fig_dir)
    print("[ok] gini(train_minutes) =", round(g_train, 4))
    print("[ok] gini(total_minutes) =", round(g_all, 4))


if __name__ == "__main__":
    main()
