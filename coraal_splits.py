#!/usr/bin/env python3
# split_coraal_silos_from_disk.py
#
# CORAAL on-disk splitter for ASR experiments.
#
# Adds:
# - --alphas: run multiple Dirichlet alphas (more non-IID when alpha is smaller)
# - per-alpha output folders: <out_dir>_alpha{val}
# - extra Gini metrics saved in mappings/inequality_metrics.txt (minutes per client, plus source/gender gini)

from __future__ import annotations

import os
import re
import csv
import json
import math
import argparse
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# Helpers
# -------------------------
def is_na(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str):
        s = x.strip().lower()
        return s in {"", "na", "n/a", "none", "null", "nan", "unknown", "unspecified"}
    return False


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def canon_gender(x: Any) -> str:
    if is_na(x):
        return "unknown"
    s = norm_ws(str(x)).lower()
    if s in {"m", "male", "man"}:
        return "male"
    if s in {"f", "female", "woman"}:
        return "female"
    return "other"


def canon_group(x: Any) -> str:
    if is_na(x):
        return "unknown"
    s = norm_ws(str(x))
    return s if s else "unknown"


def to_float(x: Any) -> Optional[float]:
    if is_na(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        rows = [row for row in r]
    return header, rows


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def fmt_alpha(a: float) -> str:
    # folder-safe formatting: 0.1 -> 0p10, 1.0 -> 1p00
    s = f"{a:.3f}"
    s = s.rstrip("0").rstrip(".") if "." in s else s
    # still keep 2 decimals for readability in folder naming
    s2 = f"{a:.2f}".replace(".", "p")
    return s2


def parse_alphas(s: str) -> List[float]:
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
            if v > 0:
                out.append(v)
        except Exception:
            pass
    # keep stable, unique, sorted
    return sorted(set(out))


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= 0.0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * (cumx / cumx[-1]).sum()) / n)


# -------------------------
# Data model
# -------------------------
@dataclass
class UttRow:
    row_id: int
    basefile: str
    segment_filename: str
    audio_path: str
    text: str
    dur_s: float
    gender: str
    group: str
    age: str
    race_ethnicity: str


def make_audio_path(audio_root: str, seg: str) -> str:
    if os.path.isabs(seg):
        return seg
    return os.path.join(audio_root, seg)


def build_utterances(rows: List[Dict[str, str]], audio_root: str) -> List[UttRow]:
    out: List[UttRow] = []
    for i, r in enumerate(rows):
        seg = r.get("segment_filename", "")
        base = r.get("basefile", "")
        txt = r.get("content", "")

        if is_na(seg) or is_na(base) or is_na(txt):
            continue

        seg = norm_ws(str(seg))
        base = norm_ws(str(base))
        txt = norm_ws(str(txt))

        dur = to_float(r.get("duration", "")) or 0.0
        gender = canon_gender(r.get("gender", ""))
        group = canon_group(r.get("source", ""))

        age = "" if is_na(r.get("age", "")) else norm_ws(str(r.get("age")))
        race = "" if is_na(r.get("race_ethnicity", "")) else norm_ws(str(r.get("race_ethnicity")))

        apath = make_audio_path(audio_root, seg)

        out.append(
            UttRow(
                row_id=i,
                basefile=base,
                segment_filename=seg,
                audio_path=apath,
                text=txt,
                dur_s=float(dur),
                gender=gender,
                group=group,
                age=age,
                race_ethnicity=race,
            )
        )
    return out


def split_within_basefile(
    utts_by_basefile: Dict[str, List[UttRow]],
    seed: int,
    frac_train: float,
    frac_dev: float,
    frac_test: float,
    use_time_order: bool,
) -> Dict[int, str]:
    rng = np.random.default_rng(seed)
    rowid_to_split: Dict[int, str] = {}

    for base, lst in utts_by_basefile.items():
        if not lst:
            continue

        if use_time_order:
            lst_sorted = sorted(lst, key=lambda x: (x.segment_filename, x.row_id))
        else:
            lst_sorted = lst[:]
            rng.shuffle(lst_sorted)

        n = len(lst_sorted)
        n_train = int(math.floor(frac_train * n))
        n_dev = int(math.floor(frac_dev * n))
        n_test = n - n_train - n_dev

        if n >= 3:
            if n_train == 0:
                n_train = 1
                n_test = max(0, n_test - 1)
            if n_dev == 0:
                n_dev = 1
                n_test = max(0, n_test - 1)
            if n_test == 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                elif n_dev > 1:
                    n_dev -= 1

        for j, u in enumerate(lst_sorted):
            if j < n_train:
                rowid_to_split[u.row_id] = "train"
            elif j < n_train + n_dev:
                rowid_to_split[u.row_id] = "dev"
            else:
                rowid_to_split[u.row_id] = "test"

    return rowid_to_split


# -------------------------
# Partition basefiles into K silos with hierarchical Dirichlet
# -------------------------
def _dirichlet(alpha_vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = rng.gamma(shape=np.maximum(alpha_vec, 1e-8), scale=1.0)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)


def partition_basefiles_hdir(
    basefiles: List[str],
    basefile_meta: Dict[str, Tuple[str, str, float]],  # base -> (group, gender, minutes)
    K: int,
    alpha_group: float,
    alpha_gender: float,
    seed: int,
    min_speakers: int,
    min_minutes: float,
) -> Tuple[Dict[str, int], Dict[int, dict]]:
    rng = np.random.default_rng(seed)

    N = len(basefiles)

    max_min_speakers = max(1, N // K)
    if min_speakers > max_min_speakers:
        print(
            f"[warn] min_speakers={min_speakers} infeasible for N={N}, K={K}. "
            f"Clamping to {max_min_speakers}."
        )
        min_speakers = max_min_speakers

    if K > N:
        print(f"[warn] K={K} > N={N}. Clamping K to {N}.")
        K = N

    groups = sorted({basefile_meta[b][0] for b in basefiles})
    genders = ["male", "female", "other", "unknown"]

    c_grp = Counter([basefile_meta[b][0] for b in basefiles])
    q_grp = np.array([c_grp[g] for g in groups], dtype=np.float64)
    q_grp = q_grp / max(q_grp.sum(), 1.0)

    c_gen = Counter([basefile_meta[b][1] for b in basefiles])
    q_gen = np.array([c_gen[g] for g in genders], dtype=np.float64)
    q_gen = q_gen / max(q_gen.sum(), 1.0)

    pool: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for b in basefiles:
        pool[(basefile_meta[b][0], basefile_meta[b][1])].append(b)
    for kk in list(pool.keys()):
        rng.shuffle(pool[kk])

    base = N // K
    targets = [base] * K
    for i in range(N - base * K):
        targets[i] += 1

    keys = [(g, s) for g in groups for s in genders]

    client_pi: List[Dict[Tuple[str, str], float]] = []
    for _ in range(K):
        theta = _dirichlet(alpha_group * q_grp, rng)
        phi = {grp: _dirichlet(alpha_gender * q_gen, rng) for grp in groups}
        pi: Dict[Tuple[str, str], float] = {}
        for gi, grp in enumerate(groups):
            for sj, gen in enumerate(genders):
                pi[(grp, gen)] = float(theta[gi] * phi[grp][sj])
        z = sum(pi.values())
        for kk in pi:
            pi[kk] /= max(z, 1e-12)
        client_pi.append(pi)

    client_lists: List[List[str]] = [[] for _ in range(K)]

    for k in range(K):
        pvec = np.array([client_pi[k][kk] for kk in keys], dtype=np.float64)
        pvec = pvec / max(pvec.sum(), 1e-12)
        counts = rng.multinomial(targets[k], pvec)
        for kk, c in zip(keys, counts):
            if c <= 0:
                continue
            avail = pool.get(kk, [])
            take = min(c, len(avail))
            if take > 0:
                client_lists[k].extend(avail[:take])
                pool[kk] = avail[take:]

    remaining: List[str] = []
    for avail in pool.values():
        remaining.extend(avail)
    rng.shuffle(remaining)

    def deficit(k: int) -> int:
        return max(0, targets[k] - len(client_lists[k]))

    idx = 0
    while idx < len(remaining):
        ds = sorted([(deficit(k), k) for k in range(K)], reverse=True)
        if ds[0][0] <= 0:
            break
        k = ds[0][1]
        client_lists[k].append(remaining[idx])
        idx += 1

    small = [k for k in range(K) if len(client_lists[k]) < min_speakers]
    if small:
        donors = sorted(range(K), key=lambda k: len(client_lists[k]), reverse=True)
        for ks in small:
            while len(client_lists[ks]) < min_speakers:
                donor = next((d for d in donors if len(client_lists[d]) > min_speakers), None)
                if donor is None:
                    break
                client_lists[ks].append(client_lists[donor].pop())

    def minutes(k: int) -> float:
        return sum(basefile_meta[b][2] for b in client_lists[k])

    if min_minutes > 0:
        for k in range(K):
            tries = 0
            while minutes(k) < min_minutes and tries < 10000:
                donor = max(range(K), key=lambda d: minutes(d))
                if donor == k or minutes(donor) <= min_minutes:
                    break
                if not client_lists[donor]:
                    break
                client_lists[k].append(client_lists[donor].pop())
                tries += 1

    basefile_to_client: Dict[str, int] = {}
    summary: Dict[int, dict] = {}
    for k in range(K):
        grp_counts = Counter([(basefile_meta[b][0], basefile_meta[b][1]) for b in client_lists[k]])
        for b in client_lists[k]:
            basefile_to_client[b] = k
        summary[k] = {
            "n_speakers": len(client_lists[k]),
            "minutes": float(minutes(k)),
            "group_counts": {f"{grp}__{gen}": int(n) for (grp, gen), n in grp_counts.items()},
        }

    return basefile_to_client, summary


# -------------------------
# Write per-client CSVs
# -------------------------
def write_clients(
    utts: List[UttRow],
    rowid_to_split: Dict[int, str],
    basefile_to_client: Dict[str, int],
    out_dir: str,
    K: int,
    check_audio: bool,
) -> Dict[str, dict]:
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "mappings"))
    ensure_dir(os.path.join(out_dir, "clients"))

    fieldnames = [
        "client",
        "basefile",
        "segment_filename",
        "audio_path",
        "duration_s",
        "text",
        "source",
        "gender",
        "age",
        "race_ethnicity",
        "row_id",
    ]

    writers: Dict[Tuple[int, str], csv.DictWriter] = {}
    handles: Dict[Tuple[int, str], Any] = {}

    for split in ["train", "dev", "test"]:
        for cid in range(K):
            fp = os.path.join(out_dir, "clients", f"client_{cid:03d}_{split}.csv")
            f = open(fp, "w", encoding="utf-8", newline="")
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            writers[(cid, split)] = w
            handles[(cid, split)] = f

    stats = {
        "train": {"rows": 0, "seconds": 0.0},
        "dev": {"rows": 0, "seconds": 0.0},
        "test": {"rows": 0, "seconds": 0.0},
        "skipped": {"no_client": 0, "no_split": 0, "missing_audio": 0},
    }

    for u in utts:
        split = rowid_to_split.get(u.row_id)
        if split is None:
            stats["skipped"]["no_split"] += 1
            continue
        cid = basefile_to_client.get(u.basefile)
        if cid is None:
            stats["skipped"]["no_client"] += 1
            continue
        if check_audio and (not u.audio_path or not os.path.exists(u.audio_path)):
            stats["skipped"]["missing_audio"] += 1
            continue

        writers[(cid, split)].writerow(
            {
                "client": cid,
                "basefile": u.basefile,
                "segment_filename": u.segment_filename,
                "audio_path": u.audio_path,
                "duration_s": f"{u.dur_s:.3f}",
                "text": u.text,
                "source": u.group,
                "gender": u.gender,
                "age": u.age,
                "race_ethnicity": u.race_ethnicity,
                "row_id": u.row_id,
            }
        )

        stats[split]["rows"] += 1
        stats[split]["seconds"] += float(u.dur_s)

    for f in handles.values():
        f.close()

    return stats


# -------------------------
# Extra non-IID metrics (Gini)
# -------------------------
def compute_and_write_ginis(
    out_dir: str,
    K: int,
    basefiles: List[str],
    basefile_meta: Dict[str, Tuple[str, str, float]],  # (group, gender, minutes)
    basefile_to_client: Dict[str, int],
) -> None:
    client_total = np.zeros(K, dtype=float)

    # group and gender minutes per client
    groups = sorted({basefile_meta[b][0] for b in basefiles})
    genders = ["male", "female", "other", "unknown"]

    grp_mat = np.zeros((len(groups), K), dtype=float)
    gen_mat = np.zeros((len(genders), K), dtype=float)
    g2i = {g: i for i, g in enumerate(groups)}
    s2i = {s: i for i, s in enumerate(genders)}

    for b in basefiles:
        cid = basefile_to_client.get(b, None)
        if cid is None or cid < 0 or cid >= K:
            continue
        grp, gen, mins = basefile_meta[b]
        m = float(mins)
        client_total[cid] += m
        grp_mat[g2i[grp], cid] += m
        gen_mat[s2i.get(gen, s2i["unknown"]), cid] += m

    g_total = gini(client_total)

    # For each attribute value, gini across clients of minutes in that value.
    grp_ginis = {groups[i]: gini(grp_mat[i, :]) for i in range(len(groups))}
    gen_ginis = {genders[i]: gini(gen_mat[i, :]) for i in range(len(genders))}

    # Also a single weighted summary gini (weighted by total minutes in that category)
    grp_weights = grp_mat.sum(axis=1)
    gen_weights = gen_mat.sum(axis=1)
    grp_wavg = float(
        np.sum(np.array(list(grp_ginis.values())) * (grp_weights / max(grp_weights.sum(), 1e-12)))
    ) if grp_weights.sum() > 0 else float("nan")
    gen_wavg = float(
        np.sum(np.array(list(gen_ginis.values())) * (gen_weights / max(gen_weights.sum(), 1e-12)))
    ) if gen_weights.sum() > 0 else float("nan")

    p = os.path.join(out_dir, "mappings", "inequality_metrics.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"Gini(total_minutes_per_client) = {g_total:.4f}\n")
        f.write(f"WeightedAvgGini(source_minutes_per_client) = {grp_wavg:.4f}\n")
        f.write(f"WeightedAvgGini(gender_minutes_per_client) = {gen_wavg:.4f}\n")
        f.write("\nPer-source Gini (minutes across clients):\n")
        for k in sorted(grp_ginis.keys()):
            f.write(f"  {k}: {grp_ginis[k]:.4f}\n")
        f.write("\nPer-gender Gini (minutes across clients):\n")
        for k in genders:
            f.write(f"  {k}: {gen_ginis[k]:.4f}\n")


# -------------------------
# Main driver for one alpha
# -------------------------
def run_once(args, alpha: float) -> str:
    root = os.path.abspath(args.root)
    transcripts_csv = os.path.join(root, args.transcripts_csv)
    audio_root = os.path.join(root, args.audio_root)

    if not os.path.exists(transcripts_csv):
        raise FileNotFoundError(f"Missing transcripts_csv: {transcripts_csv}")

    header, rows = read_csv_rows(transcripts_csv)

    required = {
        "segment_filename",
        "basefile",
        "age",
        "gender",
        "source",
        "duration",
        "race_ethnicity",
        "content",
    }
    missing = [c for c in required if c not in set(header)]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {header}")

    utts = build_utterances(rows, audio_root=audio_root)
    if not utts:
        raise RuntimeError("No usable utterances. Check that basefile and content are populated.")

    utts_by_basefile: Dict[str, List[UttRow]] = defaultdict(list)
    for u in utts:
        utts_by_basefile[u.basefile].append(u)

    basefile_meta: Dict[str, Tuple[str, str, float]] = {}
    for base, lst in utts_by_basefile.items():
        grp = Counter([u.group for u in lst]).most_common(1)[0][0]
        gen = Counter([u.gender for u in lst]).most_common(1)[0][0]
        mins = float(sum(u.dur_s for u in lst) / 60.0)
        basefile_meta[base] = (grp, gen, mins)

    basefiles = sorted(list(utts_by_basefile.keys()))

    # output folder includes alpha tag
    out_dir = os.path.join(root, f"{args.out_dir}_alpha{fmt_alpha(alpha)}")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "mappings"))

    basefile_to_client, client_summary = partition_basefiles_hdir(
        basefiles=basefiles,
        basefile_meta=basefile_meta,
        K=args.K,
        alpha_group=alpha,
        alpha_gender=alpha,
        seed=args.seed,
        min_speakers=args.min_speakers,
        min_minutes=args.min_minutes,
    )

    rowid_to_split = split_within_basefile(
        utts_by_basefile=utts_by_basefile,
        seed=args.seed,
        frac_train=args.frac_train,
        frac_dev=args.frac_dev,
        frac_test=args.frac_test,
        use_time_order=args.use_time_order,
    )

    with open(os.path.join(out_dir, "mappings", "basefile_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {b: {"group": g, "gender": s, "minutes": m} for b, (g, s, m) in basefile_meta.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(os.path.join(out_dir, "mappings", "basefile_to_client.json"), "w", encoding="utf-8") as f:
        json.dump(basefile_to_client, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "mappings", "client_summary.json"), "w", encoding="utf-8") as f:
        json.dump(client_summary, f, indent=2, ensure_ascii=False)

    split_stats = write_clients(
        utts=utts,
        rowid_to_split=rowid_to_split,
        basefile_to_client=basefile_to_client,
        out_dir=out_dir,
        K=args.K,
        check_audio=args.check_audio,
    )

    with open(os.path.join(out_dir, "mappings", "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2, ensure_ascii=False)

    # Write extra gini metrics
    compute_and_write_ginis(
        out_dir=out_dir,
        K=args.K,
        basefiles=basefiles,
        basefile_meta=basefile_meta,
        basefile_to_client=basefile_to_client,
    )

    print("[done] out_dir:", out_dir)
    print("[done] alpha:", alpha)
    print("[done] total_basefiles:", len(basefiles))
    print("[done] clients_K:", args.K)
    print("[done] split_stats:", split_stats)
    print("[done] example client file:", os.path.join(out_dir, "clients", "client_000_train.csv"))
    return out_dir


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="CORAAL_asr-disparities-master")
    ap.add_argument("--transcripts_csv", type=str, default="asr-disparities-master/input/CORAAL_transcripts_wav.csv")
    ap.add_argument("--audio_root", type=str, default="asr-disparities-master/input/CORAAL_audio")
    ap.add_argument("--audio_ext", type=str, default=".wav")

    ap.add_argument("--out_dir", type=str, default="output/partitions_coraal")

    # Stable client count for N~73 basefiles
    ap.add_argument("--K", type=int, default=20)

    # If you use --alphas, these are ignored anyway, but keep consistent
    ap.add_argument("--alpha_group", type=float, default=30.0)
    ap.add_argument("--alpha_gender", type=float, default=30.0)

    # Run a single "near-natural" split by default (avoid alpha=0)
    ap.add_argument(
        "--alphas",
        type=str,
        default="30",
        help="Comma-separated list of alphas to run. If set, overrides alpha_group and alpha_gender for runs.",
    )

    # Feasibility: avoid tiny tail clients (you saw ~8-12 min before in other runs)
    ap.add_argument("--min_speakers", type=int, default=5)
    ap.add_argument("--min_minutes", type=float, default=20.0)

    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--frac_train", type=float, default=0.8)
    ap.add_argument("--frac_dev", type=float, default=0.1)
    ap.add_argument("--frac_test", type=float, default=0.1)

    # Optional: keep false unless you explicitly want time-ordered splits
    ap.add_argument("--use_time_order", action="store_true")

    # I suggest enabling this for one run to catch path issues early
    ap.add_argument("--check_audio", action="store_true")

    # Compatibility args (unused)
    ap.add_argument("--speaker_col", type=str, default="")
    ap.add_argument("--text_col", type=str, default="")
    ap.add_argument("--recording_col", type=str, default="")
    ap.add_argument("--audio_path_col", type=str, default="")
    ap.add_argument("--start_col", type=str, default="")
    ap.add_argument("--end_col", type=str, default="")
    ap.add_argument("--duration_col", type=str, default="")
    ap.add_argument("--gender_col", type=str, default="")
    ap.add_argument("--group_col", type=str, default="")
    ap.add_argument("--default_group", type=str, default="coraal")

    args = ap.parse_args()

    if args.alphas.strip():
        alphas = parse_alphas(args.alphas)
        if not alphas:
            raise ValueError(f"--alphas was set but parsed empty: {args.alphas}")
        for a in alphas:
            run_once(args, a)
    else:
        # single run with existing two-level params, but still write gini metrics
        # to a folder that includes both values to avoid overwriting if you vary them manually
        # if you prefer the old behavior, replace this out_dir naming with args.out_dir directly
        a = float(args.alpha_group)
        # if user runs single alpha_group/alpha_gender, keep folder distinct anyway
        args2 = args
        args2.alphas = ""
        run_once(args2, a)


if __name__ == "__main__":
    main()
