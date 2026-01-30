#!/usr/bin/env python3
# splits.py
#
# Build per-language split CSVs in each language directory for ML_SUPERB FLEURS layout.
# Output files created inside each language dir:
#   train.csv, validation.csv, test.csv
#
# Each CSV has columns: path,text,uid,duration_sec
#
# Robustness:
# - Recursively indexes audio files under each language directory (works even if audio is nested)
# - Rebuilds manifests if existing CSV has 0 valid rows (missing audio paths, empty text, etc)
# - Can drop languages that fail to build usable train+validation manifests
#
# Usage examples:
#   python splits.py --data_root ML_SUPERB/fleurs --num_langs 8 --force_rebuild
#   python splits.py --data_root ML_SUPERB/fleurs --langs afr,amh,ara
#
import os
import re
import csv
import glob
import json
import argparse
import unicodedata
from typing import Dict, List, Optional, Tuple

import torchaudio


_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".webm")
_AUDIO_INDEX_CACHE: Dict[str, Dict[str, str]] = {}

_PARENS = re.compile(r"\([^)]*\)")
_BAD = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_transcript(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _PARENS.sub(" ", s)
    s = _BAD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def discover_langs(data_root: str, exclude: set) -> List[str]:
    langs: List[str] = []
    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if os.path.isdir(p) and name not in exclude:
            langs.append(name)
    return langs


def build_audio_index(lang_dir: str) -> Dict[str, str]:
    if lang_dir in _AUDIO_INDEX_CACHE:
        return _AUDIO_INDEX_CACHE[lang_dir]

    idx: Dict[str, str] = {}
    for ext in _AUDIO_EXTS:
        for p in glob.glob(os.path.join(lang_dir, "**", f"*{ext}"), recursive=True):
            ap = os.path.abspath(p)
            base = os.path.basename(ap)
            stem = os.path.splitext(base)[0]
            idx.setdefault(base, ap)
            idx.setdefault(stem, ap)
    _AUDIO_INDEX_CACHE[lang_dir] = idx
    return idx


def resolve_audio_path(lang_dir: str, token: str) -> Optional[str]:
    token = (token or "").strip()
    if not token:
        return None

    if os.path.isabs(token) and os.path.isfile(token):
        return token

    rel = os.path.join(lang_dir, token)
    if os.path.isfile(rel):
        return os.path.abspath(rel)

    idx = build_audio_index(lang_dir)
    base = os.path.basename(token)
    stem = os.path.splitext(base)[0]

    if base in idx:
        return idx[base]
    if stem in idx:
        return idx[stem]
    if token in idx:
        return idx[token]

    return None


def _audio_duration_sec_fast(path: str) -> Optional[float]:
    try:
        info = torchaudio.info(path)
        if info.num_frames and info.sample_rate:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        return None
    return None


def pick_transcript_file(lang_dir: str, split: str) -> Optional[str]:
    # Prefer 10min, then 1h, then any transcript_*_{split}.txt
    if split == "train":
        prefs = ["transcript_10min_train.txt", "transcript_1h_train.txt"]
        pats = ["transcript_*_train.txt"]
    elif split in ["validation", "dev", "val"]:
        prefs = ["transcript_10min_dev.txt", "transcript_1h_dev.txt"]
        pats = ["transcript_*_dev.txt"]
    elif split == "test":
        prefs = ["transcript_10min_test.txt", "transcript_1h_test.txt"]
        pats = ["transcript_*_test.txt"]
    else:
        return None

    for fn in prefs:
        p = os.path.join(lang_dir, fn)
        if os.path.isfile(p):
            return p

    for pat in pats:
        hits = sorted(glob.glob(os.path.join(lang_dir, pat)))
        if hits:
            return hits[0]

    return None


def build_split_csv_from_transcript(lang_id: str, lang_dir: str, split: str, out_csv: str) -> int:
    tpath = pick_transcript_file(lang_dir, split)
    if tpath is None:
        return 0

    build_audio_index(lang_dir)

    rows: List[Dict[str, str]] = []
    seen = set()

    with open(tpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Most ML_SUPERB dumps follow: token something transcription(with spaces)
            parts = line.split(maxsplit=2)
            if len(parts) < 3:
                continue

            wav_token = parts[0]
            text = normalize_transcript(parts[2])
            if not text:
                continue

            ap = resolve_audio_path(lang_dir, wav_token)
            if ap is None:
                continue

            if ap in seen:
                continue
            seen.add(ap)

            dur = _audio_duration_sec_fast(ap)
            rows.append(
                {
                    "path": ap,
                    "text": text,
                    "uid": os.path.splitext(os.path.basename(ap))[0],
                    "duration_sec": "" if dur is None else f"{dur:.3f}",
                }
            )

    if not rows:
        return 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "text", "uid", "duration_sec"])
        w.writeheader()
        w.writerows(rows)

    return len(rows)


def validate_manifest(csv_path: str, lang_dir: str, max_show: int = 3) -> Tuple[int, int, List[str]]:
    ok = 0
    bad = 0
    examples: List[str] = []

    if not os.path.isfile(csv_path):
        return 0, 0, examples

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            p = (r.get("path") or "").strip()
            tx = normalize_transcript(r.get("text") or "")
            if not p or not tx:
                bad += 1
                if len(examples) < max_show:
                    examples.append("empty path/text")
                continue
            if not os.path.isabs(p):
                p = os.path.join(lang_dir, p)
            if not os.path.isfile(p):
                bad += 1
                if len(examples) < max_show:
                    examples.append(f"missing file {p}")
                continue
            ok += 1

    return ok, bad, examples


def ensure_language_csv_splits(
    lang_id: str,
    lang_dir: str,
    force_rebuild: bool = False,
    fail_if_missing_dev: bool = True,
) -> Dict[str, int]:
    stats: Dict[str, int] = {}

    split_to_csv = {
        "train": os.path.join(lang_dir, "train.csv"),
        "validation": os.path.join(lang_dir, "validation.csv"),
        "test": os.path.join(lang_dir, "test.csv"),
    }

    for split, csv_path in split_to_csv.items():
        required = (split != "test")

        rebuild = force_rebuild
        if os.path.isfile(csv_path) and not force_rebuild:
            ok, _, ex = validate_manifest(csv_path, lang_dir)
            if ok == 0:
                rebuild = True
                print(f"[manifests][warn] {lang_id} {split}.csv has 0 valid rows, rebuilding. examples={ex}")

        if (not os.path.isfile(csv_path)) or rebuild:
            n = build_split_csv_from_transcript(lang_id, lang_dir, split, csv_path)
            stats[split] = n
        else:
            stats[split] = -1  # exists and valid

        if required and stats[split] == 0:
            if split == "validation" and not fail_if_missing_dev:
                continue
            raise FileNotFoundError(f"Could not build usable {split}.csv for {lang_id} in {lang_dir}")

    return stats


def build_manifests_for_langs(
    data_root: str,
    langs: List[str],
    force_rebuild: bool,
    fail_fast: bool,
) -> List[str]:
    ok_langs: List[str] = []
    for lang_id in langs:
        lang_dir = os.path.join(data_root, lang_id)
        try:
            st = ensure_language_csv_splits(
                lang_id=lang_id,
                lang_dir=lang_dir,
                force_rebuild=force_rebuild,
                fail_if_missing_dev=True,
            )
            msg = []
            for k, v in st.items():
                msg.append(f"{k}:exists" if v == -1 else f"{k}:built({v})")
            print(f"[manifests] {lang_id} " + " ".join(msg))
            ok_langs.append(lang_id)
        except Exception as e:
            print(f"[manifests][DROP] {lang_id} {e}")
            if fail_fast:
                raise
    print(f"[manifests] usable_langs={len(ok_langs)} dropped={len(langs) - len(ok_langs)}")
    return ok_langs


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="../ML_SUPERB/fleurs")
    ap.add_argument("--langs", type=str, default="")
    ap.add_argument("--num_langs", type=int, default=30)
    ap.add_argument("--exclude_langs", type=str, default="cmn,jpn,kor,zh_cn,ja_jp,ko_kr")

    ap.add_argument("--force_rebuild", action="store_true")
    ap.add_argument("--fail_fast", action="store_true")
    ap.add_argument("--write_langs_json", type=str, default="langs_used.json")

    args = ap.parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root not found: {args.data_root}")

    exclude = {x.strip() for x in args.exclude_langs.split(",") if x.strip()}

    if args.langs.strip():
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    else:
        discovered = discover_langs(args.data_root, exclude=exclude)
        langs = discovered[: max(1, args.num_langs)]

    langs = [l for l in langs if l not in exclude]
    if not langs:
        raise RuntimeError("No languages selected")

    ok_langs = build_manifests_for_langs(
        data_root=args.data_root,
        langs=langs,
        force_rebuild=args.force_rebuild,
        fail_fast=args.fail_fast,
    )

    if args.write_langs_json:
        with open(args.write_langs_json, "w", encoding="utf-8") as f:
            json.dump({"data_root": args.data_root, "langs": ok_langs}, f, indent=2)
        print(f"[manifests] wrote {args.write_langs_json}")

    if not ok_langs:
        raise RuntimeError("No usable languages after manifest build")


if __name__ == "__main__":
    main()
