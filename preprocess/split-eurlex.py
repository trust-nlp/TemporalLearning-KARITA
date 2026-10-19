#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal splits of EurLex (English portion of MultiEURLEX).

Input: the raw MultiEURLEX files train.jsonl / dev.jsonl / test.jsonl (or .jsonl.gz), from
       https://huggingface.co/datasets/nlpaueb/multi_eurlex (data/multi_eurlex.tar.gz).
       The original train/dev/test split is ignored: all 65k documents are pooled.

Every document is assigned to an interval by its publication year
    T1 1958-1985 | T2 1986-2000 | T3 2001-2010 | T4 2011-2016
and each interval is split 70/30 into train/test with a seed derived from (seed, lang, interval).

Output (<out_dir>, jsonl with .json extension):
    T{1,2,3,4}_train.json, T{1,2,3,4}_test.json   70/30 split of each interval
    T{1,2,3,4}_all.json                           whole interval
    T123_all.json                                 T1_all + T2_all + T3_all (retrieval pool of KARITA)
Record: {celex_id, publication_date, lang, text, level_1, level_2, level_3, all_levels}

Example:
  python preprocess/split-eurlex.py --raw_dir multi_eurlex --out_dir data/EURLEX
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import re
from datetime import datetime

SPLITS = ("train", "dev", "test")
BUCKETS = {
    "T1": (1958, 1985),
    "T2": (1986, 2000),
    "T3": (2001, 2010),
    "T4": (2011, 2016),
}
YEAR_RE = re.compile(r"^(\d{4})")


def open_split(raw_dir, split):
    for name in (f"{split}.jsonl", f"{split}.jsonl.gz"):
        path = os.path.join(raw_dir, name)
        if os.path.isfile(path):
            return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
    raise FileNotFoundError(f"{split}.jsonl(.gz) not found in {raw_dir}")


def extract_year(pubdate):
    if not pubdate:
        return None
    s = str(pubdate).strip()
    try:
        return datetime.fromisoformat(s).year
    except ValueError:
        m = YEAR_RE.match(s)
        return int(m.group(1)) if m else None


def which_bucket(year):
    if year is None:
        return None
    for name, (lo, hi) in BUCKETS.items():
        if lo <= year <= hi:
            return name
    return None


def derived_seed(base_seed, lang, bucket):
    s = f"{base_seed}|{lang}|{bucket}".encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)


def str_list(x):
    """String list, order kept, duplicates removed."""
    seen, out = set(), []
    for i in x or []:
        si = str(i)
        if si not in seen:
            seen.add(si)
            out.append(si)
    return out


def main():
    parser = argparse.ArgumentParser(description="Temporal train/test splits of EurLex")
    parser.add_argument("--raw_dir", required=True, help="Directory with MultiEURLEX train/dev/test.jsonl")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1) pool the MultiEURLEX splits (train, dev, test order) and bucket by publication year
    buckets = {b: [] for b in BUCKETS}
    dropped = 0
    for split in SPLITS:
        with open_split(args.raw_dir, split) as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                bucket = which_bucket(extract_year(obj.get("publication_date")))
                if bucket is None:
                    dropped += 1
                    continue
                txt = (obj.get("text") or {}).get(args.lang)
                if not txt:
                    continue
                ev = obj.get("eurovoc_concepts") or {}
                buckets[bucket].append(json.dumps({
                    "celex_id": obj.get("celex_id"),
                    "publication_date": obj.get("publication_date"),
                    "lang": args.lang,
                    "text": txt,
                    "level_1": str_list(ev.get("level_1")),
                    "level_2": str_list(ev.get("level_2")),
                    "level_3": str_list(ev.get("level_3")),
                    "all_levels": str_list(ev.get("all_levels")),
                }, ensure_ascii=False))
    print(f"Dropped (no year / out of range): {dropped}")

    # 2) 70/30 split inside every interval
    os.makedirs(args.out_dir, exist_ok=True)
    for b in sorted(BUCKETS):
        lines = buckets[b]
        n = len(lines)
        rng = random.Random(derived_seed(args.seed, args.lang, b))
        train_idx = set(rng.sample(range(n), int(round(n * args.train_ratio))))
        with open(os.path.join(args.out_dir, f"{b}_train.json"), "w", encoding="utf-8") as ftrain, \
             open(os.path.join(args.out_dir, f"{b}_test.json"), "w", encoding="utf-8") as ftest, \
             open(os.path.join(args.out_dir, f"{b}_all.json"), "w", encoding="utf-8") as fall:
            for i, line in enumerate(lines):
                (ftrain if i in train_idx else ftest).write(line + "\n")
                fall.write(line + "\n")
        print(f"{b} {BUCKETS[b]}: total={n}, train={len(train_idx)}, test={n - len(train_idx)}")

    # 3) retrieval pool T1-T3 (train + test)
    with open(os.path.join(args.out_dir, "T123_all.json"), "w", encoding="utf-8") as fout:
        for b in ("T1", "T2", "T3"):
            for line in buckets[b]:
                fout.write(line + "\n")
    print(f"Saved splits to {args.out_dir}")


if __name__ == "__main__":
    main()
