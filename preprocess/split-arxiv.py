#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal splits of arXiv-CS.

Input: https://huggingface.co/datasets/nick007x/arxiv-papers (downloaded through `datasets`).

Steps
  1. year            = last 4-digit number in [1990, 2100] of `submission_date`
  2. keep the papers whose primary category is cs.* (`primary_subject`, e.g. "Machine Learning (cs.LG)")
  3. cs_subject_codes = the cs.* codes of `subjects`; labels = the ones among the 30 most frequent
     cs.* codes; papers without any of them are removed
  4. shuffle once (seed 42), assign each paper to
         T1 1991-2010 | T2 2011-2015 | T3 2016-2020 | T4 2021-2025
     and split every interval 70/30 into train/test

Output (<out_dir>, jsonl with .json extension):
    T{1,2,3,4}-train.json, T{1,2,3,4}-test.json
    T123-train.json        T1 + T2 + T3 train (retrieval pool of KARITA)

Example:
  python preprocess/split-arxiv.py --out_dir data/ArXiv_CS
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

BUCKETS = {
    "T1": (1991, 2010),
    "T2": (2011, 2015),
    "T3": (2016, 2020),
    "T4": (2021, 2025),
}


def extract_year(text):
    if not isinstance(text, str):
        return None
    for y in reversed(re.findall(r"\b(\d{4})\b", text)):
        if 1990 <= int(y) <= 2100:
            return int(y)
    return None


def extract_code(primary_subject):
    m = re.search(r"\((.*?)\)", primary_subject)
    return m.group(1) if m else primary_subject


def unique_preserve_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def convert_value(v):
    """numpy / pandas values -> JSON-serialisable Python values."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return [convert_value(x) for x in v.tolist()]
    if isinstance(v, list):
        return [convert_value(x) for x in v]
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def to_lines(df):
    return [json.dumps({k: convert_value(v) for k, v in row.items()}, ensure_ascii=False)
            for row in df.to_dict(orient="records")]


def main():
    parser = argparse.ArgumentParser(description="Temporal train/test splits of arXiv-CS")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--hf_dataset", default="nick007x/arxiv-papers")
    parser.add_argument("--top_k", type=int, default=30, help="number of most frequent cs.* labels kept")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset
    df = load_dataset(args.hf_dataset, split="train").to_pandas()

    df["year"] = df["submission_date"].apply(extract_year)
    df["primary_code"] = df["primary_subject"].apply(extract_code)
    df["subject_codes"] = df["subjects"].apply(lambda s: re.findall(r"\((.*?)\)", s))
    df["primary_major"] = df["primary_code"].apply(lambda c: c.split(".")[0])
    df["major_labels"] = df["subject_codes"].apply(lambda codes: unique_preserve_order([c.split(".")[0] for c in codes]))

    # CS-primary papers, top-k cs.* labels
    df_cs = df[df["primary_major"] == "cs"].copy()
    df_cs["cs_subject_codes"] = df_cs["subject_codes"].apply(lambda codes: [c for c in codes if c.startswith("cs.")])
    kept = set(df_cs["cs_subject_codes"].explode().value_counts().head(args.top_k).index)
    df_cs["labels"] = df_cs["cs_subject_codes"].apply(lambda codes: [c for c in codes if c in kept])
    df_cs = df_cs[df_cs["labels"].map(len) > 0]
    print(f"{len(df_cs)} CS papers, top-{args.top_k} labels: {sorted(kept)}")

    # shuffle once, then 70/30 inside each interval
    os.makedirs(args.out_dir, exist_ok=True)
    df_shuf = df_cs.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    pool = []
    for name, (start, end) in BUCKETS.items():
        lines = to_lines(df_shuf[(df_shuf["year"] >= start) & (df_shuf["year"] <= end)])
        n_train = int(len(lines) * args.train_ratio)
        for split, part in (("train", lines[:n_train]), ("test", lines[n_train:])):
            with open(os.path.join(args.out_dir, f"{name}-{split}.json"), "w", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in part)
        if name != "T4":
            pool.extend(lines[:n_train])
        print(f"{name} {BUCKETS[name]}: total={len(lines)}, train={n_train}, test={len(lines) - n_train}")

    with open(os.path.join(args.out_dir, "T123-train.json"), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in pool)
    print(f"Saved splits to {args.out_dir}")


if __name__ == "__main__":
    main()
