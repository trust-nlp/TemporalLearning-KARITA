#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Select the top-rho fraction of target samples by a shift score.

  feature : feature_shift_scores.tsv          (column F_x)    -> uid_feature_shift-{rho}%.txt
  ontology: ontology4_tail_shift_scores.tsv   (column O_tail) -> uid_ontology_shift-tail-{rho}%.txt

The output uid files are what karita.py reads via --feature_shift_uids / --ontology_shift_uids.

Example:
  python shift_detection/select-top-shift.py \
      --scores $MODEL_PATH/shift_detector/feature/feature_shift_scores.tsv --rho 0.1
"""

import os
import argparse

import pandas as pd

# score column -> output file prefix
SCORE_COLUMNS = {
    "F_x": "uid_feature_shift",
    "O_tail": "uid_ontology_shift-tail",
}


def main():
    parser = argparse.ArgumentParser(description="Select the top-rho shifted target samples")
    parser.add_argument("--scores", required=True, help="Score tsv written by detect-*-shift.py")
    parser.add_argument("--rho", type=float, nargs="+", default=[0.1],
                        help="One or more fractions, e.g. --rho 0.05 0.1 0.2 0.3")
    parser.add_argument("--out_dir", default=None, help="Default: directory of --scores")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.scores))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.scores, sep=r"\s+", engine="python", dtype={"uid": str})
    score_col = next((c for c in SCORE_COLUMNS if c in df.columns), None)
    if score_col is None:
        raise ValueError(f"{args.scores} has none of the score columns {list(SCORE_COLUMNS)}")

    # larger score = more shifted
    df_sorted = df.sort_values(by=score_col, ascending=False)
    for rho in args.rho:
        top_k = max(1, int(len(df_sorted) * rho))
        out_file = os.path.join(out_dir, f"{SCORE_COLUMNS[score_col]}-{round(rho * 100)}%.txt")
        df_sorted.head(top_k)["uid"].to_csv(out_file, index=False, header=False)
        print(f"[{score_col}] top {rho:.0%}: {top_k}/{len(df_sorted)} uids -> {out_file}")


if __name__ == "__main__":
    main()
