#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature-based shift detection.

For every target sample x, compute a feature-shift score F(x) from the [CLS]
embedding of the source-trained model:

    mu       = mean [CLS] embedding over the source set
    d2(x)    = ||e(x) - mu||^2
    z(x)     = (d2(x) - mean_d2) / std_d2      (mean/std of d2 over the source set)
    F(x)     = sigmoid(z(x))

Outputs (in --out_dir):
    feature_shift_scores.tsv   uid \t F_x   (all target samples)
    uids_feature_shift.txt     uids with F(x) > --tau_f

The top-rho fraction used by KARITA is selected afterwards with
shift_detection/select-top-shift.py.
"""

import os
import json
import math
import argparse

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# (id field, text field) of each dataset
DATASET_FIELDS = {
    "mimic": ("uid", "text"),
    "eurlex": ("celex_id", "text"),
    "arxiv": ("arxiv_id", "abstract"),
}


class JsonlTextDataset(Dataset):
    """Reads a jsonl file and keeps {"uid": str, "text": str} per line."""

    def __init__(self, jsonl_path, tokenizer, max_length=None, dataset="mimic"):
        if dataset not in DATASET_FIELDS:
            raise ValueError(f"Unknown dataset type: {dataset}")
        id_field, text_field = DATASET_FIELDS[dataset]

        self.records = []
        self.tokenizer = tokenizer
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                uid = obj.get(id_field)
                # to string, avoid mixing None/int
                uid = "" if uid is None else str(uid)
                self.records.append({"uid": uid, "text": obj.get(text_field, "")})

        # None -> use the tokenizer default (4096 for Longformer, 512 for XLM-R)
        if max_length is None:
            self.max_length = getattr(tokenizer, "model_max_length", 512)
        else:
            self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "uid": rec["uid"],
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
        "uids": [item["uid"] for item in batch],
    }


def cls_embeddings(model, batch, device):
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        output_hidden_states=True,
    )
    # [CLS] embedding of the last layer, shape (B, D)
    return outputs.hidden_states[-1][:, 0, :].cpu().numpy()


@torch.no_grad()
def compute_source_stats(source_jsonl, model, tokenizer, device, dataset,
                         batch_size=16, max_length=None):
    """
    On the source set compute:
      1) the mean embedding mu
      2) mean / std of d2 = ||e - mu||^2  (Welford, only statistics are kept)
    Returns: mu, mean_d2, std_d2
    """
    ds = JsonlTextDataset(source_jsonl, tokenizer, max_length=max_length, dataset=dataset)

    # ---- Pass 1: mean embedding ----
    print(f"[FeatureShift] Pass 1: compute mean embedding (mu) from {source_jsonl}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    mu = None
    n_total = 0
    for batch in tqdm(loader):
        cls_np = cls_embeddings(model, batch, device)
        if mu is None:
            mu = np.zeros((cls_np.shape[1],), dtype=np.float64)
        mu += cls_np.sum(axis=0)
        n_total += cls_np.shape[0]

    if n_total == 0:
        raise ValueError(f"No samples found in {source_jsonl}")
    mu /= float(n_total)
    print(f"[FeatureShift] Done Pass 1. #source={n_total}, emb_dim={mu.shape[0]}")

    # ---- Pass 2: mean / std of the squared distance d2 ----
    print("[FeatureShift] Pass 2: compute mean/std of squared L2 distance d2")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    mean_d2 = 0.0
    m2_d2 = 0.0
    count = 0
    for batch in tqdm(loader):
        cls_np = cls_embeddings(model, batch, device)
        diff = cls_np - mu[None, :]
        for d2 in np.sum(diff * diff, axis=1):
            count += 1
            delta = d2 - mean_d2
            mean_d2 += delta / count
            m2_d2 += delta * (d2 - mean_d2)

    var_d2 = m2_d2 / (count - 1) if count >= 2 else 0.0
    std_d2 = math.sqrt(max(var_d2, 1e-12))
    print(f"[FeatureShift] Done Pass 2. mean_d2={mean_d2:.4f}, std_d2={std_d2:.4f}")
    return mu, mean_d2, std_d2


@torch.no_grad()
def compute_feature_shift_for_target(target_jsonl, model, tokenizer, device, dataset,
                                     mu, mean_d2, std_d2, out_dir,
                                     batch_size=16, max_length=None, tau_f=0.8):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[FeatureShift] Computing F(x) for target {target_jsonl}")
    ds = JsonlTextDataset(target_jsonl, tokenizer, max_length=max_length, dataset=dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    uids_shift = []
    all_scores = []
    for batch in tqdm(loader):
        cls_np = cls_embeddings(model, batch, device)
        diff = cls_np - mu[None, :]
        for uid, d2 in zip(batch["uids"], np.sum(diff * diff, axis=1)):
            z = (d2 - mean_d2) / (std_d2 + 1e-12)
            F_x = 1.0 / (1.0 + math.exp(-z))
            all_scores.append((uid, F_x))
            if F_x > tau_f:
                uids_shift.append(uid)

    uids_path = os.path.join(out_dir, "uids_feature_shift.txt")
    with open(uids_path, "w") as f:
        for uid in uids_shift:
            f.write(str(uid) + "\n")
    print(f"[FeatureShift] Saved {len(uids_shift)} uids with F(x) > {tau_f} to {uids_path}")

    scores_path = os.path.join(out_dir, "feature_shift_scores.tsv")
    with open(scores_path, "w") as f:
        f.write("uid\tF_x\n")
        for uid, score in all_scores:
            f.write(f"{uid}\t{score:.6f}\n")
    print(f"[FeatureShift] Saved all scores to {scores_path}")


def main():
    parser = argparse.ArgumentParser(description="Feature-based shift detection")
    parser.add_argument("--dataset", choices=list(DATASET_FIELDS), required=True)
    parser.add_argument("--source_jsonl", required=True, help="Source split, e.g. T1_train.json")
    parser.add_argument("--target_jsonl", required=True, help="Target split, e.g. T4_test.json")
    parser.add_argument("--model_path", required=True, help="Source-trained checkpoint")
    parser.add_argument("--out_dir", required=True, help="e.g. $MODEL_PATH/shift_detector/feature")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=None,
                        help="Default: tokenizer.model_max_length")
    parser.add_argument("--tau_f", type=float, default=0.8,
                        help="Threshold for uids_feature_shift.txt (not used by KARITA, which takes the top-rho)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA not available, fallback to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"[Config] {vars(args)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    mu, mean_d2, std_d2 = compute_source_stats(
        args.source_jsonl, model, tokenizer, device, args.dataset,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    compute_feature_shift_for_target(
        args.target_jsonl, model, tokenizer, device, args.dataset,
        mu, mean_d2, std_d2, args.out_dir,
        batch_size=args.batch_size, max_length=args.max_length, tau_f=args.tau_f,
    )


if __name__ == "__main__":
    main()
