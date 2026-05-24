#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

class JsonlTextDataset(Dataset):
    """
    dataset is in jsonl format. Each JSON has an id and a text field, but the field names differ by dataset:
    - MIMIC:   uid, text
    - ArXiv:   arxiv_id, abstract
    - EurLex:  celex_id, text
    统一存成 {"uid": ..., "text": ...}
    """
    def __init__(self, jsonl_path, tokenizer, max_length=None, dataset='mimic'):
        self.records = []
        self.dataset = dataset.lower()
        self.tokenizer = tokenizer

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                # ---------- 1.  uid ----------
                if self.dataset == "mimic":
                    uid = obj.get("uid")
                elif self.dataset == "arxiv":
                    uid = obj.get("arxiv_id")
                elif self.dataset == "eurlex":
                    uid = obj.get("celex_id")
                else:
                    raise ValueError(f"Unknown dataset type: {self.dataset}")

                # to string, avoid mixing None/int
                uid = "" if uid is None else str(uid)

                # ---------- 2. take the text for cls----------
                if self.dataset == "arxiv":
                    # ArXiv is "abstract"
                    text = obj.get("abstract", "")
                else:
                    # MIMIC / EurLex is "text"
                    text = obj.get("text", "")

                self.records.append({"uid": uid, "text": text})

        # ---------- 3.  max_length ----------
        if max_length is None:
            # use tokenizer default max_length
            self.max_length = getattr(tokenizer, "model_max_length", 512)
        else:
            self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["text"]
        uid = rec["uid"]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "uid": uid,
        }
        return item


def collate_fn(batch):
    # batch is a list of dicts returned by JsonlTextDataset.__getitem__:
    # {"input_ids": tensor(L), "attention_mask": tensor(L), "uid": str}

    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    uids = [item["uid"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "uids": uids,
    }




@torch.no_grad()
def compute_source_stats_from_source(
    source_jsonl,
    model,
    tokenizer,
    device,
    dataset,
    batch_size=16,
    max_length=512,
):
    """
    on the source dataset (T1), compute:
    1) embedding mean mu
    2) d2 = ||e - mu||^2 的 mean/std
    return: mu, mean_d2, std_d2
    """
    print(f"[FeatureShift] Pass 1: compute mean embedding (mu) from {source_jsonl}")
    ds = JsonlTextDataset(source_jsonl, tokenizer,max_length=max_length, dataset=dataset)  
    loader = DataLoader(ds, batch_size=batch_size,shuffle=False, num_workers=0,collate_fn=collate_fn)


    mu = None
    n_total = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # last layer
        cls_emb = hidden[:, 0, :]          # [CLS] embedding, shape (B, D)

        cls_np = cls_emb.cpu().numpy()
        if mu is None:
            dim = cls_np.shape[1]
            mu = np.zeros((dim,), dtype=np.float64)

        mu += cls_np.sum(axis=0)
        n_total += cls_np.shape[0]

    if n_total == 0:
        raise ValueError(f"No samples found in {source_jsonl}")

    mu /= float(n_total)
    print(f"[FeatureShift] Done Pass 1. #source={n_total}, emb_dim={mu.shape[0]}")

    # ---- Pass 2: compute mean and std of squared distance d2 = ||e - mu||^2 ----
    print(f"[FeatureShift] Pass 2: compute mean/std of squared L2 distance d2")
    ds2 = JsonlTextDataset(source_jsonl, tokenizer, max_length=max_length, dataset=dataset)   # ★ 新增
    loader2 = DataLoader(ds2, batch_size=batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    # Welford on d2
    mean_d2 = 0.0
    m2_d2 = 0.0
    count = 0

    for batch in tqdm(loader2):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        cls_emb = hidden[:, 0, :]

        cls_np = cls_emb.cpu().numpy()
        # 计算每个样本的 d2 = ||e - mu||^2
        diff = cls_np - mu[None, :]
        d2_batch = np.sum(diff * diff, axis=1)  # shape (B,)

        for d2 in d2_batch:
            count += 1
            delta = d2 - mean_d2
            mean_d2 += delta / count
            delta2 = d2 - mean_d2
            m2_d2 += delta * delta2

    if count < 2:
        var_d2 = 0.0
    else:
        var_d2 = m2_d2 / (count - 1)
    std_d2 = math.sqrt(max(var_d2, 1e-12))

    print(f"[FeatureShift] Done Pass 2. mean_d2={mean_d2:.4f}, std_d2={std_d2:.4f}")
    return mu, mean_d2, std_d2


@torch.no_grad()
def compute_feature_shift_for_target(
    target_jsonl,
    model,
    tokenizer,
    device,
    dataset,
    mu,
    mean_d2,
    std_d2,
    out_dir,
    batch_size=16,
    max_length=512,
    tau_f=0.8,
):
    """
    for target dataset  x compute:
      d2(x) = ||e(x) - mu||^2
      z(x)  = (d2(x) - mean_d2) / std_d2
      F(x)  = sigmoid(z(x))
    if F(x) > tau_f, write to uids_feature_shift.txt。
    save(uid, F(x)) to feature_shift_scores.tsv。
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[FeatureShift] Computing F(x) for target {target_jsonl}")
    ds = JsonlTextDataset(target_jsonl, tokenizer, max_length=max_length, dataset=dataset)   # ★ 新增
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    uids_shift = []
    all_scores = []

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        uids = batch["uids"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        cls_emb = hidden[:, 0, :]
        cls_np = cls_emb.cpu().numpy()

        diff = cls_np - mu[None, :]
        d2_batch = np.sum(diff * diff, axis=1)  # squared distances

        for uid, d2 in zip(uids, d2_batch):
            z = (d2 - mean_d2) / (std_d2 + 1e-12)
            F_x = 1.0 / (1.0 + math.exp(-z))  # sigmoid

            all_scores.append((uid, F_x))
            if F_x > tau_f:
                uids_shift.append(uid)

    # write to uids_feature_shift.txt
    uids_path = os.path.join(out_dir, "uids_feature_shift.txt")
    with open(uids_path, "w") as f:
        for uid in uids_shift:
            f.write(str(uid) + "\n")
    print(f"[FeatureShift] Saved {len(uids_shift)} shifted uids to {uids_path}")

    # write to feature_shift_scores.tsv
    scores_path = os.path.join(out_dir, "feature_shift_scores.tsv")
    with open(scores_path, "w") as f:
        f.write("uid\tF_x\n")
        for uid, score in all_scores:
            f.write(f"{uid}\t{score:.6f}\n")
    print(f"[FeatureShift] Saved all scores to {scores_path}")

def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_jsonl', required=True)
    parser.add_argument('--target_jsonl', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--out_dir', required=True)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--tau_f', type=float, default=0.8)
    parser.add_argument('--device', default='cuda')

    
    parser.add_argument(
        '--dataset',
        choices=['mimic', 'eurlex', 'arxiv'],
        required=True,
        help='Which dataset format to use: mimic / eurlex / arxiv'
    )

    # MIMIC (MeSH) 
    parser.add_argument(
        '--mesh_desc',
        type=str,
        default='/project/wliu9/Dataset/MESH/desc2025.xml'
    )
    parser.add_argument(
        '--mesh_supp',
        type=str,
        default='/project/wliu9/Dataset/MESH/supp2025.xml'
    )

    # EurLex (EuroVoc excel)
    parser.add_argument(
        '--eurovoc_xlsx',
        type=str,
        default='/project/wliu9/Dataset/EuroVoc/EuroVoc_Excel_export-4.22/eurovoc_export_en.xlsx'
    )

    # ArXiv-CS (CSO lexicon json)
    parser.add_argument(
        '--cs_lex_json',
        type=str,
        default='/project/wliu9/Dataset/cso-classifier/cs_wikidata_lexicon_from_cso.json'
    )

    # if there's  LLM augmentor, pass the model name here, and the code will automatically call it to augment the text before feeding into the feature shift detector. 
    # This is optional, if not passed then no augmentation will be done and the original text will be used for feature shift detection.
    parser.add_argument('--llm_model', type=str, default=None)

    return parser
def parse_example(example: dict, dataset: str):
    """
    based on dataset, extract text, labels and uid from a json record.
    return: (text, labels, uid)
    """
    if dataset == 'mimic':
        text = example.get('text') or example.get('note') or ''
        labels = example.get('labels', [])
        uid = example.get('uid')
        return text, labels, uid

    elif dataset == 'arxiv':
        text = example.get('abstract', '')
        labels = example.get('labels', [])
        uid = example.get('arxiv_id')
        return text, labels, uid

    elif dataset == 'eurlex':
        text = example.get('text', '')
        labels = example.get('level_1', [])
        uid = example.get('celex_id')
        return text, labels, uid

    else:
        raise ValueError(f"Unknown dataset type: {dataset}")

def main():
    parser = build_argparser()
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA not available, fallback to CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    print(f"[Config] source_jsonl={args.source_jsonl}")
    print(f"[Config] target_jsonl={args.target_jsonl}")
    print(f"[Config] model_path={args.model_path}")
    print(f"[Config] out_dir={args.out_dir}")
    print(f"[Config] device={device}, batch_size={args.batch_size}, max_length={args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    if args.max_length is None:
        max_length = tokenizer.model_max_length
        print(f"[Info] Using tokenizer.model_max_length = {max_length}")
    else:
        max_length = args.max_length
        print(f"[Info] Using user-specified max_length = {max_length}")

    # 1) compute mu, mean_d2, std_d2
    mu, mean_d2, std_d2 = compute_source_stats_from_source(
        source_jsonl=args.source_jsonl,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # 2) compute feature shifts for target data
    compute_feature_shift_for_target(
        target_jsonl=args.target_jsonl,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dataset=args.dataset,
        mu=mu,
        mean_d2=mean_d2,
        std_d2=std_d2,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tau_f=args.tau_f,
    )


if __name__ == "__main__":
    main()
