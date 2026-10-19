"""
KARITA: Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation.

Adapts a source-trained multi-label classifier on a stream of target batches:

    for each target batch (--batch_size_per_iteration samples):
      1. shift detection   uncertainty (online, from the current model)
                           ∪ feature-shift uids ∪ ontology-shift uids (pre-computed, top-rho)
      2. retrieval         top-k most similar source samples for every detected target sample
                           ([CLS] cosine similarity under the frozen source model)
      3. augmentation      knowledge-driven synonym replacement of the retrieved sources
      4. update            fine-tune the current model on the augmented sources (source labels)
    finally predict the whole target split with the adapted model (--prediction_mode 2).

Ablations used in the paper:
    w/o detection    --selection_method none --random_target_total <10% of target>
    w/o retrieval    --retrieval least_similar
    w/o augmentation --augmentation_option option0
    single detector  keep only one of: --selection_method both / --feature_shift_uids / --ontology_shift_uids
"""

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.syno_aug_pipline import (
    CSOWikidataSynonymAugmentor,
    LexiconSynonymAugmentor,
    LexiconSynonymExtractor,
    MeSHSynonymAugmentor,
    MeSHSynonymExtractor,
)

DATASET_CONFIGS = {
    "mimic": dict(text_field="text", label_field="label", id_field="uid",
                  max_length=4096, lexicon_type="mesh"),
    "eurlex": dict(text_field="text", label_field="level_1", id_field="celex_id",
                   max_length=512, lexicon_type="lexicon_jsonl"),
    "arxiv": dict(text_field="abstract", label_field="labels", id_field="arxiv_id",
                  max_length=512, lexicon_type="cso_wikidata"),
}


# =====================================================================
# Data
# =====================================================================

def load_jsonl(fp):
    with open(fp, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(data, fp):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_id_set(path):
    if not path:
        return set()
    with open(path, encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    print(f"✓ Loaded {len(ids)} ids from {path}")
    return ids


class MultiLabelDataset(Dataset):
    def __init__(self, samples, tokenizer, label2id, cfg, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.cfg = cfg
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(s.get(self.cfg["text_field"]) or "", padding="max_length",
                             truncation=True, max_length=self.max_length, return_tensors="pt")

        labels = [0] * self.num_labels
        label_raw = s.get(self.cfg["label_field"]) or []
        if label_raw and isinstance(label_raw[0], str):
            for code in label_raw:
                li = self.label2id.get(code)
                if li is not None:
                    labels[li] = 1
        elif label_raw:
            # already a multi-hot vector
            labels[: len(label_raw)] = label_raw[: self.num_labels]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(labels, dtype=torch.float),
            "uid": str(s.get(self.cfg["id_field"])),
        }


def make_loader(samples, tokenizer, label2id, cfg, args, batch_size, shuffle=False):
    dataset = MultiLabelDataset(samples, tokenizer, label2id, cfg, args.max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True)


# =====================================================================
# Uncertainty-based detection & prediction
# =====================================================================

def binary_entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels, uids = [], [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).logits
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["label"].numpy())
        uids.extend(batch["uid"])
    return np.vstack(probs), np.vstack(labels), uids


def select_uncertain(model, loader, device, method, threshold_conf, threshold_entropy):
    """
    Uncertainty trigger on the current model.
      entropy : mean binary entropy > threshold_entropy
      max_prob: max label probability < threshold_conf
      both    : entropy OR max_prob  (the setting used for the paper results)
    """
    probs, _, uids = predict_probs(model, loader, device)
    entropy = binary_entropy(probs).mean(axis=1)
    max_prob = probs.max(axis=1)
    if method == "entropy":
        mask = entropy > threshold_entropy
    elif method == "max_prob":
        mask = max_prob < threshold_conf
    elif method == "both":
        mask = (entropy > threshold_entropy) | (max_prob < threshold_conf)
    else:
        raise ValueError(f"Unknown method: {method}")
    return [u for u, m in zip(uids, mask) if m]


def prediction_details(probs, labels, uids, label_list, threshold):
    preds = (probs >= threshold).astype(int)
    max_prob = probs.max(axis=1)
    avg_entropy = binary_entropy(probs).mean(axis=1)
    details = []
    for i, uid in enumerate(uids):
        pred_set = {label_list[j] for j in np.flatnonzero(preds[i])}
        true_set = {label_list[j] for j in np.flatnonzero(labels[i])}
        details.append({
            "uid": uid,
            "max_prob": float(max_prob[i]),
            "avg_entropy": float(avg_entropy[i]),
            "predicted_labels": sorted(pred_set),
            "true_labels": sorted(true_set),
            "is_fully_correct": pred_set == true_set,
        })
    return preds, details


def compute_metrics_multilabel(predictions, labels):
    y_pred = np.asarray(predictions).astype(int)
    y_true = np.asarray(labels).astype(int)
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg, name in [("samples", "sample"), ("micro", "micro"), ("macro", "macro"), ("weighted", "weighted")]:
        metrics[f"{name}_precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        metrics[f"{name}_recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        metrics[f"{name}_f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    return metrics


def save_predictions(details, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pred_details.jsonl", "w", encoding="utf-8") as f:
        for item in details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(details)} predictions to {output_dir / 'pred_details.jsonl'}")


# =====================================================================
# Source backtracking retrieval
# =====================================================================

class RetrievalEngine:
    """[CLS] embeddings of the frozen source model; source embeddings are cached."""

    def __init__(self, source_data, model_path, device, cfg, args, label2id, cache_dir):
        self.source_data = source_data
        self.device = device
        self.cfg = cfg
        self.args = args
        self.label2id = label2id
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source_embeddings = None

    def _cache_path(self):
        # same fingerprint as the original experiment code, so existing caches can be reused
        h = hashlib.md5()
        h.update(self.model_path.encode())
        h.update(str(len(self.source_data)).encode())
        return self.cache_dir / f"emb_{h.hexdigest()}.npz"

    @torch.no_grad()
    def _encode(self, samples, desc):
        loader = make_loader(samples, self.tokenizer, self.label2id, self.cfg, self.args,
                             batch_size=self.args.batch_size_test)
        embs = []
        for batch in tqdm(loader, desc=desc, disable=len(samples) < 1000):
            out = self.model(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device),
                             output_hidden_states=True)
            embs.append(out.hidden_states[-1][:, 0, :].cpu().numpy())
        return np.vstack(embs)

    def precompute(self):
        cache_path = self._cache_path()
        if cache_path.exists():
            self.source_embeddings = np.load(cache_path, allow_pickle=True)["embeddings"]
            print(f"✓ Loaded cached source embeddings from {cache_path} (shape={self.source_embeddings.shape})")
            return
        print(f"Encoding {len(self.source_data)} source samples ...")
        self.source_embeddings = self._encode(self.source_data, "Encoding source")
        np.savez_compressed(cache_path, embeddings=self.source_embeddings,
                            meta=np.array({"model_path": self.model_path,
                                           "count": len(self.source_data)}, dtype=object))
        print(f"✓ Saved source embeddings to {cache_path}")

    def retrieve(self, target_samples, k, least_similar=False):
        """
        top-k most similar sources per target sample
        (least_similar=True: the k least similar ones, i.e. the w/o-retrieval ablation)
        """
        sims = cosine_similarity(self._encode(target_samples, "Encoding targets"), self.source_embeddings)
        order = np.argsort(sims, axis=1)
        idx = order[:, :k] if least_similar else order[:, -k:][:, ::-1]

        id_field = self.cfg["id_field"]
        retrieved, records = {}, []
        for i, sample in enumerate(target_samples):
            uid = str(sample.get(id_field))
            retrieved[uid] = [self.source_data[j] for j in idx[i]]
            records.append({
                "target_uid": uid,
                "retrieved_uids": [str(self.source_data[j].get(id_field)) for j in idx[i]],
                "similarities": sims[i, idx[i]].tolist(),
            })
        return retrieved, records


# =====================================================================
# Knowledge-driven augmentation
# =====================================================================

def build_augmentor(args):
    if args.lexicon_type == "mesh":
        return MeSHSynonymAugmentor(MeSHSynonymExtractor(args.mesh_desc, args.mesh_supp))
    if args.lexicon_type == "lexicon_jsonl":
        return LexiconSynonymAugmentor(
            LexiconSynonymExtractor(args.lexicon_path, mode="gpt_jsonl", fast_match=args.fast_match))
    if args.lexicon_type == "eurovoc_xlsx":
        return LexiconSynonymAugmentor(
            LexiconSynonymExtractor(args.lexicon_path, mode="eurovoc_xlsx", fast_match=args.fast_match))
    if args.lexicon_type == "cso_wikidata":
        return CSOWikidataSynonymAugmentor(args.lexicon_path)
    raise ValueError(f"Unknown lexicon_type: {args.lexicon_type}")


def augment_sources(augmentor, sources, option, cfg, max_replacements):
    """option0: retrieved sources as they are; option1: synonym replacement."""
    if option == "option0":
        return list(sources)
    return [augmentor.augment_single(src, max_replacements=max_replacements, text_field=cfg["text_field"])
            for src in sources]


# =====================================================================
# Model update
# =====================================================================

def train_on_augmented(model, aug_data, tokenizer, label2id, cfg, args):
    """
    Fine-tune on the augmented sources. A fresh AdamW is created for every target batch.
    The original code used mini-batches of max(2, batch_size_train // 2) and divided the
    loss by 2 while still stepping on every mini-batch; this is kept unchanged so that
    the paper results can be reproduced.
    """
    loader = make_loader(aug_data, tokenizer, label2id, cfg, args,
                         batch_size=max(2, args.batch_size_train // 2))
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    start = time.time()
    epoch_losses = []
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device)).logits
            loss = criterion(logits, batch["label"].to(args.device)) / 2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss / len(loader))
        print(f"  Epoch {epoch + 1}/{args.num_epochs} loss: {epoch_losses[-1]:.4f}")
    return epoch_losses, time.time() - start


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="KARITA temporal adaptation")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS), required=True)
    p.add_argument("--source_file", required=True, help="Labeled source pool for retrieval, e.g. T123_train.json")
    p.add_argument("--target_file", required=True, help="Target stream, e.g. T4_test.json")
    p.add_argument("--model_path", required=True, help="Source-trained checkpoint (classification.py output)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=None, help="Default: 4096 for mimic, 512 otherwise")
    p.add_argument("--num_workers", type=int, default=4)

    # stream / training
    p.add_argument("--batch_size_per_iteration", type=int, default=100, help="Target samples per adaptation step")
    p.add_argument("--batch_size_train", type=int, default=4)
    p.add_argument("--batch_size_test", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=3, help="Epochs over the augmented data of each step")
    p.add_argument("--learning_rate", type=float, default=3e-5)

    # 1. shift detection
    p.add_argument("--selection_method", choices=["none", "entropy", "max_prob", "both"], default="both",
                   help="Uncertainty trigger; 'none' disables it")
    p.add_argument("--threshold_conf", type=float, default=0.5, help="tau_p on the max label probability")
    p.add_argument("--threshold_entropy", type=float, default=0.25, help="tau_H on the mean binary entropy")
    p.add_argument("--feature_shift_uids", default=None, help="uid_feature_shift-10%%.txt (select-top-shift.py)")
    p.add_argument("--ontology_shift_uids", default=None, help="uid_ontology_shift-tail-10%%.txt")
    p.add_argument("--random_target_total", type=int, default=0,
                   help=">0: additionally select this many random target samples (w/o-detection ablation)")

    # 2. retrieval
    p.add_argument("--k_retrieve", type=int, default=3)
    p.add_argument("--retrieval", choices=["topk", "least_similar"], default="topk",
                   help="least_similar = w/o-retrieval ablation")
    p.add_argument("--emb_cache_dir", default=None, help="Default: <model_path>/emb_cache")

    # 3. augmentation
    p.add_argument("--augmentation_option", choices=["option0", "option1"], default="option1",
                   help="option0: no augmentation; option1: synonym replacement")
    p.add_argument("--lexicon_type", choices=["mesh", "lexicon_jsonl", "eurovoc_xlsx", "cso_wikidata"], default=None,
                   help="Default: mesh (mimic), lexicon_jsonl (eurlex), cso_wikidata (arxiv)")
    p.add_argument("--mesh_desc", default=None, help="MeSH desc2025.xml")
    p.add_argument("--mesh_supp", default=None, help="MeSH supp2025.xml")
    p.add_argument("--lexicon_path", default=None, help="Lexicon file for lexicon_jsonl / eurovoc_xlsx / cso_wikidata")
    p.add_argument("--max_replacements", type=int, default=3)
    p.add_argument("--fast_match", action="store_true",
                   help="Regex term matching for lexicon_jsonl/eurovoc_xlsx (faster, not paper-exact)")

    # prediction
    p.add_argument("--prediction_mode", type=int, choices=[1, 2], default=2,
                   help="1: predict each target batch right after its update; 2: predict all target data at the end")
    p.add_argument("--prediction_threshold", type=float, default=0.5)
    p.add_argument("--save_model", action="store_true", help="Save the adapted model to <output_dir>/adapted_model")

    args = p.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    args.max_length = args.max_length or cfg["max_length"]
    args.lexicon_type = args.lexicon_type or cfg["lexicon_type"]
    if args.augmentation_option == "option1":
        if args.lexicon_type == "mesh" and not args.mesh_desc:
            p.error("--mesh_desc is required for lexicon_type=mesh")
        if args.lexicon_type != "mesh" and not args.lexicon_path:
            p.error(f"--lexicon_path is required for lexicon_type={args.lexicon_type}")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_times = dict(detection=0.0, retrieval=0.0, augmentation=0.0, training=0.0, prediction=0.0)
    global_start = time.time()

    # ---------- data & models ----------
    source = load_jsonl(args.source_file)
    target = load_jsonl(args.target_file)
    num_batches = (len(target) + args.batch_size_per_iteration - 1) // args.batch_size_per_iteration
    print(f"✓ {len(source)} source samples, {len(target)} target samples, {num_batches} target batches")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(args.device)
    label2id = model.config.label2id
    label_list = sorted(label2id, key=label2id.get)
    print(f"✓ {len(label_list)} labels: {label_list[:5]} ...")

    retrieval = RetrievalEngine(source, args.model_path, args.device, cfg, args, label2id,
                                cache_dir=args.emb_cache_dir or Path(args.model_path) / "emb_cache")
    retrieval.precompute()
    augmentor = build_augmentor(args) if args.augmentation_option == "option1" else None

    feature_shift_ids = load_id_set(args.feature_shift_uids)
    ontology_shift_ids = load_id_set(args.ontology_shift_uids)
    random_target_ids = set()
    if args.random_target_total > 0:
        all_uids = [str(s.get(cfg["id_field"])) for s in target]
        random_target_ids = set(random.sample(all_uids, min(args.random_target_total, len(all_uids))))
        print(f"✓ Randomly selected {len(random_target_ids)} target samples")

    retrieval_records, training_history = [], []
    stream_probs, stream_labels, stream_uids = [], [], []

    # ---------- adaptation over the target stream ----------
    for b in range(num_batches):
        batch = target[b * args.batch_size_per_iteration:(b + 1) * args.batch_size_per_iteration]
        batch_loader = make_loader(batch, tokenizer, label2id, cfg, args, args.batch_size_test)
        batch_uids = [str(s.get(cfg["id_field"])) for s in batch]
        print(f"\n{'=' * 80}\nBatch {b + 1}/{num_batches} ({len(batch)} target samples)\n{'=' * 80}")

        # 1. shift detection
        t0 = time.time()
        selected = set()
        sources_count = {}
        if args.selection_method != "none":
            unc = select_uncertain(model, batch_loader, args.device, args.selection_method,
                                   args.threshold_conf, args.threshold_entropy)
            selected.update(unc)
            sources_count["uncertainty"] = len(unc)
        for name, ids in [("feature", feature_shift_ids), ("ontology", ontology_shift_ids),
                          ("random", random_target_ids)]:
            if ids:
                hit = [u for u in batch_uids if u in ids]
                selected.update(hit)
                sources_count[name] = len(hit)
        detected = [s for s, u in zip(batch, batch_uids) if u in selected]
        phase_times["detection"] += time.time() - t0
        print(f"[1] detected {len(detected)} shifted samples {sources_count}")

        if detected:
            # 2. retrieval
            t0 = time.time()
            retrieved, records = retrieval.retrieve(detected, args.k_retrieve,
                                                    least_similar=args.retrieval == "least_similar")
            retrieval_records.append({"batch_idx": b + 1, "records": records})
            phase_times["retrieval"] += time.time() - t0

            # 3. augmentation
            t0 = time.time()
            aug_data = []
            for srcs in retrieved.values():
                aug_data.extend(augment_sources(augmentor, srcs, args.augmentation_option, cfg,
                                                args.max_replacements))
            phase_times["augmentation"] += time.time() - t0
            print(f"[2-3] retrieved {args.k_retrieve} sources per sample -> {len(aug_data)} training samples")

            # 4. update
            losses, train_time = train_on_augmented(model, aug_data, tokenizer, label2id, cfg, args)
            phase_times["training"] += train_time
            training_history.append({"batch_idx": b + 1, "num_detected": len(detected),
                                     "detected_by": sources_count, "num_train_samples": len(aug_data),
                                     "epoch_losses": losses})
        else:
            print("No shifted samples in this batch, skip the update.")

        if args.prediction_mode == 1:
            t0 = time.time()
            probs, labels, uids = predict_probs(model, batch_loader, args.device)
            stream_probs.append(probs)
            stream_labels.append(labels)
            stream_uids.extend(uids)
            phase_times["prediction"] += time.time() - t0

    # ---------- prediction ----------
    t0 = time.time()
    if args.prediction_mode == 2:
        full_loader = make_loader(target, tokenizer, label2id, cfg, args, args.batch_size_test)
        probs, labels, uids = predict_probs(model, full_loader, args.device)
    else:
        probs, labels, uids = np.vstack(stream_probs), np.vstack(stream_labels), stream_uids
    preds, details = prediction_details(probs, labels, uids, label_list, args.prediction_threshold)
    metrics = compute_metrics_multilabel(preds, labels)
    phase_times["prediction"] += time.time() - t0

    # ---------- save ----------
    save_predictions(details, output_dir / "final_predictions")
    results_dir = output_dir / "results"
    save_json(metrics, results_dir / "metrics_summary.json")
    save_json(retrieval_records, results_dir / "retrieval_records.json")
    save_json(training_history, results_dir / "training_history.json")
    save_json({
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "num_target_batches": num_batches,
        "num_target_samples": len(target),
        "num_detected_samples": sum(h["num_detected"] for h in training_history),
        "num_train_samples": sum(h["num_train_samples"] for h in training_history),
        "phase_times_s": {k: round(v, 2) for k, v in phase_times.items()},
        "total_runtime_s": round(time.time() - global_start, 2),
        "final_metrics": metrics,
    }, results_dir / "summary.json")
    if args.save_model:
        model.save_pretrained(output_dir / "adapted_model")
        tokenizer.save_pretrained(output_dir / "adapted_model")

    print(f"\n{'=' * 80}\nFinal metrics on {args.target_file}")
    for k in ["sample_precision", "sample_recall", "sample_f1", "micro_f1", "macro_f1"]:
        print(f"  {k:17s} {metrics[k]:.4f}")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
