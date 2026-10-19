#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ontology-based terminology shift detection (O_tail).

1) Load an ontology thesaurus as  surface form -> set(concept_id)
     MIMIC : MeSH (desc + supp XML)
     EurLex: EuroVoc (xlsx export)
     arXiv : CSO lexicon (cs_wikidata_lexicon_from_cso.json)
2) Extract the concepts C(x) of every source / target document (substring match).
3) p_src(c) = document frequency of concept c in the source split / #source docs.
4) For each target document:
       O_tail(x) = mean_{c in C(x), p_src(c) > 0}  -log(p_src(c) + eps)
   A larger O_tail means the document relies on long-tail terminology
   under the source concept distribution.

Notes on the implementation (kept as-is so the released scores match the paper runs):
  * only concepts with f_src(c) + f_tgt(c) >= --min_total are counted;
  * concepts unseen in the source split (p_src = 0) are skipped in the mean;
  * at most 128 concepts are extracted per document.

Output (in --out_dir):
    ontology4_tail_shift_scores.tsv   uid \t O_tail

The top-rho fraction used by KARITA is selected afterwards with
shift_detection/select-top-shift.py.
"""

import os
import json
import math
import argparse
from collections import Counter, defaultdict

from tqdm import tqdm
import xml.etree.ElementTree as ET

import pandas as pd

# (id field, text field) of each dataset
DATASET_FIELDS = {
    "mimic": ("uid", "text"),
    "eurlex": ("celex_id", "text"),
    "arxiv": ("arxiv_id", "abstract"),
}


def iter_jsonl_for_dataset(jsonl_path: str, dataset: str):
    id_field, text_field = DATASET_FIELDS[dataset]
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield str(obj.get(id_field)), obj.get(text_field, "")


# =========================
# 1. Ontology loaders: surface form -> set(concept_id)
# =========================

def load_mesh_lexicon(desc_path, supp_path=None):
    term2concepts = defaultdict(set)

    # ---- DescriptorRecordSet ----
    if desc_path and os.path.exists(desc_path):
        print(f"[Ontology] Loading MeSH descriptors from {desc_path}")
        root = ET.parse(desc_path).getroot()
        for dr in tqdm(root.findall("./DescriptorRecord"), desc="MeSH-DescriptorRecord"):
            cui = dr.findtext("./DescriptorUI")
            if not cui:
                continue
            pref = dr.findtext("./DescriptorName/String")
            if pref:
                term2concepts[pref.lower().strip()].add(cui)
            # all terms under ConceptList/TermList/Term/String
            for term_node in dr.findall(".//TermList/Term/String"):
                if term_node.text:
                    term2concepts[term_node.text.lower().strip()].add(cui)
    else:
        print(f"[Ontology] WARNING: desc_path not found: {desc_path}")

    # ---- SupplementalRecordSet ----
    if supp_path and os.path.exists(supp_path):
        print(f"[Ontology] Loading MeSH supplementals from {supp_path}")
        root = ET.parse(supp_path).getroot()
        for sr in tqdm(root.findall("./SupplementalRecord"), desc="MeSH-SupplementalRecord"):
            cui = sr.findtext("./SupplementalRecordUI")
            if not cui:
                continue
            pref = sr.findtext("./SupplementalRecordName/String")
            if pref:
                term2concepts[pref.lower().strip()].add(cui)
            for term_node in sr.findall(".//TermList/Term/String"):
                if term_node.text:
                    term2concepts[term_node.text.lower().strip()].add(cui)
    elif supp_path:
        print(f"[Ontology] WARNING: supp_path not found: {supp_path}")

    print(f"[Ontology] MeSH term2concepts size = {len(term2concepts)} surface forms")
    return term2concepts


def load_eurovoc_lexicon(xlsx_path: str):
    """EuroVoc Excel export; uses the columns ID, TERMS (PT-NPT) and PT."""
    term2concepts = defaultdict(set)
    if not os.path.exists(xlsx_path):
        print(f"[Ontology] WARNING: EuroVoc xlsx not found: {xlsx_path}")
        return term2concepts

    print(f"[Ontology] Loading EuroVoc from {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    col_id = col_terms = col_pt = None
    for c in df.columns:
        cname = str(c).strip()
        if cname == "ID":
            col_id = c
        elif "TERMS" in cname and "PT-NPT" in cname:
            col_terms = c
        elif cname == "PT":
            col_pt = c
    if col_id is None or col_terms is None or col_pt is None:
        print("[Ontology] WARNING: EuroVoc columns not found as expected; check header.")
        print("Columns:", df.columns)
        return term2concepts

    for _, row in tqdm(df.iterrows(), total=len(df), desc="EuroVoc-rows"):
        cid = row[col_id]
        if pd.isna(cid):
            continue
        cid = str(cid).strip()
        for form in (row[col_terms], row[col_pt]):
            if isinstance(form, str):
                s = form.lower().strip()
                if s:
                    term2concepts[s].add(cid)

    print(f"[Ontology] EuroVoc term2concepts size = {len(term2concepts)} surface forms")
    return term2concepts


def load_cs_lexicon(json_path: str):
    """
    cs_wikidata_lexicon_from_cso.json:
    {
      "0-1 knapsack problem": {
        "wikidata_id": "Q105415774",
        "forms": ["0-1 knapsack problem", "binary knapsack problem"],
        ...
      },
      ...
    }
    concept id = wikidata_id if present, otherwise the canonical key.
    """
    term2concepts = defaultdict(set)
    if not os.path.exists(json_path):
        print(f"[Ontology] WARNING: CS lexicon json not found: {json_path}")
        return term2concepts

    print(f"[Ontology] Loading CS lexicon from {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    for canonical, info in tqdm(data.items(), desc="CS-lex-entries"):
        cid = info.get("wikidata_id") or canonical
        forms = [canonical]
        extra_forms = info.get("forms", [])
        if isinstance(extra_forms, list):
            forms.extend(extra_forms)
        for form in forms:
            if isinstance(form, str):
                s = form.lower().strip()
                if s:
                    term2concepts[s].add(cid)

    print(f"[Ontology] CS term2concepts size = {len(term2concepts)} surface forms")
    return term2concepts


# =========================
# 2. Concept extraction & frequencies
# =========================

def extract_concepts_from_text(text, term2concepts, max_concepts_per_doc=128):
    """Plain substring matching against every surface form (in lexicon order)."""
    text_l = text.lower()
    concepts = set()
    for term, cuis in term2concepts.items():
        if term and term in text_l:
            concepts.update(cuis)
            if len(concepts) >= max_concepts_per_doc:
                break
    return concepts


def compute_concept_frequencies(source_jsonl, target_jsonl, dataset, term2concepts):
    """Document frequencies of every concept in the source and target splits."""
    freq_source, freq_target = Counter(), Counter()
    target_samples = []
    n_docs_source = n_docs_target = 0

    print(f"[Ontology] Scanning source: {source_jsonl}")
    for _, text in tqdm(iter_jsonl_for_dataset(source_jsonl, dataset), desc="source"):
        n_docs_source += 1
        freq_source.update(extract_concepts_from_text(text, term2concepts))

    print(f"[Ontology] Scanning target: {target_jsonl}")
    for uid, text in tqdm(iter_jsonl_for_dataset(target_jsonl, dataset), desc="target"):
        n_docs_target += 1
        concepts = extract_concepts_from_text(text, term2concepts)
        freq_target.update(concepts)
        target_samples.append((uid, concepts))

    print(f"[Ontology] #source concepts (non-unique) = {sum(freq_source.values())}, unique={len(freq_source)}")
    print(f"[Ontology] #target concepts (non-unique) = {sum(freq_target.values())}, unique={len(freq_target)}")
    print(f"[Ontology] #docs source = {n_docs_source}, #docs target = {n_docs_target}")
    if n_docs_source <= 0 or n_docs_target <= 0:
        raise ValueError(f"Invalid doc counts: source={n_docs_source}, target={n_docs_target}")
    return freq_source, freq_target, target_samples, n_docs_source


def source_concept_probs(freq_source, freq_target, n_docs_source, min_total=5):
    """p_src(c) = f_src(c) / N_src for concepts with f_src(c) + f_tgt(c) >= min_total."""
    p_src = {}
    for c in set(freq_source) | set(freq_target):
        f_src = freq_source.get(c, 0)
        if f_src + freq_target.get(c, 0) < min_total:
            continue
        p_src[c] = f_src / float(n_docs_source)
    return p_src


# =========================
# 3. Document-level O_tail
# =========================

def compute_ontology_tail_scores(target_samples, p_src, out_dir, eps=1e-12):
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    print(f"[Ontology] Computing O_tail for {len(target_samples)} target samples")
    for uid, concepts in tqdm(target_samples, desc="O_tail"):
        vals = [-math.log(p_src[c] + eps) for c in concepts if p_src.get(c, 0.0) > 0.0]
        rows.append((uid, sum(vals) / len(vals) if vals else 0.0))

    tail_path = os.path.join(out_dir, "ontology4_tail_shift_scores.tsv")
    with open(tail_path, "w") as f:
        f.write("uid\tO_tail\n")
        for uid, score in rows:
            f.write(f"{uid}\t{score:.6f}\n")
    print(f"[Ontology] Saved O_tail scores to {tail_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Ontology-based terminology shift detection (O_tail)")
    parser.add_argument("--dataset", choices=list(DATASET_FIELDS), required=True)
    parser.add_argument("--source_jsonl", required=True, help="Source split, e.g. T1_train.json")
    parser.add_argument("--target_jsonl", required=True, help="Target split, e.g. T4_test.json")
    parser.add_argument("--out_dir", required=True, help="e.g. $MODEL_PATH/shift_detector/ontology")
    # MIMIC
    parser.add_argument("--mesh_desc", default="", help="MeSH desc2025.xml")
    parser.add_argument("--mesh_supp", default="", help="MeSH supp2025.xml")
    # EurLex
    parser.add_argument("--eurovoc_xlsx", default="", help="EuroVoc export, e.g. eurovoc_export_en.xlsx (v4.22)")
    # arXiv
    parser.add_argument("--cs_lexicon_json", default="", help="cs_wikidata_lexicon_from_cso.json")
    parser.add_argument("--min_total", type=int, default=5,
                        help="Minimum f_src + f_tgt for a concept to be counted")
    args = parser.parse_args()

    print(f"[Config] {vars(args)}")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "mimic":
        term2concepts = load_mesh_lexicon(args.mesh_desc, args.mesh_supp)
    elif args.dataset == "eurlex":
        term2concepts = load_eurovoc_lexicon(args.eurovoc_xlsx)
    else:
        term2concepts = load_cs_lexicon(args.cs_lexicon_json)

    freq_source, freq_target, target_samples, n_docs_source = compute_concept_frequencies(
        args.source_jsonl, args.target_jsonl, args.dataset, term2concepts)
    p_src = source_concept_probs(freq_source, freq_target, n_docs_source, min_total=args.min_total)
    compute_ontology_tail_scores(target_samples, p_src, args.out_dir)


if __name__ == "__main__":
    main()
