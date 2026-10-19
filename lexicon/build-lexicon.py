#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge the GPT term expansions with an external ontology thesaurus into the
term lexicon used for augmentation (karita.py --lexicon_type lexicon_jsonl).

  1) GPT terms:  term -> {synonyms}           (output of gpt-term-expansion.py)
  2) enrichment: if a GPT term is a form of an ontology concept, all other forms
                 of that concept are added to its synonyms
       eurovoc: EuroVoc Excel export (PT / NPT of the same concept)
       cso    : cso_synonyms_clean.json from the CSO classifier assets
  3) symmetric closure: A in syn(B)  =>  B in syn(A)

Output jsonl, one line per term: {"term": "...", "synonyms": ["...", ...]}

Example:
  python lexicon/build-lexicon.py --ontology eurovoc \
      --gpt_terms lexicon/eurlex/T4_test.with_terms.json \
      --ontology_path EuroVoc_Excel_export-4.22/eurovoc_export_en.xlsx \
      --output lexicon/eurlex/eurlex_gpt_terms_lexicon_with_eurovoc.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set


def _normalize_term(s: str) -> str:
    """Key used for matching: lower-case, unify dashes, drop spaces and hyphens."""
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[-\s]+", "", s)
    return s.lower()


def _variants_for(term: str) -> Set[str]:
    raw = term.strip()
    t = raw.replace("–", "-").replace("—", "-")
    no_space = re.sub(r"\s+", "", t)
    hyphened = re.sub(r"\s+", "-", t)
    return {raw, t, no_space, hyphened}


def load_gpt_terms(gpt_jsonl: str) -> Dict[str, Set[str]]:
    term_to_syns: Dict[str, Set[str]] = defaultdict(set)
    with open(gpt_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ge = json.loads(line).get("gpt_term_expansion")
            if not ge:
                continue
            for ent in ge.get("entities") or []:
                term = (ent.get("term") or "").strip()
                if not term:
                    continue
                for s in ent.get("synonyms") or []:
                    s = (s or "").strip()
                    if s and s != term:
                        term_to_syns[term].add(s)
    print(f"✓ Loaded {len(term_to_syns)} GPT terms from {gpt_jsonl}")
    return term_to_syns


# ---------------- EuroVoc ----------------

def load_eurovoc_termkey_to_syns(xlsx_path: str) -> Dict[str, Set[str]]:
    """normalised form -> all other forms of the same EuroVoc concept"""
    import pandas as pd

    df = pd.read_excel(xlsx_path, sheet_name=0, dtype=str).fillna("")
    col_id = [c for c in df.columns if str(c).strip().lower() == "id"][0]
    col_terms = [c for c in df.columns if "term" in str(c).lower()][0]  # TERMS (PT-NPT)
    col_pt = [c for c in df.columns if str(c).strip().lower() == "pt"]
    col_pt = col_pt[0] if col_pt else None

    by_id: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        cid = row[col_id].strip()
        if not cid:
            continue
        by_id.setdefault(cid, set())
        term = row[col_terms].strip()
        pt = row[col_pt].strip() if col_pt else ""
        if term:
            by_id[cid].update(_variants_for(term))
        if pt:
            by_id[cid].update(_variants_for(pt))

    termkey_to_syns: Dict[str, Set[str]] = defaultdict(set)
    for terms in by_id.values():
        terms = {t for t in terms if t}
        for t in terms:
            key = _normalize_term(t)
            if key:
                termkey_to_syns[key].update(other for other in terms if other != t)
    print(f"✓ Loaded EuroVoc synonyms for {len(termkey_to_syns)} normalized keys from {xlsx_path}")
    return termkey_to_syns


def enrich_with_eurovoc(term_to_syns: Dict[str, Set[str]], xlsx_path: str):
    termkey_to_syns = load_eurovoc_termkey_to_syns(xlsx_path)
    hit = 0
    for term, syns in term_to_syns.items():
        key = _normalize_term(term)
        if key and key in termkey_to_syns:
            before = len(syns)
            syns.update(termkey_to_syns[key])
            hit += len(syns) > before
    print(f"✓ EuroVoc-enriched {hit} GPT terms")


# ---------------- CSO ----------------

def enrich_with_cso(term_to_syns: Dict[str, Set[str]], cso_json: str):
    """cso_synonyms_clean.json: {key: {"preferred": "...", "synonyms": [...]}}"""
    data = json.loads(Path(cso_json).read_text(encoding="utf-8"))

    form_norm_to_preferred: Dict[str, str] = {}
    preferred_to_forms: Dict[str, Set[str]] = {}
    for preferred_key, rec in data.items():
        preferred = (rec.get("preferred") or preferred_key or "").strip()
        if not preferred:
            continue
        forms = {preferred} | {(s or "").strip() for s in rec.get("synonyms") or [] if (s or "").strip()}
        preferred_to_forms[preferred] = forms
        for form in forms:
            k = _normalize_term(form)
            if k:
                form_norm_to_preferred.setdefault(k, preferred)
    print(f"✓ Loaded CSO index: {len(preferred_to_forms)} preferred concepts from {cso_json}")

    hit = 0
    for term, syns in term_to_syns.items():
        pref = form_norm_to_preferred.get(_normalize_term(term))
        if not pref:
            continue
        before = len(syns)
        syns.update(form for form in preferred_to_forms.get(pref, set()) if form and form != term)
        hit += len(syns) > before
    print(f"✓ CSO-enriched {hit} GPT terms")


def symmetric_closure(term_to_syns: Dict[str, Set[str]]):
    for term, syns in list(term_to_syns.items()):
        for s in list(syns):
            term_to_syns.setdefault(s, set()).add(term)


def main():
    parser = argparse.ArgumentParser(description="Build the GPT + ontology term lexicon")
    parser.add_argument("--gpt_terms", required=True, help="Output of gpt-term-expansion.py")
    parser.add_argument("--ontology", choices=["eurovoc", "cso", "none"], required=True)
    parser.add_argument("--ontology_path", default=None,
                        help="eurovoc: eurovoc_export_en.xlsx; cso: cso_synonyms_clean.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    term_to_syns = load_gpt_terms(args.gpt_terms)
    if args.ontology == "eurovoc":
        enrich_with_eurovoc(term_to_syns, args.ontology_path)
    elif args.ontology == "cso":
        enrich_with_cso(term_to_syns, args.ontology_path)
    symmetric_closure(term_to_syns)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for term, syns in term_to_syns.items():
            f.write(json.dumps({"term": term, "synonyms": sorted(syns)}, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {len(term_to_syns)} terms to {out}")


if __name__ == "__main__":
    main()
