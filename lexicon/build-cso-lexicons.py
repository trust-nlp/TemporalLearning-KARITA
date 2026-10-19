#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the two CSO-based lexicons used for arXiv-CS from the assets of the
CSO Classifier (https://github.com/angelosalatino/cso-classifier, v4.0.0, CSO v3.55):

    pip install cso-classifier
    python -c "from cso_classifier import CSOClassifier; CSOClassifier.setup()"   # downloads the assets

1) synonyms : cso.csv  --(klink:relatedEquivalent clusters)-->  cso_synonyms_clean.json
              {"<preferred>": {"preferred": "...", "synonyms": [...], "count": n}, ...}
              used by build-lexicon.py --ontology cso (GPT + CSO lexicon)

2) wikidata : CSO topics found in token-to-cso-combined.json, each looked up on Wikidata
              (best match + English label/aliases)  -->  cs_wikidata_lexicon_from_cso.json
              {"<preferred>": {"wikidata_id": "Q...", "forms": [...], "source": {...}}, ...}
              used by detect-ontology-shift.py (arXiv) and karita.py --lexicon_type cso_wikidata
              NOTE: Wikidata changes over time, so a new run will not give exactly the same file.

Examples:
  python lexicon/build-cso-lexicons.py synonyms --assets_dir <cso_classifier>/assets --output resources/CSO/cso_synonyms_clean.json
  python lexicon/build-cso-lexicons.py wikidata --assets_dir <cso_classifier>/assets --output resources/CSO/cs_wikidata_lexicon_from_cso.json \
      --user_agent "your-project/0.1 (contact: you@example.org)"
"""

import argparse
import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def default_assets_dir():
    try:
        import cso_classifier
        return str(Path(cso_classifier.__file__).parent / "assets")
    except ImportError:
        return None


# =========================
# 1) relatedEquivalent clusters -> cso_synonyms_clean.json
# =========================

def build_synonyms(assets_dir: Path, output: Path):
    import pandas as pd

    df = pd.read_csv(assets_dir / "cso.csv", sep=";", header=None, names=["source", "relation", "target"])
    df = df[df["relation"] == "klink:relatedEquivalent"]

    neighbours = defaultdict(set)
    for _, row in df.iterrows():
        a = row["source"].strip().lower()
        b = row["target"].strip('"').strip().lower()
        neighbours[a].update((a, b))
        neighbours[b].update((a, b))

    # connected components
    visited, groups = set(), []
    for node in neighbours:
        if node in visited:
            continue
        stack, comp = [node], set()
        while stack:
            x = stack.pop()
            if x not in visited:
                visited.add(x)
                comp.add(x)
                stack.extend(neighbours[x] - visited)
        groups.append(comp)

    lexicon = {}
    for comp in groups:
        pref = sorted(comp, key=lambda x: (len(x), x))[0]  # preferred: shortest, then alphabetical
        other = sorted(comp - {pref})
        lexicon[pref] = {"preferred": pref, "synonyms": other, "count": 1 + len(other)}

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(lexicon, f, indent=2)
    print(f"Saved {len(lexicon)} concepts to {output}")


# =========================
# 2) CSO topics + Wikidata aliases -> cs_wikidata_lexicon_from_cso.json
# =========================

def clean_form(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def candidate_terms(assets_dir: Path):
    """Preferred labels of the CSO topics that occur in token-to-cso-combined.json."""
    try:
        with open(assets_dir / "cso.p", "rb") as f:
            cso_data = pickle.load(f)
    except FileNotFoundError:
        cso_data = {}
    topic2preferred = {}
    for topic_uri, info in cso_data.items():
        topic_id = topic_uri.rsplit("/", 1)[-1]
        labels = info.get("labels", []) if isinstance(info, dict) else []
        if isinstance(labels, str):
            topic2preferred[topic_id] = labels
        elif isinstance(labels, list) and labels:
            topic2preferred[topic_id] = labels[0]

    with open(assets_dir / "token-to-cso-combined.json", encoding="utf-8") as f:
        token2topics = json.load(f)
    topic_ids = {e.get("topic") for entries in token2topics.values() for e in entries
                 if isinstance(e.get("topic"), str)}

    terms = set()
    for tid in topic_ids:
        preferred = clean_form(topic2preferred.get(tid, tid.replace("_", " ")))
        if re.search(r"[a-zA-Z]", preferred) and len(preferred) > 2:
            terms.add(preferred)
    print(f"{len(topic_ids)} CSO topics, {len(terms)} unique terms to query")
    return sorted(terms)


def is_good_alias(preferred: str, alias: str) -> bool:
    """Conservative alias filter."""
    p, a = clean_form(preferred).lower(), clean_form(alias).lower()
    if not re.search(r"[a-zA-Z]", a) or len(a) <= 1 or a == p or re.search(r"https?://", a):
        return False
    if len(a) <= 5:  # short aliases such as CNN, GAN
        return True
    # longer aliases must share a word with the preferred label
    p_words = {w for w in re.split(r"\W+", p) if len(w) > 2}
    a_words = {w for w in re.split(r"\W+", a) if len(w) > 2}
    return not (p_words and a_words and not (p_words & a_words))


def wikidata_synonyms(session, term: str, language: str = "en"):
    r = session.get(WIKIDATA_API, timeout=10, params={
        "action": "wbsearchentities", "search": term, "language": language, "format": "json", "limit": 1})
    r.raise_for_status()
    hits = r.json().get("search", [])
    if not hits:
        return None
    qid = hits[0]["id"]

    r = session.get(WIKIDATA_API, timeout=10, params={
        "action": "wbgetentities", "ids": qid, "languages": language, "props": "labels|aliases", "format": "json"})
    r.raise_for_status()
    entity = r.json()["entities"][qid]
    label = entity.get("labels", {}).get(language, {}).get("value") or term
    aliases = [a["value"] for a in entity.get("aliases", {}).get(language, [])]

    preferred = clean_form(label)
    return qid, preferred, sorted({clean_form(a) for a in aliases if is_good_alias(preferred, clean_form(a))})


def build_wikidata(assets_dir: Path, output: Path, user_agent: str, sleep: float):
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    terms = candidate_terms(assets_dir)
    lexicon = {}
    for i, term in enumerate(terms, start=1):
        try:
            info = wikidata_synonyms(session, term)
        except Exception as e:
            print(f"[{i}/{len(terms)}] {term}: Wikidata error {e}")
            continue
        if info is not None:
            qid, preferred, aliases = info
            forms = set(aliases) | {preferred, term}
            lexicon[preferred] = {
                "wikidata_id": qid,
                "forms": sorted({clean_form(f) for f in forms}),
                "source": {"wikidata": True, "dbpedia": False},
            }
        if i % 500 == 0:
            print(f"[{i}/{len(terms)}] lexicon size {len(lexicon)}")
        time.sleep(sleep)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(lexicon)} concepts to {output}")


def main():
    parser = argparse.ArgumentParser(description="Build CSO-based lexicons for arXiv-CS")
    parser.add_argument("mode", choices=["synonyms", "wikidata"])
    parser.add_argument("--assets_dir", default=default_assets_dir(),
                        help="CSO Classifier assets directory (default: from the installed cso_classifier)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--user_agent", default="karita-cso-lexicon/0.1",
                        help="Wikidata asks for a descriptive User-Agent with contact information")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds between Wikidata requests")
    args = parser.parse_args()
    if not args.assets_dir:
        parser.error("--assets_dir is required when cso_classifier is not installed")

    if args.mode == "synonyms":
        build_synonyms(Path(args.assets_dir), Path(args.output))
    else:
        build_wikidata(Path(args.assets_dir), Path(args.output), args.user_agent, args.sleep)


if __name__ == "__main__":
    main()
