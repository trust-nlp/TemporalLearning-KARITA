#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task-relevant term identification and synonym generation with GPT-4o-mini
(paper Sec. 4.3 / Appendix A.2).

For every document of the target split, the LLM picks 3-10 label-informative terms
and generates synonyms / earlier phrasings for each of them. The output is a jsonl
with one line per document:

    {"<id_field>": "...", "gpt_term_expansion": {"entities": [{"term": "...", "synonyms": [...]}, ...]}}

which is merged with an ontology by lexicon/build-lexicon.py.

The API key is read from the OPENAI_API_KEY environment variable.

Example:
  export OPENAI_API_KEY=...
  python lexicon/gpt-term-expansion.py --dataset eurlex \
      --input_jsonl data/EURLEX/T4_test.json --output_jsonl lexicon/eurlex/T4_test.with_terms.json
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

EURLEX_LEVEL1_LABELS = [
    "POLITICS", "INTERNATIONAL RELATIONS", "EUROPEAN UNION", "LAW", "ECONOMICS", "TRADE",
    "FINANCE", "SOCIAL QUESTIONS", "EDUCATION AND COMMUNICATIONS", "SCIENCE",
    "BUSINESS AND COMPETITION", "EMPLOYMENT AND WORKING CONDITIONS", "TRANSPORT", "ENVIRONMENT",
    "AGRICULTURE", "AGRI-FOODSTUFFS", "PRODUCTION", "ENERGY", "INDUSTRY", "GEOGRAPHY",
    "INTERNATIONAL ORGANISATIONS",
]

ARXIV_CS_LABELS = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.GR", "cs.GT",
    "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.NE",
    "cs.NI", "cs.PL", "cs.RO", "cs.SD", "cs.SE", "cs.SI",
]

# dataset -> (dataset name in the prompt, labels, text field, id field)
DATASETS = {
    "eurlex": ("EURLEX Level-1", EURLEX_LEVEL1_LABELS, "text", "celex_id"),
    "arxiv": ("arXiv Computer Science", ARXIV_CS_LABELS, "abstract", "arxiv_id"),
}

TERM_EXPANSION_PROMPT = """
You are helping to build a terminology lexicon for a multi-label
text classification task on the dataset: {dataset_name}.

You will receive:
- One document text.
- The full list of possible labels for this dataset.

Your tasks:

1. Read the document text carefully.

2. Choose 3–10 words or short phrases ("terms") that are especially
   informative for deciding which labels to assign from the given label set.
   These terms should:
   - Be specific to the subject or content of the document, not generic words
     like "data", "paper", "regulation", "method", "result", etc.
   - Strongly indicate one or more labels when they appear.

2. For each selected term, generate:
   - "synonyms": close paraphrases, especially earlier phrasings that could appear in older literature or legal/technical writing, preferred terms if applicable,

You MUST return a SINGLE JSON object with EXACTLY this structure:

{{
  "entities": [
    {{
      "term": "...",
      "synonyms": ["...", "..."]
    }}
  ]
}}


No extra keys. No comments. No explanations outside JSON.

--------------------
DATASET NAME: {dataset_name}

FULL LABEL SET:
{label_block}

--------------------
DOCUMENT TEXT:
{text}
"""


def call_gpt_for_terms(client, text: str, dataset_name: str, labels: List[str],
                       model: str, max_chars: int, retry: int = 3) -> Optional[Dict[str, Any]]:
    """Returns {"entities": [...]} or None if every attempt fails."""
    # very long documents are truncated
    text = text[:max_chars]
    prompt = TERM_EXPANSION_PROMPT.format(
        dataset_name=dataset_name,
        label_block="\n".join(f"- {lab}" for lab in labels),
        text=text,
    )
    for attempt in range(1, retry + 1):
        try:
            resp = client.responses.create(
                model=model,
                instructions="You are a careful information extraction model. Always output strict JSON.",
                input=prompt,
            )
            data = json.loads(resp.output_text)
            if not isinstance(data, dict) or "entities" not in data:
                raise ValueError("JSON does not contain 'entities' key")
            return data
        except Exception as e:
            print(f"[WARN] GPT call failed (attempt {attempt}/{retry}): {e}")
            time.sleep(2 * attempt)
    return None


def main():
    parser = argparse.ArgumentParser(description="GPT term identification and synonym generation")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--input_jsonl", required=True, help="Target split, e.g. T4_test.json")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_chars", type=int, default=4000)
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds between requests")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Please set the OPENAI_API_KEY environment variable.")
    client = OpenAI()
    dataset_name, labels, text_field, id_field = DATASETS[args.dataset]

    # resume: skip documents that are already in the output file
    done = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, encoding="utf-8") as f:
            done = {json.loads(line)[id_field] for line in f if line.strip()}
        print(f"Resuming: {len(done)} documents already processed")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    total = written = 0
    with open(args.input_jsonl, encoding="utf-8") as fin, \
            open(args.output_jsonl, "a", encoding="utf-8", buffering=1) as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            uid = record.get(id_field, f"line_{total}")
            if uid in done:
                continue

            text = record.get(text_field, "")
            terms = None
            if text:
                terms = call_gpt_for_terms(client, text, dataset_name, labels, args.model, args.max_chars)
            else:
                print(f"[WARN] {uid} has empty '{text_field}', skipping GPT call.")

            fout.write(json.dumps({id_field: uid, "gpt_term_expansion": terms}, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"{written} documents written ({total} read)")
            time.sleep(args.sleep)

    print(f"Done. Read {total} documents, wrote {written} new records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
