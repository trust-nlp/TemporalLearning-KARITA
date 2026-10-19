#!/bin/bash
# Step 3: LLM-based term identification + ontology enrichment (EurLex and arXiv-CS).
# MIMIC uses MeSH directly (no LLM, because of the data use agreement).
#
# The LLM is called once on the target split; the lexicon is then reused during adaptation.
#
# Usage: OPENAI_API_KEY=... bash scripts/3-build-lexicon.sh {eurlex|arxiv}
set -euo pipefail

DATASET=${1:?usage: $0 {eurlex|arxiv}}
DATA_DIR=${DATA_DIR:-data}
RES_DIR=${RES_DIR:-resources}
LEX_DIR=$RES_DIR/lexicon/$DATASET

case $DATASET in
  eurlex)
    TARGET=$DATA_DIR/EURLEX/T4_test.json
    ONTOLOGY=eurovoc; ONTOLOGY_PATH=$RES_DIR/EuroVoc/eurovoc_export_en.xlsx
    LEXICON=$LEX_DIR/eurlex_gpt_terms_lexicon_with_eurovoc.jsonl ;;
  arxiv)
    TARGET=$DATA_DIR/ArXiv_CS/T4-test.json
    ONTOLOGY=cso; ONTOLOGY_PATH=$RES_DIR/CSO/cso_synonyms_clean.json
    LEXICON=$LEX_DIR/arxivcs_gpt_terms_lexicon_with_cso.jsonl ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

python lexicon/gpt-term-expansion.py \
  --dataset $DATASET \
  --input_jsonl $TARGET \
  --output_jsonl $LEX_DIR/T4_test.with_terms.jsonl \
  --model gpt-4o-mini

python lexicon/build-lexicon.py \
  --gpt_terms $LEX_DIR/T4_test.with_terms.jsonl \
  --ontology $ONTOLOGY \
  --ontology_path $ONTOLOGY_PATH \
  --output $LEXICON
