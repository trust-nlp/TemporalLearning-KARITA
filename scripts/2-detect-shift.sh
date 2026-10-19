#!/bin/bash
# Step 2: pre-compute the feature- and ontology-based shift scores of the target split
# and keep the top-rho (default 10%) samples of each.
# (The uncertainty-based detection runs online inside karita.py.)
#
# Usage: bash scripts/2-detect-shift.sh {mimic|eurlex|arxiv}
set -euo pipefail

DATASET=${1:?usage: $0 {mimic|eurlex|arxiv}}
DATA_DIR=${DATA_DIR:-data}
OUT_DIR=${OUT_DIR:-outputs}
RES_DIR=${RES_DIR:-resources}
RHO=${RHO:-"0.1"}            # e.g. RHO="0.05 0.1 0.2 0.3" for the sensitivity analysis

SOURCE_MODEL=$OUT_DIR/$DATASET/source_model
SHIFT_DIR=$OUT_DIR/$DATASET/shift_detector

case $DATASET in
  mimic)
    SOURCE=$DATA_DIR/MIMIC-IV-Notes/T1_train.json; TARGET=$DATA_DIR/MIMIC-IV-Notes/T4_test.json
    ONTOLOGY_ARGS="--mesh_desc $RES_DIR/MeSH/desc2025.xml --mesh_supp $RES_DIR/MeSH/supp2025.xml" ;;
  eurlex)
    SOURCE=$DATA_DIR/EURLEX/T1_train.json; TARGET=$DATA_DIR/EURLEX/T4_test.json
    ONTOLOGY_ARGS="--eurovoc_xlsx $RES_DIR/EuroVoc/eurovoc_export_en.xlsx" ;;
  arxiv)
    SOURCE=$DATA_DIR/ArXiv_CS/T1-train.json; TARGET=$DATA_DIR/ArXiv_CS/T4-test.json
    ONTOLOGY_ARGS="--cs_lexicon_json $RES_DIR/CSO/cs_wikidata_lexicon_from_cso.json" ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

# feature-based shift (GPU)
python shift_detection/detect-feature-shift.py \
  --dataset $DATASET \
  --source_jsonl $SOURCE \
  --target_jsonl $TARGET \
  --model_path $SOURCE_MODEL \
  --out_dir $SHIFT_DIR/feature

# ontology-based terminology shift (CPU)
python shift_detection/detect-ontology-shift.py \
  --dataset $DATASET \
  --source_jsonl $SOURCE \
  --target_jsonl $TARGET \
  $ONTOLOGY_ARGS \
  --min_total 5 \
  --out_dir $SHIFT_DIR/ontology

# top-rho selection -> uid_feature_shift-10%.txt / uid_ontology_shift-tail-10%.txt
python shift_detection/select-top-shift.py --scores $SHIFT_DIR/feature/feature_shift_scores.tsv --rho $RHO
python shift_detection/select-top-shift.py --scores $SHIFT_DIR/ontology/ontology4_tail_shift_scores.tsv --rho $RHO
