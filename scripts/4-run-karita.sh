#!/bin/bash
# Step 4: KARITA adaptation on the target stream (T4 test).
#
# Usage: bash scripts/4-run-karita.sh {mimic|eurlex|arxiv} [variant]
#   variant: full (default) | wo_detection | wo_retrieval | wo_augmentation
#            | uncertainty_only | feature_only | ontology_only
set -euo pipefail

DATASET=${1:?usage: $0 {mimic|eurlex|arxiv} [variant]}
VARIANT=${2:-full}
DATA_DIR=${DATA_DIR:-data}
OUT_DIR=${OUT_DIR:-outputs}
RES_DIR=${RES_DIR:-resources}
SEED=${SEED:-42}
# term order in the MeSH augmentor depends on Python's string hashing
export PYTHONHASHSEED=$SEED

SOURCE_MODEL=$OUT_DIR/$DATASET/source_model
SHIFT_DIR=$OUT_DIR/$DATASET/shift_detector

case $DATASET in
  mimic)
    # retrieval pool: all labeled source-period data (T1-T3 train)
    SOURCE=$DATA_DIR/MIMIC-IV-Notes/T123_train.json; TARGET=$DATA_DIR/MIMIC-IV-Notes/T4_test.json
    LEXICON_ARGS="--mesh_desc $RES_DIR/MeSH/desc2025.xml --mesh_supp $RES_DIR/MeSH/supp2025.xml" ;;
  eurlex)
    SOURCE=$DATA_DIR/EURLEX/T123_all.json; TARGET=$DATA_DIR/EURLEX/T4_test.json
    LEXICON_ARGS="--lexicon_type lexicon_jsonl --lexicon_path $RES_DIR/lexicon/eurlex/eurlex_gpt_terms_lexicon_with_eurovoc.jsonl" ;;
  arxiv)
    SOURCE=$DATA_DIR/ArXiv_CS/T123-train.json; TARGET=$DATA_DIR/ArXiv_CS/T4-test.json
    # Table 2 was obtained with the CSO lexicon. To use the GPT + CSO lexicon instead:
    #   --lexicon_type lexicon_jsonl --lexicon_path $RES_DIR/lexicon/arxiv/arxivcs_gpt_terms_lexicon_with_cso.jsonl
    LEXICON_ARGS="--lexicon_type cso_wikidata --lexicon_path $RES_DIR/CSO/cs_wikidata_lexicon_from_cso.json" ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

FEATURE_UIDS=$SHIFT_DIR/feature/uid_feature_shift-10%.txt
ONTOLOGY_UIDS=$SHIFT_DIR/ontology/uid_ontology_shift-tail-10%.txt
NUM_RANDOM=$(( $(wc -l < $TARGET) / 10 ))   # 10% of the target split

DETECTION="--selection_method both --feature_shift_uids $FEATURE_UIDS --ontology_shift_uids $ONTOLOGY_UIDS"
EXTRA=""
case $VARIANT in
  full)             ;;
  wo_detection)     DETECTION="--selection_method none --random_target_total $NUM_RANDOM" ;;
  wo_retrieval)     EXTRA="--retrieval least_similar" ;;
  wo_augmentation)  EXTRA="--augmentation_option option0" ;;
  uncertainty_only) DETECTION="--selection_method both" ;;
  feature_only)     DETECTION="--selection_method none --feature_shift_uids $FEATURE_UIDS" ;;
  ontology_only)    DETECTION="--selection_method none --ontology_shift_uids $ONTOLOGY_UIDS" ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac

python karita.py \
  --dataset $DATASET \
  --source_file $SOURCE \
  --target_file $TARGET \
  --model_path $SOURCE_MODEL \
  --output_dir $OUT_DIR/$DATASET/karita/$VARIANT-seed$SEED \
  --seed $SEED \
  --batch_size_per_iteration 100 \
  --batch_size_train 4 \
  --batch_size_test 8 \
  --num_epochs 3 \
  --learning_rate 3e-5 \
  --threshold_conf 0.5 \
  --threshold_entropy 0.25 \
  --k_retrieve 3 \
  --prediction_mode 2 \
  $DETECTION $LEXICON_ARGS $EXTRA
