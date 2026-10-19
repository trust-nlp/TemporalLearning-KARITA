#!/bin/bash
# Step 1: train the source model on the earliest interval (T1) and evaluate it on T1 / T4.
#
# Usage: bash scripts/1-train-source.sh {mimic|eurlex|arxiv}
#
# The hyper-parameters below are the ones recorded in training_args.bin of the
# checkpoints used in the paper (classification.py default max_seq_length = 384).
set -euo pipefail

DATASET=${1:?usage: $0 {mimic|eurlex|arxiv}}
DATA_DIR=${DATA_DIR:-data}
OUT_DIR=${OUT_DIR:-outputs}

case $DATASET in
  mimic)
    MODEL=allenai/longformer-base-4096; SPLIT_DIR=$DATA_DIR/MIMIC-IV-Notes
    TRAIN=$SPLIT_DIR/T1_train.json; TEST=$SPLIT_DIR/T1_test.json
    TEXT=text; LABEL=label; EPOCHS=10; LR=3e-5; BS=8 ;;      # 2 GPUs -> effective batch 16
  eurlex)
    MODEL=xlm-roberta-base; SPLIT_DIR=$DATA_DIR/EURLEX
    TRAIN=$SPLIT_DIR/T1_train.json; TEST=$SPLIT_DIR/T1_test.json
    TEXT=text; LABEL=level_1; EPOCHS=10; LR=3e-5; BS=8 ;;    # 1 GPU  -> effective batch 8
  arxiv)
    MODEL=xlm-roberta-base; SPLIT_DIR=$DATA_DIR/ArXiv_CS
    TRAIN=$SPLIT_DIR/T1-train.json; TEST=$SPLIT_DIR/T1-test.json
    TEXT=abstract; LABEL=labels; EPOCHS=3; LR=5e-5; BS=8 ;;  # 2 GPUs -> effective batch 16
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

SOURCE_MODEL=$OUT_DIR/$DATASET/source_model

python classification.py \
  --model_name_or_path $MODEL \
  --train_file $TRAIN \
  --test_file $TEST \
  --text_column_names $TEXT \
  --label_column_name $LABEL \
  --do_train --do_predict \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --per_device_train_batch_size $BS \
  --save_strategy epoch --save_total_limit 1 \
  --overwrite_output_dir \
  --output_dir $SOURCE_MODEL

# source model on the target split (Table 2, part II);
# --train_file is only read to rebuild the label list
python classification.py \
  --model_name_or_path $SOURCE_MODEL \
  --train_file $TRAIN \
  --test_file $( [ $DATASET = arxiv ] && echo $SPLIT_DIR/T4-test.json || echo $SPLIT_DIR/T4_test.json ) \
  --text_column_names $TEXT \
  --label_column_name $LABEL \
  --do_predict \
  --output_dir $SOURCE_MODEL/test_on_T4
