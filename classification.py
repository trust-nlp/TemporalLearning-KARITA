#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
from utils.config_classes import ModelArguments, DataTrainingArguments
import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import evaluate
import numpy as np
import json, math
import torch
from datasets import Value, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
#from scipy.special import expit as sigmoid

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

# this is to find the low confidence example(hard example) need to be retrieved
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_entropy(p, eps=1e-12):
    # p: [N, L] in [0,1]
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))  # [N, L]

# ----- parameters -----
TAU = 0.5            
CONF_THRESH = 0.5    
ENTROPY_THRESH = 0.25
TOPK_RETRIEVE = 5   
#------------------

logger = logging.getLogger(__name__)


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a mutli-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined togather
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token, 
            #download_mode='force_redownload'
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {}            
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir,token=model_args.token)

        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
            else:
                raise ValueError("Need either a dataset name or a test file for `do_predict`.")
            
        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")
    

    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # Trying to have good defaults here, don't hesitate to tweak to your needs.

    is_regression = (
        raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if data_args.do_regression is None
        else data_args.do_regression
    )

    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        # regession requires float as label type, let's cast it if needed
        for split in raw_datasets.keys():
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                )
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error

    else:  # classification
        if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        
        label_list = get_label_list(raw_datasets, split="train")

    
        for split in ["validation", "test"]:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, "
                        f"they will be IGNORED (not added to label_list)."
                    )

        label_list = [l for l in label_list if l != -1 and str(l) != "-1"]

        label_list = sorted(set(str(l) for l in label_list))
        num_labels = len(label_list)

        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train and not is_regression:  # classification, training
        label_to_id = {v: i for i, v in enumerate(label_list)}
        print('NOTICE\n','label_to_id:',label_to_id,'\n','model.config.label2id:',model.config.label2id )
        # update config with label infos
        if model.config.label2id != label_to_id:
            logger.warning(
                "The label2id key in the model config.json is not equal to the label2id key of this "
                "run. You can ignore this if you are doing finetuning."
            )
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif not is_regression:  # classification, but not training
        #print('label_to_id:',{v: i for i, v in enumerate(label_list)},'\n','model.config.label2id:',model.config.label2id )
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(model.config.label2id))
        label_to_id = model.config.label2id
    else:  # regression
        label_to_id = None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        # print('labels in multi_labels_to_ids:',labels)
        # print('label_to_id in multi_labels_to_ids:',label_to_id)
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids
    

    # IGNORE_INDEX = -100  
    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
                
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        # remove label column if it exists
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    '''# Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")'''

    if data_args.metric_name is not None:
        metric = (
            evaluate.load(data_args.metric_name, config_name="multilabel")
            if is_multi_label
            else evaluate.load(data_args.metric_name)
        )
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        if is_regression:
            metric = evaluate.load("mse")
            logger.info("Using mean squared error (mse) as regression score, you can use --metric_name to overwrite.")
        else:
            if is_multi_label:
                '''metric = evaluate.load("f1",config_name="multilabel")
                logger.info(
                    "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
                )'''
                f1_metric = evaluate.load("f1",config_name="multilabel")
                accuracy_metric = evaluate.load("accuracy",config_name="multilabel")
                precision_metric = evaluate.load("precision",config_name="multilabel")
                recall_metric = evaluate.load("recall",config_name="multilabel")
                logger.info(
                    "Using multilabel F1, accuracy, precision, recall for multi-label classification task, you can use --metric_name to overwrite."
                )
            else:
                #metric = evaluate.combine("f1","accuracy","precision","recall")
                f1_metric = evaluate.load("f1")
                accuracy_metric = evaluate.load("accuracy")
                precision_metric = evaluate.load("precision")
                recall_metric = evaluate.load("recall")
                logger.info("Using F1, accuracy, precision, recall as classification score, you can use --metric_name to overwrite.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        elif is_multi_label:
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            # Micro F1 is commonly used in multi-label classification
            #result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
            accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
            sample_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="samples")["f1"]
            sample_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="samples")["precision"]
            sample_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="samples")["recall"]
            micro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="micro")["f1"]
            micro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="micro")["precision"]
            micro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="micro")["recall"]
            macro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
            macro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="macro")["precision"]
            macro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="macro")["recall"]
            weighted_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"]
            weighted_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["precision"]
            weighted_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["recall"]
            result = {
                "accuracy": accuracy,
                "sample_precision": sample_precision,
                "sample_recall": sample_recall, 
                "sample_f1": sample_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall, 
                "micro_f1": micro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_precision":weighted_precision,
                "weighted_recall":weighted_recall,
                "weighted_f1":weighted_f1,
            }
        else:
            preds = np.argmax(preds, axis=1)
            if num_labels == 2:
                accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
                binary_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="binary")["precision"]
                binary_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="binary")["recall"]
                binary_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="binary")["f1"]
                result = {
                    "accuracy": accuracy,
                    "precision": binary_precision,
                    "recall": binary_recall,
                    "f1": binary_f1,
                }
            else:    
                accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
                micro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="micro")["f1"]
                micro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="micro")["precision"]
                micro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="micro")["recall"]
                macro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
                macro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="macro")["precision"]
                macro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="macro")["recall"]
                weighted_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"]
                weighted_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["precision"]
                weighted_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["recall"]
                result = {
                    "accuracy": accuracy,
                    "micro_precision": micro_precision,
                    "micro_recall": micro_recall, 
                    "micro_f1": micro_f1,
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1": macro_f1,
                    "weighted_precision":weighted_precision,
                    "weighted_recall":weighted_recall,
                    "weighted_f1":weighted_f1,
                }
        '''if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()'''
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    
    if training_args.do_predict:
        logger.info("*** Predict ***")
        pred_out = trainer.predict(predict_dataset, metric_key_prefix="predict")
        raw_logits = pred_out.predictions  # [N, L] logits
        # print('raw_logits',raw_logits)
        metrics = pred_out.metrics

        # uid
        # ---- pick uid column (celex_id or uid) and normalize to str ----
        if "celex_id" in predict_dataset.features:
            uid_col = "celex_id"
        elif "arxiv_id" in predict_dataset.features:
            uid_col = "arxiv_id"
        elif "link" in predict_dataset.features:
            uid_col = "link"
        elif "uid" in predict_dataset.features:
            uid_col = "uid"
        else:
            raise RuntimeError("predict_dataset needs a 'uid' or 'celex_id' column to record hard examples.")
        uids = [str(u) for u in predict_dataset[uid_col]]


        probs = sigmoid(raw_logits)  # [N, L] in [0,1]
        max_prob = probs.max(axis=1)  # [N]

        # entropy = -p log p - (1-p) log (1-p) for each label, then average over labels
        ent = binary_entropy(probs)         # [N, L]
        avg_entropy = ent.mean(axis=1)      # [N]

        # empty prediction: if all p_i < TAU
        is_empty = (probs < TAU).all(axis=1)  # [N] bool

        # predicted labels: those with p_i >= TAU
        bin_pred = (probs >= TAU).astype(int)   # [N, L]
        pred_label_lists = []
        for row in bin_pred:
            indices = [i for i, v in enumerate(row) if v == 1]
            pred_label_lists.append([label_list[i] for i in indices])

        # record UID
        # low_conf_uids = [int(u) for u, c in zip(uids, max_prob) if c < CONF_THRESH]
        # high_entropy_uids = [int(u) for u, h in zip(uids, avg_entropy) if h > ENTROPY_THRESH]
        # #empty_pred_uids = [int(u) for u, e in zip(uids, is_empty) if e]
        low_conf_uids = [u for u, c in zip(uids, max_prob) if c < CONF_THRESH]
        high_entropy_uids = [u for u, h in zip(uids, avg_entropy) if h > ENTROPY_THRESH]
        
        retrieve_uids = sorted(set(low_conf_uids) | set(high_entropy_uids))

        # export JSONL（uid, max_prob, avg_entropy, is_empty, pred_labels）
           
        details_path = os.path.join(training_args.output_dir, "pred_details.jsonl")
        with open(details_path, "w", encoding="utf-8") as f:
            for i, (u, c, h, e, labs) in enumerate(zip(uids, max_prob, avg_entropy, is_empty, pred_label_lists)):
                if "label" in predict_dataset.features:
                    # HuggingFace Dataset: predict_dataset[i]["label"] 是 multi-hot 或 id 列表
                    raw_label = predict_dataset[i]["label"]
                    # print('raw_label',raw_label)
                    
                    if isinstance(raw_label, list):
                        # 1: multi-hot vector (float 0/1)
                        if all(isinstance(v, (int, float)) for v in raw_label):
                            if max(raw_label) <= 1:  #  multi-hot
                                true_labels = [label_list[j] for j, v in enumerate(raw_label) if int(v) == 1]
                            else:  
                                true_labels = [label_list[int(j)] for j in raw_label]
                        else:
                            true_labels = raw_label
                    else:
                        true_labels = []


                # predic and true label set for analysis
                pred_set = set(labs)
                true_set = set(true_labels)
                correct = len(pred_set & true_set)
                wrong_pred = len(pred_set - true_set)
                fully_correct = (pred_set == true_set)

                rec = {
                    "uid": u,#int(u),
                    "max_prob": float(c),
                    "avg_entropy": float(h),
                    "is_empty": bool(e),
                    "predicted_labels": labs,
                    "true_labels": true_labels,
                    "correct_count": correct,
                    "wrong_count": wrong_pred,
                    "is_fully_correct": fully_correct
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Per-sample prediction details (with true labels) saved to {details_path}")

        def dump_list(lst, name):
            path = os.path.join(training_args.output_dir, name)
            with open(path, "w") as f:
                for u in lst:
                    f.write(f"{u}\n")
            logger.info(f"Saved {name}: {len(lst)} uids @ {path}")

        dump_list(low_conf_uids, "uids_low_conf.txt")
        dump_list(high_entropy_uids, "uids_high_entropy.txt")
        dump_list(retrieve_uids, "uids_for_retrieval.txt")

        #  metrics & predict_results.txt 
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()