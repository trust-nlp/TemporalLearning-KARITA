# TemporalLearning-KARITA

**Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation** (ACL 2026)

KARITA is a framework for temporal domain adaptation in specialized multi-label text classification. It addresses the challenge of model performance degradation when models trained on historical data are applied to future data, due to temporal distribution shifts in specialized domains.

KARITA introduces:

- **Multi-signal shift detection** combining uncertainty-based, feature-based, and ontology-based signals to identify samples affected by temporal drift
- **Shift-aware retrieval** of semantically similar historical samples to provide targeted support for shifted instances
- **Knowledge-driven augmentation** using external ontologies (MeSH, EuroVoc, CSO) and LLM-based synonym generation to bridge terminological gaps across time periods

The framework is evaluated across three domains: clinical notes (MIMIC-IV-Notes), legal documents (EUR-Lex), and computer science papers (arXiv-CS).

## Repository structure

```
classification.py                 # Step 1  train / evaluate the source model (HF Trainer)
shift_detection/
  detect-feature-shift.py         # Step 2  feature-based shift score F(x)
  detect-ontology-shift.py        # Step 2  ontology-based terminology shift O_tail(x)
  select-top-shift.py             # Step 2  keep the top-rho target samples of a score
lexicon/
  gpt-term-expansion.py           # Step 3  GPT-4o-mini term identification + synonym generation
  build-lexicon.py                # Step 3  merge GPT terms with EuroVoc / CSO
  build-cso-lexicons.py           #         CSO lexicons for arXiv-CS (from the CSO Classifier assets)
karita.py                         # Step 4  detection -> retrieval -> augmentation -> update loop
utils/
  config_classes.py               # arguments of classification.py
  syno_aug_pipline.py             # MeSH / EuroVoc / CSO synonym augmentation
preprocess/
  split-eurlex.py                 # Step 0  EurLex temporal splits (from MultiEURLEX)
  split-arxiv.py                  # Step 0  arXiv-CS temporal splits (from nick007x/arxiv-papers)
scripts/                          # one shell script per step
```

## Setup

```bash
conda create -n karita python=3.10 && conda activate karita
pip install -r requirements.txt
```

The scripts read three directories, all configurable through environment variables:

| variable   | default     | content |
|------------|-------------|---------|
| `DATA_DIR` | `data`      | temporal splits of the three datasets |
| `RES_DIR`  | `resources` | ontologies and lexicons |
| `OUT_DIR`  | `outputs`   | models, shift scores and results |

## Data

Each split is a jsonl file (one JSON object per line).

| dataset | files (`$DATA_DIR/...`) | id | text | labels |
|---|---|---|---|---|
| MIMIC-IV-Notes | `MIMIC-IV-Notes/T{1,2,3,4}_{train,test}.json`, `T123_train.json` | `uid` | `text` | `label` (top-50 ICD-10 codes) |
| EurLex | `EURLEX/T{1,2,3,4}_{train,test}.json`, `T123_all.json` | `celex_id` | `text` | `level_1` (21 EuroVoc level-1 ids) |
| arXiv-CS | `ArXiv_CS/T{1,2,3,4}-{train,test}.json`, `T123-train.json` | `arxiv_id` | `abstract` | `labels` (top-30 `cs.*` categories) |

Every dataset is split into four time intervals T1–T4, each with a train and a test split:

| dataset | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| MIMIC-IV-Notes | 2008–2010 | 2011–2013 | 2014–2016 | 2017–2019 |
| EurLex | 1958–1985 | 1986–2000 | 2001–2010 | 2011–2016 |
| arXiv-CS | 1991–2010 | 2011–2015 | 2016–2020 | 2021–2025 |

The source model is trained on `T1` train; KARITA adapts it on the stream of `T4` test documents and retrieves
labeled source documents from `T1–T3` (`T123_train.json` / `T123_all.json` / `T123-train.json`).

The EurLex and arXiv-CS splits are produced by the scripts in `preprocess/` (fixed seed 42; the files
used in the paper are reproduced byte for byte). MIMIC-IV-Notes notes are processed following
[TemporalAssessment](https://github.com/trust-nlp/TemporalAssessment) (`data/MIMIC-IV-Note/build-mimic.py`, `map_icd10_and_split.py`).

```bash
# EurLex: pool the MultiEURLEX train/dev/test files, keep English, bucket by publication year,
# 70/30 split per interval; T123_all.json = all T1-T3 documents (train + test)
python preprocess/split-eurlex.py --raw_dir <multi_eurlex dir with train/dev/test.jsonl> --out_dir $DATA_DIR/EURLEX
# arXiv-CS: papers with a cs.* primary category, labels = 30 most frequent cs.* codes,
# year from the submission date, 70/30 split per interval; T123-train.json = T1-T3 train
python preprocess/split-arxiv.py --out_dir $DATA_DIR/ArXiv_CS
```

Data sources: [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/) (credentialed access; ICD-9 codes mapped to ICD-10 with [ICD-Mappings](https://github.com/snovaisg/ICD-Mappings)),
[MultiEURLEX](https://huggingface.co/datasets/nlpaueb/multi_eurlex) (English portion), arXiv abstracts from [nick007x/arxiv-papers](https://huggingface.co/datasets/nick007x/arxiv-papers).

## External knowledge resources

```
$RES_DIR/MeSH/desc2025.xml, supp2025.xml                      # https://www.nlm.nih.gov/mesh/
$RES_DIR/EuroVoc/eurovoc_export_en.xlsx                       # EuroVoc 4.22 English export
$RES_DIR/CSO/cs_wikidata_lexicon_from_cso.json                 # built below
$RES_DIR/CSO/cso_synonyms_clean.json                           # built below
$RES_DIR/lexicon/{eurlex,arxiv}/*.jsonl                        # produced by step 3
```

For arXiv-CS, `cs_wikidata_lexicon_from_cso.json` (CSO topics + Wikidata aliases) is the lexicon used in the paper,
both for the ontology shift score and for augmentation. `cso_synonyms_clean.json` (CSO relatedEquivalent clusters)
is only needed for the optional GPT + CSO lexicon (`scripts/3-build-lexicon.sh arxiv`).
The two CSO files are derived from the assets of the [CSO Classifier](https://github.com/angelosalatino/cso-classifier)
(v4.0.0, [CSO](https://cso.kmi.open.ac.uk/) v3.55):

```bash
pip install cso-classifier
python -c "from cso_classifier import CSOClassifier; CSOClassifier.setup()"   # downloads the assets
# CSO relatedEquivalent clusters
python lexicon/build-cso-lexicons.py synonyms --output $RES_DIR/CSO/cso_synonyms_clean.json
# CSO topics + English label / aliases of their best Wikidata match (needs internet;
# Wikidata keeps changing, so the file will differ slightly from the one used in the paper)
python lexicon/build-cso-lexicons.py wikidata --output $RES_DIR/CSO/cs_wikidata_lexicon_from_cso.json \
    --user_agent "your-project/0.1 (contact: you@example.org)"
```

Older ontology releases closer to the target period can be used by pointing the scripts to them
(e.g. MeSH 2019 for MIMIC T4 2017–2019, EuroVoc 4.8 for EurLex); the MIMIC sensitivity analysis
(Appendix B.1) was run with MeSH 2019.

## Pipeline

```bash
# 1. source model (T1), evaluated on T1 and T4
bash scripts/1-train-source.sh mimic

# 2. feature / ontology shift scores on T4 test + top-10% selection
bash scripts/2-detect-shift.sh mimic

# 3. LLM term lexicon (EurLex and arXiv-CS only)
OPENAI_API_KEY=... bash scripts/3-build-lexicon.sh eurlex

# 4. KARITA adaptation
bash scripts/4-run-karita.sh mimic                  # full method
bash scripts/4-run-karita.sh mimic wo_retrieval     # ablations, see below
```

Results are written to `$OUT_DIR/<dataset>/karita/<variant>-seed<seed>/`:
`results/metrics_summary.json` (sample / micro / macro / weighted P, R, F1), `results/summary.json`
(configuration, detection and training statistics, timing), `results/retrieval_records.json`
(retrieved source ids and similarities for every detected target sample) and
`final_predictions/pred_details.jsonl` (per-document predictions).

### Main hyper-parameters

| | value |
|---|---|
| target samples per adaptation step (`--batch_size_per_iteration`) | 100 |
| uncertainty thresholds `tau_p` / `tau_H` | 0.5 / 0.25 |
| feature / ontology shift: top-rho | 10% |
| retrieved sources per detected sample `k` | 3 |
| terms replaced per retrieved document | ≤ 3 |
| update: epochs / learning rate / batch size | 3 / 3e-5 / 4 |
| max sequence length | 4096 (MIMIC, Longformer), 512 (EurLex, arXiv-CS, XLM-R) |

Source models (`classification.py`): MIMIC Longformer-base 10 epochs, lr 3e-5, effective batch 16;
EurLex XLM-R-base 10 epochs, lr 3e-5, batch 8; arXiv-CS XLM-R-base 3 epochs, lr 5e-5, effective batch 16.

### Ablations (`scripts/4-run-karita.sh <dataset> <variant>`)

| variant | what changes |
|---|---|
| `wo_detection` | random 10% of the target samples instead of the three detectors |
| `wo_retrieval` | the k *least* similar source samples are retrieved |
| `wo_augmentation` | retrieved sources are used without synonym replacement |
| `uncertainty_only` / `feature_only` / `ontology_only` | a single shift detector |

For the sensitivity analysis, change `--k_retrieve` and create other top-rho files with
`RHO="0.05 0.1 0.2 0.3" bash scripts/2-detect-shift.sh <dataset>`.

## Implementation details

- **Uncertainty trigger** (`karita.py --selection_method both`): a target sample is selected when its mean
  binary entropy is above `tau_H` **or** its maximum label probability is below `tau_p`, computed with the
  current (adapted) model.
- **Feature shift**: squared Euclidean distance of the [CLS] embedding to the source mean, standardised with
  the source mean/std of that distance and passed through a sigmoid; only the ranking (top-rho) is used.
- **Ontology shift**: mean surprisal `-log p_src(c)` over the ontology concepts of a document that also occur
  in the source split (concepts with fewer than 5 occurrences in source + target are ignored).
- **Retrieval** uses the [CLS] embeddings of the *frozen source model* for both source and target documents;
  source embeddings are cached in `<model_path>/emb_cache` (or `--emb_cache_dir`).
- **Augmentation**
  - MIMIC: MeSH terms are replaced with a random synonym of the same descriptor.
  - EurLex: terms of the GPT + EuroVoc lexicon are rewritten into their canonical term.
    `--fast_match` switches to regex matching (much faster, but the matched terms differ slightly).
  - arXiv-CS: CSO terms are replaced with a random other form of the same concept
    (`--lexicon_type cso_wikidata`); the GPT + CSO lexicon can be used with `--lexicon_type lexicon_jsonl`.
- After every adaptation step the model keeps training from its current weights (fresh AdamW per step);
  with `--prediction_mode 2` the final model predicts the whole target split once.

## Citation

```bibtex
@inproceedings{liu-etal-2026-knowledge,
    title = "Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation",
    author = "Liu, Weisi and Han, Guangzeng and Huang, Xiaolei",
    booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2026",
    pages = "36130--36143",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1672/",
}
```
