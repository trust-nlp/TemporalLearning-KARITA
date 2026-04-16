# TemporalLearning-KARITA


**Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation**

KARITA is a framework for temporal domain adaptation in specialized multi-label text classification. It addresses the challenge of model performance degradation when models trained on historical data are applied to future data, due to temporal distribution shifts in specialized domains.

## Overview

KARITA introduces:

- **Multi-signal shift detection** combining uncertainty-based, feature-based, and ontology-based signals to identify samples affected by temporal drift
- **Shift-aware retrieval** of semantically similar historical samples to provide targeted support for shifted instances
- **Knowledge-driven augmentation** using external ontologies (MeSH, EuroVoc, CSO) and LLM-based synonym generation to bridge terminological gaps across time periods

The framework is evaluated across three domains: clinical notes (MIMIC-IV-Notes), legal documents (EUR-Lex), and computer science papers (ArXiv-CS).

## Code

Code and data preprocessing scripts will be released upon paper acceptance. Stay tuned!

## Citation

Coming soon.
