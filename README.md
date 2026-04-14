# ToxScreen Risk Prioritization

ToxScreen is a machine learning project for early-stage compound safety screening. It predicts whether a small-molecule drug candidate is likely to be toxic or non-toxic using the ClinTox benchmark and turns model outputs into an operational triage recommendation.

## Core story

This repository is built around one simple story:

**We use molecular structure data to predict clinical toxicity risk early, with special attention to false negatives.**

That story drives the technical choices in the project:

- molecules are represented from SMILES strings and transformed into Morgan fingerprints
- the target is clinical trial toxicity risk via `CT_TOX`
- evaluation emphasizes recall, PR-AUC, confusion matrices, and threshold tradeoffs
- threshold selection is designed to reduce dangerous misses, where a toxic compound is predicted as safe

Instead of framing this like a classroom submission, the repository is structured like a lightweight data science product handoff:

- reproducible environment
- clean command-line workflow
- config-driven training pipeline
- business-facing metrics and plots
- threshold analysis for safety-first decision making
- documentation a non-technical reviewer can follow

## Why this project matters

Drug development is expensive, slow, and risky. A toxicity screening model will never replace lab validation or clinical judgment, but it can help research teams prioritize compounds earlier and reduce the chance of pushing high-risk candidates deeper into the pipeline.

This project focuses on one business question:

**Can we build a binary classifier that helps flag potentially toxic compounds early enough to support safer portfolio triage?**

## What the project does

The pipeline:

1. Loads ClinTox data from a local CSV
2. Validates required fields and profiles class balance
3. Converts SMILES strings into Morgan fingerprints with RDKit
4. Trains two baseline production-friendly models:
   - Logistic Regression
   - Random Forest
5. Evaluates model quality with metrics suited for imbalanced safety problems
6. Sweeps decision thresholds to quantify the false-negative vs false-positive tradeoff
7. Recommends a safety-oriented operating threshold

## Project highlights

- Binary target: `CT_TOX`
- Modeling objective: predict clinical toxicity risk from molecular structure
- Molecular representation: Morgan fingerprints
- Fingerprint settings: radius `2`, bits `2048`
- Baseline model: Logistic Regression
- Comparison model: Random Forest
- Key metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
- Risk lens: false negatives are treated as the most dangerous error type
- Product-style artifact output: JSON metrics, CSV threshold tables, PNG visualizations, and a written results summary

## Repository layout

```text
toxscreen-risk-prioritization/
├── configs/
│   └── defaults.json
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   └── 01_dataset_audit.ipynb
├── outputs/
│   ├── figures/
│   └── metrics/
├── reports/
│   └── project_brief.md
├── src/
│   └── toxscreen/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── features.py
│       ├── modeling.py
│       └── pipeline.py
├── tests/
│   ├── test_config.py
│   └── test_threshold_policy.py
├── .gitignore
├── Makefile
├── main.py
└── requirements.txt
```

## Quickstart

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd toxscreen-risk-prioritization
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Place the ClinTox CSV here:

```text
data/raw/clintox.csv
```

Expected columns:

- `smiles`
- `CT_TOX`

The loader normalizes capitalization automatically.

### 5. Audit the dataset

```bash
python main.py audit --data data/raw/clintox.csv
```

This generates documented EDA artifacts, including:

- `outputs/metrics/eda_summary.md`
- `outputs/metrics/ct_tox_counts.csv`
- `outputs/metrics/ct_tox_normalized.csv`
- `outputs/metrics/fda_counts.csv`
- `outputs/metrics/fda_normalized.csv`
- `outputs/figures/ct_tox_class_distribution.png`
- `outputs/figures/fda_approved_distribution.png`
- `outputs/figures/smiles_length_distribution.png`

### 6. Train the models

```bash
python main.py train --data data/raw/clintox.csv --config configs/defaults.json
```

### 7. Run tests

```bash
pytest
```

## Example outputs

After a successful run, the project writes:

- `outputs/metrics/dataset_profile.json`
- `outputs/metrics/eda_summary.md`
- `outputs/metrics/model_metrics.json`
- `outputs/metrics/model_comparison.csv`
- `outputs/metrics/threshold_analysis.csv`
- `outputs/metrics/results_summary.md`
- `outputs/figures/confusion_matrix_logistic_regression.png`
- `outputs/figures/confusion_matrix_random_forest.png`
- `outputs/figures/threshold_tradeoffs.png`

The most important output is the written summary, because it makes the project story obvious:

- how many compounds were modeled
- whether the toxic class is the minority class
- which model performed best
- how threshold changes affect false negatives
- what operating threshold is recommended for a safety-first workflow

The EDA summary is important before modeling because it documents the class imbalance explicitly. For example, the audit workflow reproduces checks such as:

```python
df["CT_TOX"].value_counts(normalize=True)
```

## Design choices

### Why Morgan fingerprints

SMILES strings are not directly usable by most classical ML models. Morgan fingerprints convert molecular structure into a fixed-length binary vector that captures local structural neighborhoods and is widely used in cheminformatics.

### Why Logistic Regression

It gives an interpretable, credible baseline and is fast to train on sparse binary features.

### Why Random Forest

It offers a stronger nonlinear comparison model while remaining easier to explain than a deep learning architecture.

### Why threshold analysis is central

In toxicity screening, a false negative means a toxic compound is predicted to be safe. That kind of mistake is more dangerous than a false positive. This repository treats threshold selection as part of the product, not an afterthought.

## How to read the results

When reviewing project outputs, use this order:

1. `dataset_profile.json` to understand class balance and data quality
2. `model_comparison.csv` to compare baseline model performance
3. `threshold_analysis.csv` and `threshold_tradeoffs.png` to inspect the false-negative tradeoff
4. `results_summary.md` for the final plain-English interpretation

If a reviewer only reads one artifact after the README, it should be `outputs/metrics/results_summary.md`.

## Recruiter-facing framing

This repository demonstrates:

- applied binary classification on real molecular data
- feature engineering with domain-aware representations
- reproducible packaging and command-line usability
- metric selection for imbalanced risk problems
- translating model scores into business operating decisions
