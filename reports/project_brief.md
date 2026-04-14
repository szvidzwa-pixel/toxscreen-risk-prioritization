# Project Brief

## Executive summary

ToxScreen Risk Prioritization is a machine learning prototype designed to support early-stage compound safety screening. The product predicts whether a candidate molecule is likely to be toxic or non-toxic and translates those predictions into a practical triage recommendation for research teams.

The project story is intentionally narrow and clear:

**Use molecular structure data to predict clinical toxicity risk early, while treating false negatives as the most important failure mode.**

## Business problem

In drug discovery, late-stage toxicity failures are expensive and operationally painful. Screening teams need a faster way to surface risky compounds before they absorb more experimental effort, budget, and decision-making time.

From a decision-risk standpoint, the most concerning mistake is a false negative: marking a toxic compound as safe enough to move forward.

## Milestone 1 delivery

This repository delivers Milestone 1:

- dataset audit and class balance profiling
- molecular feature engineering using Morgan fingerprints
- baseline toxicity classification models
- business-facing evaluation metrics
- threshold analysis to support a safety-first operating policy

## Technical approach

- Input data: molecular SMILES strings with toxicity labels
- Target endpoint: clinical trial toxicity via `CT_TOX`
- Feature representation: Morgan fingerprints, radius 2, 2048 bits
- Models:
  - Logistic Regression
  - Random Forest
- Evaluation:
  - confusion matrix
  - precision
  - recall
  - F1
  - ROC-AUC
  - PR-AUC
- Decision policy:
  - threshold sweeping to quantify false negatives vs false positives

## Why this matters

The value of the system is not just predictive accuracy. The more important question is how to use model scores to reduce harmful misses. A compound incorrectly predicted as safe is a more serious mistake than flagging a safe compound for further review.

That is why the project emphasizes:

- class distribution analysis
- recall on the toxic class
- PR-AUC for imbalanced screening
- threshold tuning instead of relying blindly on the default `0.50` cutoff

## Future roadmap

- descriptor enrichment beyond fingerprints
- cross-validation and calibration
- model explainability for medicinal chemistry review
- deployment as an internal screening dashboard
