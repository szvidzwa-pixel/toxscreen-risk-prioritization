# Project Brief

## Executive summary

ToxScreen Risk Prioritization is a machine learning prototype designed to support early-stage compound safety screening. The product predicts whether a candidate molecule is likely to be toxic or non-toxic and translates those predictions into a practical triage recommendation for research teams.

The project story is intentionally narrow and clear:

**Use molecular structure data to predict clinical toxicity risk early, while treating false negatives as the most important failure mode.**

## Business problem

In drug discovery, late-stage toxicity failures are expensive and operationally painful. Screening teams need a faster way to surface risky compounds before they absorb more experimental effort, budget, and decision-making time.

From a decision-risk standpoint, the most concerning mistake is a false negative: marking a toxic compound as safe enough to move forward.

This framing also matters ethically. The model is designed as a screening aid, not as a replacement for toxicology expertise, experimental validation, or regulatory review. Its value comes from supporting earlier prioritization, not from automating high-stakes clinical decisions.

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
- Preprocessing choice: no feature scaling was applied because the inputs are fixed-length binary fingerprint vectors rather than continuous variables on incompatible numeric scales
- Models:
  - Logistic Regression
  - Random Forest
- Model development process:
  - stratified train/test split
  - light hyperparameter tuning on the training split only
  - cross-validation on the training split to assess stability
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

## Feature engineering rationale

The project is intentionally feature-engineering driven because molecular strings are not directly machine-readable by classical models. SMILES strings encode molecular structure, and Morgan fingerprints convert that structure into a fixed-length binary representation based on local atomic neighborhoods.

Morgan fingerprints are appropriate here for three reasons:

- they are a standard and defensible cheminformatics baseline
- they preserve local structural patterns that may relate to toxicity risk
- they work well with interpretable and production-friendly classical models such as Logistic Regression and Random Forest

The output dimensionality is 2048 bits per valid molecule. This preserves structural signal while remaining computationally manageable for baseline supervised modeling.

## Operational threshold policy

The project does not assume that the default `0.50` threshold is automatically correct. Instead, it evaluates multiple thresholds and selects an operating point that preserves a minimum precision floor while improving recall. This makes threshold selection part of the business and safety decision rather than just a software default.

## Overfitting and model risk

Because ClinTox is a relatively small and imbalanced dataset, some train-test performance gap is expected. Cross-validation is included to provide a more stable estimate of model behavior and to reduce the risk of over-reading a single split result.

## Interpretation and limitations

Random Forest feature importance can be reported at the fingerprint-bit level, but those bit indices should not be over-interpreted as directly human-readable chemical descriptors. They are useful for model diagnostics, not for strong causal claims about toxicity mechanisms.

The project should also acknowledge several limitations:

- ClinTox is a relatively small benchmark dataset
- the toxic class is a minority class, which makes threshold choice especially important
- a small number of molecules may be dropped if RDKit cannot parse them successfully
- benchmark results do not guarantee real-world generalization across therapeutic areas or assay conditions
- the model is not a clinical decision system and should not be used as one

## Future roadmap

- descriptor enrichment beyond fingerprints
- cross-validation and calibration
- model explainability for medicinal chemistry review
- deployment as an internal screening dashboard
