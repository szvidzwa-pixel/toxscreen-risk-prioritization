# ToxScreen Risk Prioritization: Early Prediction of Clinical Toxicity from Molecular Structure

## Introduction

Drug development is expensive, time-intensive, and operationally risky. One of the most costly outcomes in the development pipeline is advancing a compound that later fails because of toxicity. This project was designed to address that early-screening problem by building a supervised machine learning pipeline that predicts whether a drug candidate compound is toxic or non-toxic using publicly available molecular data from ClinTox.

The project objective is to support early-stage prioritization, not to replace toxicology expertise, laboratory testing, or clinical review. In this context, the model is best understood as a screening support tool. The core decision problem is straightforward: given a molecular structure, can a machine learning model identify compounds that are more likely to pose toxicity risk early enough to support safer triage decisions?

This framing also creates an important ethical constraint. In a toxicity setting, the most dangerous mistake is a false negative, where a toxic compound is predicted as safe. For that reason, the project does not treat classification as a generic accuracy problem. Instead, it emphasizes class imbalance, recall for the toxic class, and threshold selection that reduces dangerous misses.

## Project Background and Project Scope

This project focuses on a reproducible baseline toxicity-screening workflow that can be handed off through GitHub and executed from the command line. The scope includes:

- dataset audit and exploratory data analysis
- molecular feature engineering using RDKit
- a baseline Logistic Regression model
- a comparative tree-based Random Forest model
- model evaluation with metrics appropriate for imbalanced classification
- threshold experimentation for cost-sensitive decision support

The project deliberately stays within the scope of classical supervised learning. This matches the problem requirements and keeps the work focused on molecular representation, careful evaluation, and defensible interpretation rather than unnecessary model complexity.

## Dataset and Data Understanding

The dataset used in this project is ClinTox, a benchmark molecular toxicity dataset that includes SMILES strings and multiple labels. For this project, the target endpoint is `CT_TOX`, which represents clinical trial toxicity. The target variable is defined as:

- `1` = toxic
- `0` = non-toxic

The raw dataset contains 1484 rows and three columns:

- `smiles`
- `FDA_APPROVED`
- `CT_TOX`

The initial data audit found no missing values in the key fields used for modeling. Specifically, the dataset contained 0 missing values in `smiles`, `FDA_APPROVED`, and `CT_TOX`, and no duplicate SMILES strings were identified in the raw file.

Class balance is a central issue in this dataset. Of the 1484 rows:

- 1372 compounds were labeled non-toxic
- 112 compounds were labeled toxic

This means that only about 7.55% of the dataset belongs to the toxic class, while 92.45% belongs to the non-toxic class. That imbalance matters because a model could appear strong under accuracy while still failing to identify toxic compounds consistently. For this reason, the project emphasizes recall, PR-AUC, confusion matrices, and threshold analysis rather than relying on accuracy.

### Visualization 1: ClinTox Target Distribution

![ClinTox target distribution](../docs/figures/ct_tox_class_distribution.png)

This chart shows the imbalance in the target label. The large gap between toxic and non-toxic counts explains why the project cannot be evaluated using accuracy alone. The chart supports the modeling decision to emphasize recall for the toxic class and to examine threshold tradeoffs carefully.

### Visualization 2: FDA Approval Label Distribution

![FDA approval distribution](../docs/figures/fda_approved_distribution.png)

This chart shows the distribution of the second label in the dataset, `FDA_APPROVED`. Although this label was not used as the target in the current project, it confirms that ClinTox includes multiple endpoints and supports the decision to clearly isolate `CT_TOX` as the modeled endpoint for this milestone.

### Visualization 3: SMILES Length Distribution

![SMILES length distribution](../docs/figures/smiles_length_distribution.png)

This plot gives a simple structural proxy for the dataset by showing the distribution of SMILES string lengths. It helps identify whether the raw molecular strings vary substantially in structural complexity and provides an additional quality check before feature extraction.

## Feature Engineering

Feature engineering is the most technically important part of this project because molecular data is not directly machine-ready for standard machine learning models. The raw molecular representation is a SMILES string, which is a text-based encoding of molecular structure. While informative, a SMILES string cannot be used directly by models such as Logistic Regression or Random Forest without transformation.

This project uses RDKit to convert SMILES strings into Morgan fingerprints. Morgan fingerprints, also commonly associated with ECFP-style representations, encode local structural neighborhoods within a molecule into a fixed-length binary vector. In this project, the fingerprint settings are:

- fingerprint type: Morgan fingerprint
- radius: 2
- bit size: 2048
- output dimensionality: 2048 binary features per valid molecule

These settings were selected because Morgan fingerprints are a widely accepted baseline in cheminformatics and work well with classical supervised learning methods. They preserve local structural patterns that may correlate with toxicity while remaining computationally manageable.

The feature extraction process is:

1. parse each SMILES string using RDKit
2. generate a Morgan fingerprint bit vector
3. convert the fingerprint into a NumPy array
4. collect valid molecular rows into the final feature matrix

During featurization, four molecules could not be parsed successfully by RDKit and were excluded from model training. As a result, the final feature matrix used for modeling contained 1480 valid molecules rather than the full 1484 rows. This was documented as a project limitation rather than hidden.

No feature scaling was applied. This was a deliberate preprocessing decision. The Morgan fingerprints are fixed-length binary indicator vectors, not continuous-valued features on incompatible numeric scales, so standard scaling would not provide meaningful benefit in the same way it might for traditional numeric tabular data.

## Model Development

Two supervised classification models were implemented:

### Baseline Model: Logistic Regression

Logistic Regression was used as the interpretable baseline model. This choice is appropriate because it is fast, well-understood, and provides a strong baseline for high-dimensional sparse binary features.

### Comparative Model: Random Forest

Random Forest was selected as the comparative tree-based model. It provides a nonlinear alternative that can capture more complex interactions across molecular fingerprint bits while remaining more interpretable than a deep learning architecture.

### Data Splitting and Validation Strategy

The modeling pipeline uses a stratified train/test split to preserve class proportions in both training and test data. This helps ensure that the rare toxic class remains represented in evaluation.

To strengthen the project beyond a single holdout split, 5-fold stratified cross-validation was added on the training data. This provides a more stable view of model performance and reduces the risk of over-interpreting one particular split.

### Hyperparameter Tuning

Light hyperparameter tuning was applied on the training split only, which avoids data leakage into the held-out test set.

The tuned settings selected by cross-validated average precision were:

- Logistic Regression: `C = 0.25`
- Random Forest: `n_estimators = 200`, `max_depth = None`, `min_samples_leaf = 2`

This level of tuning is intentionally modest. The project goal is not to over-engineer the model but to build a defensible, reproducible supervised learning baseline that is strong enough to support evaluation and threshold experimentation.

## Model Evaluation

Because the dataset is imbalanced and the business risk is asymmetric, the project reports:

- confusion matrix
- precision
- recall
- F1 score
- ROC-AUC
- PR-AUC

The tuned holdout results were:

### Logistic Regression

- Precision: 0.361
- Recall: 0.591
- F1: 0.448
- ROC-AUC: 0.882
- PR-AUC: 0.474

### Random Forest

- Precision: 0.333
- Recall: 0.182
- F1: 0.235
- ROC-AUC: 0.838
- PR-AUC: 0.341

The Logistic Regression model outperformed the Random Forest model on the metrics that matter most for this project, especially recall and PR-AUC. Even though Random Forest remained a useful comparison model, it identified substantially fewer toxic compounds on the held-out set. That difference is easier to understand in concrete terms than in metric labels alone. A majority-class model would catch 0 of 22 toxic compounds on the held-out test set. Logistic Regression catches 13 of 22 at the default threshold, and after threshold tuning it catches 18 of 22. That does not eliminate risk, but it does show the model is doing more than simply following the dominant non-toxic class.

### Visualization 4: Logistic Regression Confusion Matrix

![Logistic Regression confusion matrix](../docs/figures/confusion_matrix_logistic_regression.png)

This confusion matrix shows how the tuned Logistic Regression model performed on the held-out test set at the default threshold. It makes the toxic-class detection problem easier to interpret by separating true negatives, false positives, false negatives, and true positives directly rather than leaving the analysis at the metric-table level.

### Visualization 5: Random Forest Confusion Matrix

![Random Forest confusion matrix](../docs/figures/confusion_matrix_random_forest.png)

This confusion matrix shows the weaker toxic-class detection of the Random Forest model. Relative to Logistic Regression, the Random Forest model produced fewer true positives for the toxic class, which is why it was not the preferred safety-first model in this project.

### Cross-Validation Interpretation

For the tuned Logistic Regression model, mean cross-validation performance on the training data was:

- mean recall: 0.389 +/- 0.035
- mean PR-AUC: 0.378 +/- 0.094
- mean ROC-AUC: 0.791 +/- 0.040

These values are lower than the single held-out test results, which is useful information rather than a problem. It suggests the model has some sensitivity to data partitioning, which is expected on a relatively small and imbalanced molecular benchmark.

### Feature Importance Interpretation

Random Forest feature importance was exported at the fingerprint-bit level. This can be useful for model diagnostics, but it should be interpreted cautiously. The ranked outputs correspond to hashed fingerprint bits, not directly human-readable chemical descriptors. For that reason, these importance values should not be treated as strong mechanistic evidence about toxicity. They are most appropriate as supporting information about model behavior rather than causal explanation.

## Cost-Sensitive Threshold Adjustment

This section is one of the most important parts of the project because it directly addresses the safety question raised by the professor’s instructions.

The default classification threshold of 0.50 is not automatically the best threshold for toxicity screening. In this context, the central cost-sensitive concern is reducing false negatives. A false negative means a toxic compound is predicted as safe, which is more dangerous than a false positive that simply sends a non-toxic compound for additional review.

Threshold experimentation was performed on the Logistic Regression model. Several thresholds were tested between 0.10 and 0.90, and the resulting false positives, false negatives, recall, precision, and F1 values were compared.

At the default threshold of 0.50:

- recall = 0.591
- false negatives = 9
- false positives = 23

At the recommended operating threshold of 0.30:

- recall = 0.818
- false negatives = 4
- false positives = 43

This means the lower threshold reduced dangerous misses from 9 toxic compounds to 4 toxic compounds, but at the cost of increasing false positives. That is the intended tradeoff in a safety-first screening system.

### Visualization 6: Threshold Tradeoff Plot

![Threshold tradeoffs](../docs/figures/threshold_tradeoffs.png)

This chart shows how threshold selection changes precision, recall, false negative rate, and false positive rate. It provides the quantitative basis for the operational threshold decision. In this project, a threshold of 0.30 was selected because it preserved a minimum precision floor while substantially increasing recall, making it a more appropriate screening threshold than the default 0.50 cutoff.

## Challenges Faced and Solutions

Several project challenges emerged during implementation:

### 1. Molecular strings are not ML-ready

Raw SMILES strings cannot be passed directly into standard classical machine learning models.

**Solution:** RDKit-based Morgan fingerprints were used to convert each molecule into a fixed-length binary feature vector.

### 2. Strong class imbalance

Only 7.55% of the dataset belongs to the toxic class, which creates a serious risk of misleading evaluation if accuracy is over-emphasized.

**Solution:** The evaluation framework prioritized recall, PR-AUC, confusion matrices, and threshold analysis rather than accuracy.

### 3. Invalid molecular parsing

A small number of molecules could not be parsed successfully by RDKit.

**Solution:** Invalid molecules were dropped during featurization and the difference between raw rows and valid modeled molecules was documented explicitly.

### 4. Risk of overfitting

Because ClinTox is small and imbalanced, it is easy for a model to look strong on one split and still generalize poorly.

**Solution:** Cross-validation was added, light tuning was restricted to the training split, and the train-test performance gap was discussed honestly.

### 5. Strong metrics can still mislead on imbalanced data

At first, the ROC-AUC results looked stronger to me than the minority-class behavior really was. That was an important course correction in the project.

**Solution:** ROC-AUC was kept for completeness, but recall, PR-AUC, confusion matrices, and threshold tradeoffs were treated as the primary evaluation tools because they say more about toxic-compound detection.

### 6. Interpreting tree-based feature importance

Random Forest can output feature importance scores, but fingerprint-bit importance is not naturally human-readable.

**Solution:** Feature importance was reported cautiously as model-diagnostic evidence rather than over-claimed as direct chemical interpretation.

## Business and Ethical Interpretation

From a business perspective, this project shows how machine learning can support earlier compound triage by identifying risky molecules before they absorb more downstream time and resources. The value of the project is not that it eliminates scientific uncertainty, but that it helps prioritize review earlier in the pipeline.

From an ethical perspective, the project avoids claiming that machine learning should replace toxicology expertise or experimental evidence. Instead, it frames the model as one input into a broader decision process. This is especially important in toxicity prediction, where a false negative can have more serious consequences than a false positive.

The operational recommendation in this project therefore favors safer screening over narrower precision. That is a deliberate decision aligned with the real-world asymmetry of toxicity risk.

## Limitations

This project has several important limitations:

- ClinTox is a relatively small benchmark dataset
- the toxic class is heavily underrepresented
- four molecules were dropped because RDKit could not featurize them successfully
- fingerprint-bit importance is limited in interpretability
- benchmark performance does not guarantee generalization to real-world development settings
- the model is not a clinical decision system and should not be used as one

These limitations do not invalidate the project, but they should shape how the results are interpreted. The system should be understood as a reproducible baseline screening workflow rather than a deployment-ready toxicity platform.

## Future Work

Several next steps would strengthen the project:

- include richer molecular descriptors beyond Morgan fingerprints
- evaluate probability calibration
- validate on an additional external toxicity dataset
- explore explainability tools beyond raw fingerprint-bit importance
- package the workflow into an internal screening dashboard
- test class weighting, SMOTE, and XGBoost with `scale_pos_weight` on the training split only
- keep the held-out test set naturally imbalanced so evaluation still reflects the real screening environment

These additions would improve both practical usability and interpretability while preserving the project’s core objective of early toxicity prioritization.

## Conclusion

This project developed a supervised binary classification pipeline for clinical toxicity prediction using molecular structure data from ClinTox. The work emphasized feature engineering, careful evaluation, and threshold-based decision support rather than unnecessary model complexity.

The tuned Logistic Regression model emerged as the strongest baseline, outperforming the Random Forest comparison model on the most important safety-oriented metrics. Most importantly, threshold tuning demonstrated that the model could reduce false negatives substantially, which is the central risk-management goal of this screening problem.

Overall, the project supports the conclusion that molecular structure-based supervised learning can serve as a useful early screening support tool when it is interpreted carefully, evaluated honestly, and used within a broader scientific decision framework.
