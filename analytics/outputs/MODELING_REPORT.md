# Modeling Report

The stratified split used random state **42**, with 712 training rows and 179 test rows. Stratification preserves the target class proportion across both partitions, reducing the risk that a random split creates a materially different survived/not-survived mix.

## Class balance

Train class counts: `{0: 439, 1: 273}`. Test class counts: `{0: 110, 1: 69}`.

## Classifier comparison

| name                |   accuracy |   precision |   recall |     f1 |    auc |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8045 |      0.7931 |   0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |     0.8045 |      0.8400 |   0.6087 | 0.7059 | 0.8067 |
| Random Forest       |     0.8101 |      0.7966 |   0.6812 | 0.7344 | 0.8287 |
| Tuned Random Forest |     0.8156 |      0.8750 |   0.6087 | 0.7179 | 0.8475 |

Confusion matrices and ROC curves are in `classifier_evaluation.png`; the Decision Tree visualization is `decision_tree.png` with transformed feature names and class labels.

## Imbalance handling

| name                  |   precision |   recall |     f1 |
|:----------------------|------------:|---------:|-------:|
| baseline              |      0.7966 |   0.6812 | 0.7344 |
| class_weight_balanced |      0.7500 |   0.7391 | 0.7445 |
| SMOTE                 |      0.7500 |   0.7391 | 0.7445 |

The three variants use the same split. SMOTE is embedded inside an `imblearn` pipeline after preprocessing, which guarantees that synthetic samples are generated only from the training partition. The observed comparison shows how class weighting and oversampling shift precision/recall/F1 rather than assuming that one method must always be superior.

## Random Forest hyperparameter tuning

Best parameters: `{'model__max_depth': 6, 'model__max_features': 'sqrt', 'model__n_estimators': 350}`. Best cross-validation F1: **0.7378**. OOB score of the refit `RandomForestClassifier(oob_score=True, ...)`: **0.8258**.

## Regression side-task

| Metric | Value |
|---|---:|
| MAE | 20.8977 |
| RMSE | 30.5328 |
| R² | 0.3975 |
| Adjusted R² | 0.3617 |
| Breusch–Pagan F-test p-value | 0.0000 |

The residual plot is saved as `regression_residuals.png`. Based on the Breusch–Pagan check and the visual residual spread, this run shows **evidence of heteroscedasticity**; the conclusion is specifically about this fitted sample and not a universal claim about fares.

## Separate metric groups

Classification metrics (accuracy, precision, recall, F1, AUC) and regression metrics (MAE, RMSE, R², adjusted R²) are deliberately reported in separate sections because they measure different tasks and are not comparable on one common numerical scale.

## Final deployment selection

Using the pre-declared rule of highest held-out F1, then AUC, then accuracy, the selected classifier is **Random Forest** with accuracy **0.8101**, precision **0.7966**, recall **0.6812**, F1 **0.7344**, and AUC **0.8287**. This selection prioritizes balanced positive-class performance while retaining ranking quality through AUC. The chosen pipeline includes imputation, one-hot encoding, scaling, and the classifier in one fitted object, so deployment can accept raw input rows without manual preprocessing. The model artifact is stored at `C:/Users/akshi/OneDrive/Desktop/zepto_ai_ml_capstone/analytics/models/best_classifier_pipeline.joblib` and is verified by `reload_check.py`.

## Additional engineering evidence

`feature_importance.csv` is produced from permutation importance on the held-out test set for the deployed classifier where supported. This provides a model-agnostic explanation layer without changing the grading pipeline.
