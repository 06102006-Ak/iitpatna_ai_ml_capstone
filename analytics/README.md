# Module 2 — Analytics Pipeline

## Execution order
Run `01_eda.py` first. It calls `sns.load_dataset('titanic')` exactly once, immediately saves the returned DataFrame to `analytics/titanic.csv`, then performs profiling, cleaning, EDA, and the standardization sanity check. `02_modeling.py` reads that committed CSV and never calls `sns.load_dataset(...)` again.

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

If the Seaborn repository cannot be reached during development, `01_eda.py` has an explicit offline-only fallback that reconstructs the Seaborn-compatible Titanic schema from a locally installed copy of the canonical Kaggle Titanic table. The fallback is documented and does not replace the required `sns.load_dataset('titanic')` code path when internet/cache access is available.

## Key design decisions
- Missingness handling follows the rubric threshold literally: `<5%` → drop rows; `5–30%` → impute; very high missingness → drop the column when it is not needed downstream.
- The classification pipeline uses an identical stratified train/test split for Logistic Regression, Decision Tree, and Random Forest. `ColumnTransformer` enforces train-only fitting for imputation, one-hot encoding, and scaling.
- SMOTE is inside an `imblearn.pipeline.Pipeline`, so oversampling occurs only after the training-fold preprocessing and never touches the test fold.
- Random Forest tuning is performed with `GridSearchCV`, with `oob_score=True` enabled on the estimator so the fitted best estimator exposes an OOB score.
- The selected final classifier is the model with the highest held-out F1, breaking ties by AUC, then accuracy. This is a transparent technical selection rule, not an arbitrary choice.
- The complete fitted preprocessing + estimator pipeline is saved with `joblib.dump(...)` and reloaded against raw rows in `reload_check.py`.

## Outputs
`outputs/EDA_REPORT.md` contains the measured missing percentages, IQR outlier counts, survival breakdowns, correlation analysis, standardization check, and chart interpretations. `outputs/MODELING_REPORT.md` contains all classifier metrics, imbalance comparison, GridSearchCV result, regression metrics, heteroscedasticity analysis, feature importance, and final model-selection rationale.
