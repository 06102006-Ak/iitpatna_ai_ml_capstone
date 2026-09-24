# Submission Checklist

## Data Pipeline (25)
- [x] One repository root with `/data_pipeline`.
- [x] `requests` + `BeautifulSoup` live scraper for first five catalogue pages, aiming for 100 rows.
- [x] Cleaning for `price_gbp`, `rating`, `in_stock`, `price_inr`.
- [x] Fixed 1 GBP = 105.50 INR baseline.
- [x] Normalized SQLite `categories` / `books` PK-FK schema.
- [x] Six SQL queries with required clauses + JOIN.
- [x] Query outputs saved.
- [x] Two `pd.read_sql(...)` demonstrations + `pd.merge(...)` JOIN equivalence.
- [x] Data-quality report and robust retry/timeout handling.

## Analytics (50)
- [x] `sns.load_dataset("titanic")` appears once in the EDA load path.
- [x] Immediate `analytics/titanic.csv` fallback artifact.
- [x] `info`, `describe`, `shape`, missing percentages.
- [x] Threshold-based missing handling documented with exact percentages.
- [x] Age/fare histograms + box plots + IQR outlier counts.
- [x] Fare mean/median/mode + skewness interpretation.
- [x] Survival rates by sex, class, and sex+class.
- [x] Exact six-column correlation matrix + heatmap + strongest absolute pairs.
- [x] Four distinct multivariate charts with text interpretation.
- [x] Age/fare before-after z-score sanity check.
- [x] Stratified split before preprocessing.
- [x] Train-only imputation/encoding/scaling via `ColumnTransformer`.
- [x] Logistic Regression, Decision Tree, Random Forest.
- [x] Decision Tree `plot_tree` with feature/class labels.
- [x] Confusion matrix + accuracy + precision + recall + F1 + ROC/AUC.
- [x] Baseline vs balanced class weight vs train-only SMOTE.
- [x] RF GridSearchCV over n_estimators/max_depth/max_features + `oob_score=True`.
- [x] Multivariate fare regression + MAE/RMSE/R2/Adjusted R2 + residual plot + heteroscedasticity test.
- [x] Separate classification/regression metric groups.
- [x] Transparent technical model-selection rule and final recommendation.
- [x] Full fitted pipeline saved with joblib and raw-input reload check.
- [x] Permutation importance extension.

## Support Assistant (25)
- [x] Exactly 8 provided policy documents.
- [x] Per-document chunking + `all-MiniLM-L6-v2` embeddings.
- [x] Persistent ChromaDB collection.
- [x] Role/context/task/format/length prompt + negative constraint + few-shot example.
- [x] LangGraph `StateGraph` with `TypedDict` and three named nodes.
- [x] Conditional intent routing.
- [x] Required `MOCK_LLM` branches and optional real-LLM path.
- [x] Pydantic response schema and up-to-two corrective retries for real-LLM invalid output.
- [x] FastAPI `/ask` and `/health`.
- [x] Dockerfile for local container execution.
- [x] Architecture walkthrough.

## Git
- [x] Create a feature branch.
- [x] Commit the feature branch at least twice.
- [x] Merge the feature branch back into `main`.
