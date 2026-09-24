from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)

target = "survived"

classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

X = df[classification_features].copy()
y = df[target].copy()

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("\nTRAIN/TEST SPLIT")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Train class counts: {y_train.value_counts().to_dict()}")
print(f"Test class counts: {y_test.value_counts().to_dict()}")
print(
    "Stratification preserves the observed survived/not-survived class proportions "
    "in both training and test sets."
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=5,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

classifier_results = []
fitted_models = {}

plt.figure(figsize=(9, 7))

for model_name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    classifier_results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc,
        }
    )

    fitted_models[model_name] = pipeline

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc:.4f})",
    )

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
        dpi=150,
    )
    plt.close()

    print(f"\n{model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

plt.figure(figsize=(9, 7))
for model_name, pipeline in fitted_models.items():
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc:.4f})",
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve_comparison.png", dpi=150)
plt.close()

classifier_df = pd.DataFrame(classifier_results)

print("\nCLASSIFIER COMPARISON")
print(classifier_df.to_string(index=False))

tree_pipeline = fitted_models["Decision Tree"]
tree_preprocessor = tree_pipeline.named_steps["preprocessor"]
tree_model = tree_pipeline.named_steps["model"]

feature_names = tree_preprocessor.get_feature_names_out()

plt.figure(figsize=(24, 14))
plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=7,
)
plt.title("Decision Tree")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "decision_tree.png", dpi=150)
plt.close()

print("\nCLASS IMBALANCE")
class_balance = y.value_counts().sort_index()
class_balance_percent = y.value_counts(normalize=True).sort_index() * 100

print(
    pd.DataFrame(
        {
            "Count": class_balance,
            "Percentage": class_balance_percent,
        }
    )
)

imbalance_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

X_train_processed = imbalance_preprocessor.fit_transform(X_train)
X_test_processed = imbalance_preprocessor.transform(X_test)

baseline_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

balanced_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

baseline_model.fit(X_train_processed, y_train)
balanced_model.fit(X_train_processed, y_train)

baseline_pred = baseline_model.predict(X_test_processed)
balanced_pred = balanced_model.predict(X_test_processed)

smote = SMOTE(random_state=RANDOM_STATE)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train,
)

smote_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

smote_model.fit(X_train_smote, y_train_smote)

smote_pred = smote_model.predict(X_test_processed)

imbalance_results = []

for name, predictions in [
    ("Baseline", baseline_pred),
    ("Class Weight Balanced", balanced_pred),
    ("SMOTE", smote_pred),
]:
    imbalance_results.append(
        {
            "Strategy": name,
            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "F1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
        }
    )

imbalance_df = pd.DataFrame(imbalance_results)

print("\nIMBALANCE HANDLING COMPARISON")
print(imbalance_df.to_string(index=False))

best_imbalance_f1 = imbalance_df["F1"].max()
best_imbalance_rows = imbalance_df[
    np.isclose(imbalance_df["F1"], best_imbalance_f1)
]

best_strategy_names = best_imbalance_rows["Strategy"].tolist()

baseline_row = imbalance_df[
    imbalance_df["Strategy"] == "Baseline"
].iloc[0]

best_recall = best_imbalance_rows["Recall"].max()
best_precision = best_imbalance_rows["Precision"].max()

if len(best_strategy_names) == 1:
    imbalance_conclusion = (
        f"{best_strategy_names[0]} achieved the highest F1 score of "
        f"{best_imbalance_f1:.4f}. It was preferred because its balance "
        f"between precision and recall was strongest among the tested "
        f"strategies, while the baseline F1 was {baseline_row['F1']:.4f}."
    )
else:
    imbalance_conclusion = (
        f"{' and '.join(best_strategy_names)} achieved the highest F1 score "
        f"of {best_imbalance_f1:.4f}, compared with {baseline_row['F1']:.4f} "
        f"for the baseline. These strategies increased recall to "
        f"{best_recall:.4f} while maintaining precision at approximately "
        f"{best_precision:.4f}. Therefore, they produced the strongest "
        f"observed balance between precision and recall in this test-set "
        f"comparison."
    )

print("\nIMBALANCE CONCLUSION")
print(imbalance_conclusion)

param_grid = {
    "model__n_estimators": [200, 300, 350],
    "model__max_depth": [None, 4, 6, 8],
    "model__max_features": ["sqrt", "log2"],
}

grid_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                oob_score=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

grid_search = GridSearchCV(
    estimator=grid_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_cv_f1 = grid_search.best_score_

tuned_model = grid_search.best_estimator_
tuned_rf = tuned_model.named_steps["model"]
oob_score = tuned_rf.oob_score_

tuned_pred = tuned_model.predict(X_test)
tuned_prob = tuned_model.predict_proba(X_test)[:, 1]

tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_precision = precision_score(
    y_test,
    tuned_pred,
    zero_division=0,
)
tuned_recall = recall_score(
    y_test,
    tuned_pred,
    zero_division=0,
)
tuned_f1 = f1_score(
    y_test,
    tuned_pred,
    zero_division=0,
)
tuned_auc = roc_auc_score(y_test, tuned_prob)

tuned_result = {
    "Model": "Tuned Random Forest",
    "Accuracy": tuned_accuracy,
    "Precision": tuned_precision,
    "Recall": tuned_recall,
    "F1": tuned_f1,
    "AUC": tuned_auc,
}

classifier_df = pd.concat(
    [
        classifier_df,
        pd.DataFrame([tuned_result]),
    ],
    ignore_index=True,
)

print("\nGRID SEARCH")
print(f"Best parameters: {best_params}")
print(f"Best CV F1: {best_cv_f1:.4f}")
print(f"OOB score: {oob_score:.4f}")

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked",
]

regression_target = "fare"

regression_df = df[
    regression_features + [regression_target]
].copy()

regression_df = regression_df.dropna(
    subset=[regression_target]
)

X_reg = regression_df[regression_features]
y_reg = regression_df[regression_target]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

regression_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
]

regression_categorical_features = [
    "sex",
    "embarked",
]

regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            regression_numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            regression_categorical_features,
        ),
    ]
)

regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", regression_preprocessor),
        ("model", LinearRegression()),
    ]
)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train,
)

y_reg_pred = regression_pipeline.predict(X_reg_test)

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred,
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred,
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred,
)

n = len(y_reg_test)
p = regression_pipeline.named_steps[
    "preprocessor"
].transform(X_reg_test).shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

residuals = y_reg_test - y_reg_pred

X_bp = sm.add_constant(y_reg_pred)

bp_model = sm.OLS(
    residuals,
    X_bp,
).fit()

bp_test = het_breuschpagan(
    bp_model.resid,
    bp_model.model.exog,
)

bp_lm_statistic = bp_test[0]
bp_lm_pvalue = bp_test[1]

if bp_lm_pvalue < 0.05:
    heteroscedasticity_conclusion = (
        "The residual analysis provides evidence of heteroscedasticity "
        f"(Breusch-Pagan p-value={bp_lm_pvalue:.4f})."
    )
else:
    heteroscedasticity_conclusion = (
        "The residual analysis does not provide sufficient evidence of "
        f"heteroscedasticity (Breusch-Pagan p-value={bp_lm_pvalue:.4f})."
    )

print("\nREGRESSION RESULTS")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")
print(f"Breusch-Pagan LM statistic: {bp_lm_statistic:.4f}")
print(f"Breusch-Pagan p-value: {bp_lm_pvalue:.4f}")
print(heteroscedasticity_conclusion)

plt.figure(figsize=(9, 6))
plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.7,
)
plt.axhline(
    0,
    linestyle="--",
)
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Regression Residual Plot")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "regression_residual_plot.png",
    dpi=150,
)
plt.close()

comparison_rows = []

for _, row in classifier_df.iterrows():
    comparison_rows.append(
        {
            "Model": row["Model"],
            "Accuracy": row["Accuracy"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "F1": row["F1"],
            "AUC": row["AUC"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "R²": np.nan,
            "Adjusted R²": np.nan,
        }
    )

comparison_rows.append(
    {
        "Model": "Linear Regression",
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "AUC": np.nan,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Adjusted R²": adjusted_r2,
    }
)

final_comparison_df = pd.DataFrame(comparison_rows)

print("\nFINAL MODEL COMPARISON")
print(
    final_comparison_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

final_comparison_df.to_csv(
    OUTPUT_DIR / "final_model_comparison.csv",
    index=False,
)

classifier_df.to_csv(
    OUTPUT_DIR / "classifier_comparison.csv",
    index=False,
)

imbalance_df.to_csv(
    OUTPUT_DIR / "imbalance_comparison.csv",
    index=False,
)

best_model_row = classifier_df.sort_values(
    by=["F1", "AUC", "Accuracy"],
    ascending=False,
).iloc[0]

best_model_name = best_model_row["Model"]

if best_model_name == "Tuned Random Forest":
    deployment_pipeline = tuned_model
else:
    deployment_pipeline = fitted_models[best_model_name]

model_path = MODEL_DIR / "best_classifier_pipeline.joblib"

joblib.dump(
    deployment_pipeline,
    model_path,
)

reloaded_pipeline = joblib.load(model_path)

reload_predictions = reloaded_pipeline.predict(
    X_test.head(5)
)

print("\nFINAL DEPLOYMENT SELECTION")
print(
    f"Selection rule: highest held-out F1, followed by AUC and accuracy."
)
print(f"Selected model: {best_model_name}")
print(
    f"Accuracy: {best_model_row['Accuracy']:.4f}"
)
print(
    f"Precision: {best_model_row['Precision']:.4f}"
)
print(
    f"Recall: {best_model_row['Recall']:.4f}"
)
print(
    f"F1: {best_model_row['F1']:.4f}"
)
print(
    f"AUC: {best_model_row['AUC']:.4f}"
)

print(
    f"The selected classifier achieved an F1 score of "
    f"{best_model_row['F1']:.4f} and an AUC of "
    f"{best_model_row['AUC']:.4f} on the held-out test set."
)

print(
    f"Its accuracy was {best_model_row['Accuracy']:.4f}, with "
    f"precision of {best_model_row['Precision']:.4f} and recall of "
    f"{best_model_row['Recall']:.4f}."
)

print(
    "The selection prioritizes F1 because it balances precision and recall "
    "for the classification task, with AUC and accuracy used as secondary "
    "criteria."
)

print(
    "The saved artifact contains the preprocessing pipeline together with "
    "the final classifier, allowing raw input data to be passed directly "
    "to the deployed pipeline."
)

print(f"\nSaved pipeline: {model_path}")
print(f"Reload check predictions: {reload_predictions.tolist()}")

report_path = OUTPUT_DIR / "modeling_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("TITANIC PREDICTIVE MODELING REPORT\n")
    report.write("=" * 80 + "\n\n")

    report.write("1. TRAIN/TEST SPLIT\n")
    report.write(
        "A stratified 80/20 split was used so that the survived/not-survived "
        "class proportions observed in the cleaned dataset were preserved "
        "across the training and test sets.\n\n"
    )

    report.write("2. CLASSIFIER COMPARISON\n")
    report.write(
        classifier_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    report.write("\n\n")

    report.write("3. IMBALANCE HANDLING\n")
    report.write(
        imbalance_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    report.write("\n\n")
    report.write("Conclusion:\n")
    report.write(imbalance_conclusion)
    report.write("\n\n")

    report.write("4. HYPERPARAMETER TUNING\n")
    report.write(f"Best parameters: {best_params}\n")
    report.write(f"Best CV F1: {best_cv_f1:.4f}\n")
    report.write(f"OOB score: {oob_score:.4f}\n\n")

    report.write("5. REGRESSION\n")
    report.write(f"MAE: {mae:.4f}\n")
    report.write(f"RMSE: {rmse:.4f}\n")
    report.write(f"R²: {r2:.4f}\n")
    report.write(f"Adjusted R²: {adjusted_r2:.4f}\n")
    report.write(
        f"Breusch-Pagan p-value: {bp_lm_pvalue:.4f}\n"
    )
    report.write(
        heteroscedasticity_conclusion
    )
    report.write("\n\n")

    report.write("6. FINAL MODEL COMPARISON\n")
    report.write(
        final_comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    report.write("\n\n")

    report.write("7. FINAL RECOMMENDATION\n")
    report.write(
        f"The selected classifier is {best_model_name}, based on the "
        f"predefined selection rule of highest held-out F1, followed by "
        f"AUC and accuracy. It achieved an F1 score of "
        f"{best_model_row['F1']:.4f}, AUC of "
        f"{best_model_row['AUC']:.4f}, and accuracy of "
        f"{best_model_row['Accuracy']:.4f}. Its precision was "
        f"{best_model_row['Precision']:.4f} and recall was "
        f"{best_model_row['Recall']:.4f}. The complete preprocessing and "
        f"classification pipeline was saved and successfully reloaded for "
        f"prediction on raw test inputs.\n"
    )

print(f"\nReport saved: {report_path}")
print("\nModeling completed successfully.")