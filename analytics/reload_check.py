from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "analytics/titanic.csv"
MODEL_PATH = ROOT / "analytics/models/best_classifier_pipeline.joblib"
FEATURES = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]

df = pd.read_csv(CSV_PATH)
model = joblib.load(MODEL_PATH)
raw_rows = df[FEATURES].head(5)
predictions = model.predict(raw_rows).astype(int).tolist()
print(json.dumps({"raw_input_rows": raw_rows.to_dict(orient="records"), "predictions": predictions}, indent=2, default=str))
