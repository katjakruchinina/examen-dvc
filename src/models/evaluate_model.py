import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DATA_DIR = Path("data/processed")
PRED_DIR = Path("data/predictions")
METRICS_DIR = Path("metrics")
MODEL_PATH = Path("models/model.pkl")

PRED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

X_test = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

model = joblib.load(MODEL_PATH)

predictions = model.predict(X_test)

pred_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": predictions
})
pred_df.to_csv(PRED_DIR / "predictions.csv", index=False)

scores = {
    "mse": mean_squared_error(y_test, predictions),
    "mae": mean_absolute_error(y_test, predictions),
    "r2": r2_score(y_test, predictions),
}

with open(METRICS_DIR / "scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print(scores)
