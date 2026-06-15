import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()

best_params = joblib.load(MODEL_DIR / "best_params.pkl")

model = RandomForestRegressor(
    random_state=42,
    **best_params
)

model.fit(X_train, y_train)

joblib.dump(model, MODEL_DIR / "model.pkl")
