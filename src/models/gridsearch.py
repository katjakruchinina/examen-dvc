import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()

model = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
)

grid.fit(X_train, y_train)

joblib.dump(grid.best_params_, MODEL_DIR / "best_params.pkl")

print("Best parameters:", grid.best_params_)
