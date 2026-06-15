import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/raw.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)

X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

