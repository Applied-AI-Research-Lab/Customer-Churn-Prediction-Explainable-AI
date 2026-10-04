import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Datasets", "ecommerce_customer_churn_dataset.csv")
SPLIT_DIR = os.path.join(BASE_DIR, "outputs", "splits")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "Churned"

DROP_COLUMNS = ["City"]

CLIP_RANGES = {
    "Total_Purchases": (0, None),
    "Cart_Abandonment_Rate": (0, 100),
}


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=DROP_COLUMNS)

    for col, (lo, hi) in CLIP_RANGES.items():
        if lo is not None:
            df[col] = df[col].clip(lower=lo)
        if hi is not None:
            df[col] = df[col].clip(upper=hi)

    return df


def split_data(df, val_size=0.15, test_size=0.15, random_state=42):
    remaining = 1.0 - (val_size + test_size)

    train_df, temp_df = train_test_split(
        df, test_size=1 - remaining, stratify=df[TARGET], random_state=random_state
    )

    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_ratio, stratify=temp_df[TARGET], random_state=random_state
    )

    return train_df, val_df, test_df


def build_preprocessor(df):
    feature_df = df.drop(columns=[TARGET])
    numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="passthrough",
    )

    return preprocessor, numerical_cols, categorical_cols


def main():
    df = load_and_clean_data()
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Dropped:      {DROP_COLUMNS}")
    print(f"Cleaned (clip): {list(CLIP_RANGES.keys())}")

    train_df, val_df, test_df = split_data(df)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(SPLIT_DIR, f"{name}.csv")
        split_df.to_csv(path, index=False)
        churn_rate = split_df[TARGET].mean() * 100
        print(f"  {name:>5}: {len(split_df):>6} rows ({churn_rate:.1f}% churn) → {path}")

    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    X_train = train_df.drop(columns=[TARGET])
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))

    print(f"\nNumerical features   ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print("\n✅ Preprocessing complete. Splits and preprocessor saved.")


if __name__ == "__main__":
    main()
