"""
Data preprocessing for E-commerce Customer Churn prediction.

Steps:
  1. Drop non-predictive / problematic columns
  2. Clean inappropriate values (clip out-of-range numerics)
  3. Split data into train (70%), validation (15%), test (15%) — stratified
  4. Build a reusable sklearn preprocessing pipeline (impute → encode → scale)
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Datasets", "ecommerce_customer_churn_dataset.csv")
SPLIT_DIR = os.path.join(BASE_DIR, "outputs", "splits")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "Churned"

# ── Column decisions (based on statistical analysis) ────────────────────────
# Columns to DROP and why:
#   - City: 40 unique values → high cardinality, redundant with Country,
#           churn rate uniform across cities (25-32%) → no predictive value
DROP_COLUMNS = ["City"]

# Columns with INAPPROPRIATE VALUES to clean (clip):
#   - Total_Purchases: negative values (min=-13) → impossible for a count → clip ≥ 0
#   - Cart_Abandonment_Rate: values > 100% (max=143.7%) → impossible for a rate → clip [0, 100]
CLIP_RANGES = {
    "Total_Purchases": (0, None),
    "Cart_Abandonment_Rate": (0, 100),
}

# Weak predictors (kept but noted — negligible churn difference):
#   - Membership_Years   (mean diff churned vs retained = -0.003)
#   - Payment_Method_Diversity (diff = +0.011)
#   - Gender             (churn rate ~29% across all categories)
# These are retained because tree models handle weak features automatically
# and there may be non-linear / interaction effects.


def load_and_clean_data():
    """Load dataset, drop non-predictive columns, fix inappropriate values."""
    df = pd.read_csv(DATA_PATH)

    # Drop high-cardinality / non-predictive columns
    df = df.drop(columns=DROP_COLUMNS)

    # Fix inappropriate values by clipping to valid ranges
    for col, (lo, hi) in CLIP_RANGES.items():
        if lo is not None:
            df[col] = df[col].clip(lower=lo)
        if hi is not None:
            df[col] = df[col].clip(upper=hi)

    return df


def split_data(df, val_size=0.15, test_size=0.15, random_state=42):
    """Stratified split into train (70%), validation (15%), test (15%).

    Stratification preserves the ~29% churn rate in every split.
    """
    remaining = 1.0 - (val_size + test_size)  # 0.70

    # Split 1: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=1 - remaining, stratify=df[TARGET], random_state=random_state
    )

    # Split 2: val vs test (within temp)
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_ratio, stratify=temp_df[TARGET], random_state=random_state
    )

    return train_df, val_df, test_df


def build_preprocessor(df):
    """Create a ColumnTransformer: median-impute+scale numerics,
    mode-impute+onehot categoricals."""
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
    """Load, clean, split, and save data; fit and save preprocessor."""
    df = load_and_clean_data()
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Dropped:      {DROP_COLUMNS}")
    print(f"Cleaned (clip): {list(CLIP_RANGES.keys())}")

    train_df, val_df, test_df = split_data(df)

    # Save splits as CSV
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(SPLIT_DIR, f"{name}.csv")
        split_df.to_csv(path, index=False)
        churn_rate = split_df[TARGET].mean() * 100
        print(f"  {name:>5}: {len(split_df):>6} rows ({churn_rate:.1f}% churn) → {path}")

    # Fit preprocessor on training data and save for reuse
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    X_train = train_df.drop(columns=[TARGET])
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))

    print(f"\nNumerical features   ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print("\n✅ Preprocessing complete. Splits and preprocessor saved.")


if __name__ == "__main__":
    main()