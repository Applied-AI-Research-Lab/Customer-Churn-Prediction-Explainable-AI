"""
Train two classic, research-backed models for churn prediction:

  1. Random Forest  – robust bagging ensemble; handles non-linearity,
     outliers, and class imbalance (via class_weight='balanced').
  2. XGBoost        – gradient-boosted trees; consistently a top performer
     on tabular churn datasets in research and Kaggle competitions.

Both models are evaluated on a held-out validation set (for tuning /
early-stopping) and a final test set.  Test-set churn predictions are
saved to outputs/predictions/.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

# Ensure we can import preprocessing from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    load_and_clean_data, split_data, build_preprocessor,
    SPLIT_DIR, MODEL_DIR, TARGET,
)

PRED_DIR = os.path.join(os.path.dirname(MODEL_DIR), "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(os.path.dirname(MODEL_DIR), "model_results.json")


# ── Evaluation helper ────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_proba):
    """Return a dict of standard binary-classification metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_proba),
    }


def print_metrics(metrics, label):
    print(f"\n  [{label}]")
    for k, v in metrics.items():
        print(f"    {k:>10}: {v:.4f}")


# ── Main training routine ────────────────────────────────────────────────────
def main():
    # ── 1. Load, clean, and split data ───────────────────────────────────────
    df = load_and_clean_data()
    train_df, val_df, test_df = split_data(df)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_val   = val_df.drop(columns=[TARGET])
    y_val   = val_df[TARGET]
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    # ── 2. Preprocess (fit on train, transform all) ──────────────────────────
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    preprocessor.fit(X_train)

    X_train_p = preprocessor.transform(X_train)
    X_val_p   = preprocessor.transform(X_val)
    X_test_p  = preprocessor.transform(X_test)

    # Class imbalance ratio for XGBoost
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos  # ≈ 2.46

    # ── 3. Define models ─────────────────────────────────────────────────────
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",       # handle 29% churn imbalance
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,  # handle imbalance
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            early_stopping_rounds=20,           # stop if no val improvement
        ),
    }

    all_results = {}

    # ── 4. Train & evaluate each model ───────────────────────────────────────
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        if name == "xgboost":
            model.fit(
                X_train_p, y_train,
                eval_set=[(X_val_p, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train_p, y_train)

        # ── Validation set ──────────────────────────────────────────────────
        y_val_pred  = model.predict(X_val_p)
        y_val_proba = model.predict_proba(X_val_p)[:, 1]
        val_metrics = evaluate(y_val, y_val_pred, y_val_proba)
        print_metrics(val_metrics, "Validation")

        # ── Test set ────────────────────────────────────────────────────────
        y_test_pred  = model.predict(X_test_p)
        y_test_proba = model.predict_proba(X_test_p)[:, 1]
        test_metrics = evaluate(y_test, y_test_pred, y_test_proba)
        print_metrics(test_metrics, "Test")

        print(f"\n  Classification Report (Test):")
        print(classification_report(y_test, y_test_pred, target_names=["Retained", "Churned"]))

        cm = confusion_matrix(y_test, y_test_pred)
        print(f"  Confusion Matrix (Test):")
        print(f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
        print(f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

        # Save test-set predictions
        pred_df = pd.DataFrame({
            "y_true":  y_test.values,
            "y_pred":  y_test_pred,
            "y_proba": y_test_proba,
        })
        pred_path = os.path.join(PRED_DIR, f"test_predictions_{name}.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"\n  Predictions saved → {pred_path}")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"  Model saved      → {model_path}")

        all_results[name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

    # Save preprocessor for deployment / inference
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))

    # Save all results as JSON
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── 5. Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY — Test Set Performance")
    print(f"{'='*60}")
    header = f"  {'Model':<16} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    print(header)
    print(f"  {'-'*66}")
    for name, res in all_results.items():
        m = res["test"]
        print(f"  {name:<16} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}")

    print(f"\n  Results JSON  → {RESULTS_PATH}")
    print(f"  Models        → {MODEL_DIR}/")
    print(f"  Predictions   → {PRED_DIR}/")


if __name__ == "__main__":
    main()