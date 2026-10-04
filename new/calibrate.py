from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from preprocessing import load_and_clean_data, split_data, build_preprocessor, MODEL_DIR, TARGET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_READY_CSV = os.path.join(BASE_DIR, "outputs", "llm_ready", "llm_ready_xgboost.csv")
OUT_DIR = os.path.join(BASE_DIR, "new", "outputs", "calibration")
os.makedirs(OUT_DIR, exist_ok=True)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def main() -> None:
    print("=" * 70)
    print("  WP-C1 — XGBoost probability calibration")
    print("=" * 70)

    df = load_and_clean_data()
    train_df, val_df, test_df = split_data(df)

    preprocessor, _, _ = build_preprocessor(df)
    preprocessor.fit(train_df.drop(columns=[TARGET]))

    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.joblib"))

    X_val = preprocessor.transform(val_df.drop(columns=[TARGET]))
    y_val = val_df[TARGET].to_numpy()
    val_proba_raw = xgb_model.predict_proba(X_val)[:, 1]

    print(f"  Validation set: {len(val_df)} rows (churn rate {y_val.mean():.3f})")

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_proba_raw, y_val)

    eps = 1e-6
    val_logit = np.log(np.clip(val_proba_raw, eps, 1 - eps) / np.clip(1 - val_proba_raw, eps, 1 - eps)).reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(val_logit, y_val)

    joblib.dump(isotonic, os.path.join(MODEL_DIR, "xgb_calibrator_isotonic.joblib"))
    joblib.dump(platt, os.path.join(MODEL_DIR, "xgb_calibrator_platt.joblib"))
    print(f"  Calibrators saved -> {MODEL_DIR}/xgb_calibrator_{{isotonic,platt}}.joblib")

    llm_ready = pd.read_csv(LLM_READY_CSV)
    y_true_test = llm_ready["y_true"].to_numpy()
    proba_raw_test = llm_ready["y_proba"].to_numpy()

    proba_iso_test = isotonic.predict(proba_raw_test)
    test_logit = np.log(np.clip(proba_raw_test, eps, 1 - eps) / np.clip(1 - proba_raw_test, eps, 1 - eps)).reshape(-1, 1)
    proba_platt_test = platt.predict_proba(test_logit)[:, 1]

    out = pd.DataFrame({
        "row_id": np.arange(len(llm_ready)),
        "y_true": y_true_test,
        "y_proba_raw": proba_raw_test,
        "y_proba_isotonic": proba_iso_test,
        "y_proba_platt": proba_platt_test,
    })
    out_path = os.path.join(OUT_DIR, "test_calibrated_probabilities.csv")
    out.to_csv(out_path, index=False)
    print(f"  Calibrated test probabilities saved -> {out_path}")

    print(f"\n  {'Scale':<12}{'Brier':>10}{'ECE':>10}{'n in [0.4,0.8)':>16}")
    for label, probs in [
        ("raw", proba_raw_test),
        ("isotonic", proba_iso_test),
        ("platt", proba_platt_test),
    ]:
        brier = brier_score_loss(y_true_test, probs)
        ece = expected_calibration_error(y_true_test, probs)
        n_ambiguous = int(((probs >= 0.4) & (probs < 0.8)).sum())
        print(f"  {label:<12}{brier:>10.4f}{ece:>10.4f}{n_ambiguous:>16}")

    print("\n  Done. Next: python new/build_ablation_sample.py")


if __name__ == "__main__":
    main()
