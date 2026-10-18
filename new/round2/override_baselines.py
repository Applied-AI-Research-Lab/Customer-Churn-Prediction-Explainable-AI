from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
from preprocessing import load_and_clean_data, split_data, build_preprocessor, TARGET
from roi_sensitivity import (net_value, BASE_RETENTION_COST, BASE_ACQUISITION_COST,
                             BASE_SUCCESS_RATE)

OUT_DIR = os.path.join(BASE, "outputs", "analysis")
TEST_CSV = os.path.join(BASE, "outputs", "llm_predictions", "test_llm_predictions.csv")
GRID = np.round(np.arange(0.05, 0.8001, 0.01), 2)
BINS = [(0.4, 0.6), (0.6, 0.8), (0.4, 0.8)]
N_PERM = 10000
SEED = 42

def validation_scores() -> pd.DataFrame:
    df = load_and_clean_data()
    train_df, val_df, _ = split_data(df)
    pre, _, _ = build_preprocessor(df)
    pre.fit(train_df.drop(columns=[TARGET]))
    model = joblib.load(os.path.join(BASE, "outputs", "models", "xgboost.joblib"))
    p = model.predict_proba(pre.transform(val_df.drop(columns=[TARGET])))[:, 1]
    return pd.DataFrame({"p": p, "y_true": val_df[TARGET].to_numpy(),
                         "Lifetime_Value": val_df["Lifetime_Value"].to_numpy()})

def nv(df: pd.DataFrame, pred: np.ndarray) -> float:
    tmp = df.assign(_pred=pred)
    return net_value(tmp, "_pred", BASE_RETENTION_COST, BASE_ACQUISITION_COST, BASE_SUCCESS_RATE)

def select_thresholds(val: pd.DataFrame) -> dict:
    crit = {
        "accuracy": lambda pr: accuracy_score(val["y_true"], pr),
        "balanced accuracy": lambda pr: balanced_accuracy_score(val["y_true"], pr),
        "F1": lambda pr: f1_score(val["y_true"], pr),
        "net value": lambda pr: nv(val, pr),
    }
    return {name: float(max(GRID, key=lambda t: f((val["p"] >= t).astype(int).to_numpy())))
            for name, f in crit.items()}

def metrics(df: pd.DataFrame, pred: np.ndarray, ref: np.ndarray) -> dict:
    y = df["y_true"].to_numpy()
    b = int(((pred == y) & (ref != y)).sum())
    c = int(((pred != y) & (ref == y)).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else np.nan
    out = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
               f1=f1_score(y, pred), net_value=nv(df, pred),
               right_vs_qwen=b, wrong_vs_qwen=c, mcnemar_exact_p_vs_qwen=p)
    for lo, hi in BINS:
        m = (df["y_proba"] >= lo) & (df["y_proba"] < hi)
        out[f"acc_[{lo},{hi})"] = accuracy_score(y[m], pred[m])
    return out

def main() -> None:
    val = validation_scores()
    thr = select_thresholds(val)

    test = pd.read_csv(TEST_CSV)
    p, y = test["y_proba"].to_numpy(), test["y_true"].to_numpy()
    xgb = (p >= 0.5).astype(int)
    qwen = (test["qwen_decision"].str.strip().str.lower() == "churn").astype(int).to_numpy()
    gemma = (test["gemma_decision"].str.strip().str.lower() == "churn").astype(int).to_numpy()
    iso = joblib.load(os.path.join(BASE, "outputs", "models", "xgb_calibrator_isotonic.joblib"))

    rules = {"XGBoost, t=0.50 (default)": xgb,
             "Qwen3.5-4B (A5, full panel)": qwen,
             "Gemma3-4B (A5, full panel)": gemma,
             "Isotonic-calibrated XGBoost, t=0.50": (iso.predict(p) >= 0.5).astype(int)}
    for name, t in thr.items():
        rules[f"XGBoost, t={t:.2f} (val-max {name})"] = (p >= t).astype(int)
    table = pd.DataFrame([dict(rule=k, **metrics(test, v, qwen)) for k, v in rules.items()])
    table.to_csv(os.path.join(OUT_DIR, "override_baselines.csv"), index=False)

    rng = np.random.default_rng(SEED)
    rows = []
    for lo, hi in BINS:
        in_bin = (p >= lo) & (p < hi) & (xgb == 1)
        flips = in_bin & (qwen == 0)
        n, k = int(flips.sum()), int((flips & (y == 0)).sum())
        base_rate = float((y[in_bin] == 0).mean())
        idx = np.where(in_bin)[0]
        lowest = idx[np.argsort(p[idx])][:n]
        perm = np.array([(y[rng.choice(idx, n, replace=False)] == 0).sum() for _ in range(N_PERM)])
        rows.append(dict(bin=f"[{lo},{hi})", xgb_churn_calls=int(in_bin.sum()), qwen_flips=n,
                         qwen_correct_flips=k, qwen_override_acc=k / n if n else np.nan,
                         random_flip_expected_acc=base_rate,
                         binomial_p_vs_base_rate=binomtest(k, n, base_rate).pvalue if n else np.nan,
                         permutation_p_one_sided=float((perm >= k).mean()),
                         matched_rate_correct=int((y[lowest] == 0).sum()),
                         matched_rate_acc=float((y[lowest] == 0).mean()) if n else np.nan,
                         mean_p_qwen_flipped=float(p[flips].mean()) if n else np.nan,
                         mean_p_not_flipped=float(p[in_bin & (qwen == 1)].mean())))
    flips_df = pd.DataFrame(rows)
    flips_df.to_csv(os.path.join(OUT_DIR, "override_flip_null.csv"), index=False)

    pd.set_option("display.width", 250)
    print("Validation-selected thresholds:", thr)
    print(table.round(4).to_string(index=False))
    print()
    print(flips_df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
