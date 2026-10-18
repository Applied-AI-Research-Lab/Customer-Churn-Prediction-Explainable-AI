from __future__ import annotations

import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "new"))
from preprocessing import load_and_clean_data, split_data, build_preprocessor, TARGET
from generate_llm_ready import get_feature_names, generate_top_reasons_simple
from analyze_ablation import stratified_bootstrap_ci

MERGED = os.path.join(BASE, "new", "outputs", "analysis", "ablation_merged.csv")
A2_ROWS = os.path.join(BASE, "new", "outputs", "predictions", "A2", "both.csv")
OUT_DIR = os.path.join(BASE, "new", "outputs", "round2")
os.makedirs(OUT_DIR, exist_ok=True)

SHAP_RE = re.compile(r"SHAP=([+-]?[0-9.]+)")

def displayed_top5_sum(row: pd.Series) -> float:
    total = 0.0
    for k in range(1, 6):
        m = SHAP_RE.search(str(row.get(f"reason_{k}", "")))
        if m:
            total += float(m.group(1))
    return total

def wba(df: pd.DataFrame, col: str) -> float:
    return balanced_accuracy_score(df["y_true"], df[col], sample_weight=df["weight"])

def val_top5_sums() -> tuple[np.ndarray, np.ndarray]:
    df = load_and_clean_data()
    train_df, val_df, _ = split_data(df)
    pre, _, _ = build_preprocessor(df)
    pre.fit(train_df.drop(columns=[TARGET]))
    X_val = val_df.drop(columns=[TARGET])
    Xp = pre.transform(X_val)
    names = get_feature_names(pre, df)
    model = joblib.load(os.path.join(BASE, "outputs", "models", "xgboost.joblib"))
    sv = shap.TreeExplainer(model).shap_values(Xp)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    orig = X_val.columns.tolist()
    sums = np.empty(len(X_val))
    for i in range(len(X_val)):
        reasons = generate_top_reasons_simple(sv[i], names, X_val.iloc[i], orig)[:5]
        sums[i] = sum(round(r["shap_value"], 2) for r in reasons)
    return sums, val_df[TARGET].to_numpy()

def main() -> None:
    merged = pd.read_csv(MERGED)
    merged["y_pred_bin"] = (merged["decision"].str.lower() == "churn").astype(int)
    merged["xgb_bin"] = (merged["xgb_decision"].str.upper() == "CHURN").astype(int)

    rows = []
    for (arm, model), sub in merged.groupby(["arm", "model"]):
        sub = sub.reset_index(drop=True)
        ba = wba(sub, "y_pred_bin")
        lo, hi = stratified_bootstrap_ci(sub, lambda d: wba(d, "y_pred_bin"))
        agree = np.average(sub["y_pred_bin"] == sub["xgb_bin"], weights=sub["weight"])
        alo, ahi = stratified_bootstrap_ci(
            sub, lambda d: np.average(d["y_pred_bin"] == d["xgb_bin"], weights=d["weight"]))
        rows.append(dict(arm=arm, model=model, wBA=ba, wBA_lo=lo, wBA_hi=hi,
                         agree_xgb=agree, agree_lo=alo, agree_hi=ahi))
    arms = pd.DataFrame(rows).sort_values(["arm", "model"])
    arms.to_csv(os.path.join(OUT_DIR, "arm_agreement_with_xgb.csv"), index=False)

    a2 = pd.read_csv(A2_ROWS)
    a2["shap_sum"] = a2.apply(displayed_top5_sum, axis=1)
    a2["xgb_bin"] = (a2["churn_decision"].str.upper() == "CHURN").astype(int)

    v_sum, v_y = val_top5_sums()
    grid = np.round(np.arange(-2.0, 2.0001, 0.01), 2)
    t_star = max(grid, key=lambda t: balanced_accuracy_score(v_y, (v_sum > t).astype(int)))
    val_ba = balanced_accuracy_score(v_y, (v_sum > t_star).astype(int))

    a2["rule0"] = (a2["shap_sum"] > 0).astype(int)
    a2["ruleV"] = (a2["shap_sum"] > t_star).astype(int)
    base = []
    for name, col in [("XGBoost (verdict)", "xgb_bin"), ("SHAP-sum > 0", "rule0"),
                      (f"SHAP-sum > {t_star:+.2f} (val-selected)", "ruleV")]:
        lo, hi = stratified_bootstrap_ci(a2, lambda d, c=col: wba(d, c))
        agree = np.average(a2[col] == a2["xgb_bin"], weights=a2["weight"])
        base.append(dict(rule=name, wBA=wba(a2, col), wBA_lo=lo, wBA_hi=hi, agree_xgb=agree))
    base = pd.DataFrame(base)
    base.to_csv(os.path.join(OUT_DIR, "a2_shap_sum_baselines.csv"), index=False)

    a2m = merged[merged["arm"] == "A2"]
    beyond = []
    for model, sub in a2m.groupby("model"):
        slm_ok = sub["y_pred_bin"] == sub["y_true"]
        xgb_ok = sub["xgb_bin"] == sub["y_true"]
        b = (slm_ok & ~xgb_ok)
        c = (~slm_ok & xgb_ok)
        p = binomtest(int(b.sum()), int(b.sum() + c.sum()), 0.5).pvalue if (b.sum() + c.sum()) else np.nan
        beyond.append(dict(model=model, slm_right_xgb_wrong=int(b.sum()), xgb_right_slm_wrong=int(c.sum()),
                           w_slm_right_xgb_wrong=np.average(b, weights=sub["weight"]),
                           w_xgb_right_slm_wrong=np.average(c, weights=sub["weight"]),
                           exact_binomial_p=p))
    beyond = pd.DataFrame(beyond)
    beyond.to_csv(os.path.join(OUT_DIR, "a2_value_beyond_classifier.csv"), index=False)

    pd.set_option("display.width", 200)
    print("Per-arm weighted BA and agreement with XGBoost:\n", arms.round(4).to_string(index=False))
    print(f"\nValidation-selected SHAP-sum threshold t*={t_star:+.2f} (val BA {val_ba:.4f})")
    print("A2-sample baselines:\n", base.round(4).to_string(index=False))
    print("\nA2 value beyond XGBoost:\n", beyond.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
