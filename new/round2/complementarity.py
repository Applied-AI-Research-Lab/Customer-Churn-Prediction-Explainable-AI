from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import SplineTransformer

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "new"))
from preprocessing import load_and_clean_data, split_data, TARGET
from analyze_ablation import stratified_bootstrap_ci

MERGED = os.path.join(BASE, "new", "outputs", "analysis", "ablation_merged.csv")
A1_ROWS = os.path.join(BASE, "new", "outputs", "predictions", "A1", "both.csv")
LLM_READY = os.path.join(BASE, "outputs", "llm_ready", "llm_ready_xgboost.csv")
BENCH = json.load(open(os.path.join(BASE, "outputs", "llm_ready", "population_benchmarks.json")))
OUT_DIR = os.path.join(BASE, "new", "outputs", "round2")
EPS = 1e-6

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def _loglik(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    m = LogisticRegression(penalty=None, max_iter=1000).fit(X, y, sample_weight=w)
    p = np.clip(m.predict_proba(X)[:, 1], EPS, 1 - EPS)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), m.coef_[0]

def complementarity(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, model), sub in merged.groupby(["arm", "model"]):
        sub = sub.reset_index(drop=True)
        y, d = sub["y_true"].to_numpy(), sub["slm_bin"].to_numpy()
        spline = SplineTransformer(n_knots=6, degree=3).fit(sub[["logit_p"]])

        def design(df: pd.DataFrame, with_slm: bool) -> np.ndarray:
            cols = [spline.transform(df[["logit_p"]]), df[["xgb_bin"]].to_numpy()]
            return np.column_stack(cols + ([df[["slm_bin"]].to_numpy()] if with_slm else []))

        base = design(sub, False)
        row = dict(arm=arm, model=model, n=len(sub), slm_churn_rate=d.mean(),
                   agree_xgb=float((d == sub["xgb_bin"]).mean()))
        if d.min() == d.max() or (d == sub["xgb_bin"]).all():
            rows.append(dict(row, note="constant" if d.min() == d.max() else "identical to XGBoost"))
            continue
        ll0, _ = _loglik(base, y)
        ll1, coef = _loglik(np.column_stack([base, d]), y)
        lr = 2 * (ll1 - ll0)

        def w_coef(df: pd.DataFrame) -> float:
            return _loglik(design(df, True), df["y_true"].to_numpy(), df["weight"].to_numpy())[1][-1]

        lo, hi = stratified_bootstrap_ci(sub, w_coef)
        cw = w_coef(sub)
        rows.append(dict(row, coef_unweighted=coef[-1], lr_stat=lr, lr_p=float(chi2.sf(lr, 1)),
                         coef_weighted=cw, coef_w_lo=lo, coef_w_hi=hi, odds_ratio_weighted=float(np.exp(cw))))
    return pd.DataFrame(rows)

def a1_feats(row: pd.Series) -> list[str]:
    reasons = [str(row.get(f"reason_{k}", "N/A")) for k in range(1, 6)]
    out: list[str] = []
    for r in reasons[:3]:
        for feat in BENCH:
            if feat.replace("_", " ").lower() in r.lower() or feat.lower() in r.lower():
                if feat not in out:
                    out.append(feat)
                break
    return out[:3] if out else list(BENCH)[:3]

def churn_votes(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    v = np.zeros(len(df))
    for f in feats:
        c, r = BENCH[f]["churned_mean"], BENCH[f]["retained_mean"]
        v += (np.abs(df[f] - c) < np.abs(df[f] - r)).to_numpy()
    return v

def benchmark_rules(merged: pd.DataFrame) -> pd.DataFrame:
    a1 = pd.read_csv(A1_ROWS)
    w = merged[(merged["arm"] == "A1") & (merged["model"] == "qwen")][["row_id", "weight", "stratum", "y_true"]]
    a1 = a1.drop(columns=[c for c in ("weight", "stratum", "y_true") if c in a1.columns]).merge(w, on="row_id")
    a1["rule_a1"] = [int(sum(abs(r[f] - BENCH[f]["churned_mean"]) < abs(r[f] - BENCH[f]["retained_mean"])
                             for f in fs) >= 2) for (_, r), fs in zip(a1.iterrows(), a1.apply(a1_feats, axis=1))]

    df = load_and_clean_data()
    _, val_df, _ = split_data(df)
    feats = list(BENCH)
    v_val = churn_votes(val_df, feats)
    k_star = max(range(len(feats) + 1),
                 key=lambda k: balanced_accuracy_score(val_df[TARGET], (v_val >= k).astype(int)))
    a1["rule_a0"] = (churn_votes(a1, feats) >= k_star).astype(int)

    out = []
    for name, col in [("Benchmark vote, 3 A1 features (>=2 of 3)", "rule_a1"),
                      (f"Benchmark vote, all 20 features (>={k_star}, val-selected)", "rule_a0")]:
        f = lambda d, c=col: balanced_accuracy_score(d["y_true"], d[c], sample_weight=d["weight"])
        lo, hi = stratified_bootstrap_ci(a1, f)
        out.append(dict(rule=name, wBA=f(a1), wBA_lo=lo, wBA_hi=hi))
    return pd.DataFrame(out)

def main() -> None:
    merged = pd.read_csv(MERGED)
    p = pd.read_csv(LLM_READY, usecols=["y_proba", "churn_decision"])
    merged["y_proba"] = p["y_proba"].to_numpy()[merged["row_id"].to_numpy()]
    chk = p["churn_decision"].to_numpy()[merged["row_id"].to_numpy()]
    assert (pd.Series(chk).str.upper().to_numpy() == merged["xgb_decision"].str.upper().to_numpy()).all()
    merged["logit_p"] = _logit(merged["y_proba"].to_numpy())
    merged["slm_bin"] = (merged["decision"].str.lower() == "churn").astype(int)
    merged["xgb_bin"] = (merged["xgb_decision"].str.upper() == "CHURN").astype(int)

    comp = complementarity(merged)
    comp.to_csv(os.path.join(OUT_DIR, "complementarity.csv"), index=False)
    rules = benchmark_rules(merged)
    rules.to_csv(os.path.join(OUT_DIR, "benchmark_rules.csv"), index=False)

    pd.set_option("display.width", 220)
    print(comp.round(4).to_string(index=False))
    print()
    print(rules.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
