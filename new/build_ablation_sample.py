from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_READY_CSV = os.path.join(BASE_DIR, "outputs", "llm_ready", "llm_ready_xgboost.csv")
CALIBRATED_CSV = os.path.join(BASE_DIR, "new", "outputs", "calibration", "test_calibrated_probabilities.csv")
FULL_RUN_CSV = os.path.join(BASE_DIR, "outputs", "llm_predictions", "test_llm_predictions.csv")
OUT_PATH = os.path.join(BASE_DIR, "new", "outputs", "ablation_sample.csv")


def stratified_fixed_n_sample(df: pd.DataFrame, strata_col: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = df.groupby(strata_col).size()
    raw_alloc = sizes / sizes.sum() * n
    alloc = np.floor(raw_alloc).astype(int)
    shortfall = n - int(alloc.sum())
    frac_order = (raw_alloc - alloc).sort_values(ascending=False).index
    for key in frac_order[:shortfall]:
        alloc[key] += 1
    alloc = pd.Series({k: min(v, sizes[k]) for k, v in alloc.items()})

    shortfall = n - int(alloc.sum())
    while shortfall > 0:
        spare = (sizes - alloc)
        spare = spare[spare > 0]
        if spare.empty:
            break
        key = spare.index[0]
        add = min(shortfall, int(spare.loc[key]))
        alloc[key] += add
        shortfall -= add

    parts = []
    groups = df.groupby(strata_col)
    for key, k in alloc.items():
        if k <= 0:
            continue
        g = groups.get_group(key)
        idx = rng.choice(g.index.values, size=int(k), replace=False)
        parts.append(df.loc[idx])
    return pd.concat(parts).sort_index()


def weighted_metrics(y_true, y_pred, weight) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=weight),
        "f1": f1_score(y_true, y_pred, sample_weight=weight, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred, sample_weight=weight),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("  WP-S — Freeze the ablation evaluation sample")
    print("=" * 70)

    if not os.path.exists(CALIBRATED_CSV):
        raise FileNotFoundError(
            f"{CALIBRATED_CSV} not found — run `python new/calibrate.py` first."
        )

    llm_ready = pd.read_csv(LLM_READY_CSV).reset_index(drop=True)
    llm_ready["row_id"] = np.arange(len(llm_ready))
    calib = pd.read_csv(CALIBRATED_CSV)

    df = llm_ready.merge(calib[["row_id", "y_proba_isotonic"]], on="row_id", how="left")

    raw_ambiguous = (df["y_proba"] >= 0.4) & (df["y_proba"] < 0.8)
    iso_ambiguous = (df["y_proba_isotonic"] >= 0.4) & (df["y_proba_isotonic"] < 0.8)
    stratum_a_mask = raw_ambiguous | iso_ambiguous

    print(f"  Raw-ambiguous zone       : {int(raw_ambiguous.sum())} rows")
    print(f"  Isotonic-ambiguous zone  : {int(iso_ambiguous.sum())} rows")
    print(f"  Union (Stratum A)        : {int(stratum_a_mask.sum())} rows")

    stratum_a = df.loc[stratum_a_mask].copy()
    stratum_a["stratum"] = "A_ambiguous"
    stratum_a["weight"] = 1.0

    remainder = df.loc[~stratum_a_mask].copy()
    n_b = args.n_total - len(stratum_a)
    if n_b <= 0:
        raise ValueError(
            f"Stratum A alone ({len(stratum_a)} rows) already exceeds --n-total "
            f"({args.n_total}); increase --n-total."
        )

    strata_key = (
        remainder["churn_decision"].astype(str) + "_"
        + remainder["risk_tier"].astype(str) + "_"
        + remainder["y_true"].astype(str)
    )
    remainder = remainder.assign(_strata_key=strata_key)
    stratum_b = stratified_fixed_n_sample(remainder, "_strata_key", n_b, args.seed)
    stratum_b = stratum_b.drop(columns=["_strata_key"]).copy()
    stratum_b["stratum"] = "B_confident"
    stratum_b["weight"] = len(remainder) / len(stratum_b)

    print(f"  Stratum B (confident)    : {len(stratum_b)} rows "
          f"(target {n_b}, weight={stratum_b['weight'].iloc[0]:.3f})")

    sample = pd.concat([stratum_a, stratum_b], ignore_index=True)
    sample = sample[["row_id", "stratum", "weight", "y_true", "y_proba",
                      "y_proba_isotonic", "churn_decision", "risk_tier"]]
    sample = sample.sort_values("row_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sample.to_csv(OUT_PATH, index=False)
    print(f"\n  Frozen sample saved -> {OUT_PATH}  ({len(sample)} rows)")

    print("\n" + "=" * 70)
    print("  Validation check: reweighted A5 (existing full run) vs. known full-test-set metrics")
    print("=" * 70)

    if not os.path.exists(FULL_RUN_CSV):
        print(f"  SKIPPED — {FULL_RUN_CSV} not found.")
        return

    full = pd.read_csv(FULL_RUN_CSV).reset_index(drop=True)
    full["row_id"] = np.arange(len(full))
    joined = sample.merge(
        full[["row_id", "y_true", "qwen_decision", "gemma_decision"]],
        on="row_id", how="left", suffixes=("", "_full"),
    )

    all_good = True
    for prefix in ("qwen", "gemma"):
        col = f"{prefix}_decision"
        mask = joined[col].notna() & ~joined[col].astype(str).str.startswith("[parse_error]")
        sub = joined.loc[mask]
        y_true_sample = sub["y_true"].astype(int)
        y_pred_sample = (sub[col].str.strip().str.lower() == "churn").astype(int)
        weighted = weighted_metrics(y_true_sample, y_pred_sample, sub["weight"])

        full_mask = full[col].notna() & ~full[col].astype(str).str.startswith("[parse_error]")
        full_sub = full.loc[full_mask]
        y_true_full = full_sub["y_true"].astype(int)
        y_pred_full = (full_sub[col].str.strip().str.lower() == "churn").astype(int)
        true_full = {
            "accuracy": accuracy_score(y_true_full, y_pred_full),
            "f1": f1_score(y_true_full, y_pred_full, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true_full, y_pred_full),
        }

        print(f"\n  [{prefix.upper()}]")
        for metric in ("accuracy", "f1", "balanced_accuracy"):
            delta = weighted[metric] - true_full[metric]
            flag = "OK" if abs(delta) < 0.01 else "CHECK"
            if abs(delta) >= 0.01:
                all_good = False
            print(f"    {metric:<18} reweighted={weighted[metric]:.4f}  "
                  f"full={true_full[metric]:.4f}  delta={delta:+.4f}  [{flag}]")

    print("\n  " + ("PASS — sampling/weighting reconstructs the full-test-set estimate."
                     if all_good else
                     "CHECK — a delta exceeds 0.01; inspect the sample before spending GPU time."))


if __name__ == "__main__":
    main()
