from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(BASE_DIR, "outputs", "llm_predictions", "test_llm_predictions.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

DELTA = 0.02
N_BOOT = 2000
SEED = 42


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric_fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
    return diffs


def tost(diffs: np.ndarray, delta: float):
    lo, hi = np.percentile(diffs, [5, 95])
    equivalent = (lo > -delta) and (hi < delta)
    return lo, hi, equivalent


def main() -> None:
    df = pd.read_csv(PRED_CSV)
    y_true = df["y_true"].astype(int).to_numpy()
    xgb_pred = (df["churn_decision"].str.upper() == "CHURN").astype(int).to_numpy()
    gemma_pred = (df["gemma_decision"].str.strip().str.lower() == "churn").astype(int).to_numpy()

    print("=" * 90)
    print(f"  WP-B2 — Equivalence testing, Gemma vs. XGBoost (pre-specified margin delta={DELTA})")
    print("=" * 90)

    rows = []
    for name, metric_fn in [("balanced_accuracy", balanced_accuracy_score),
                             ("f1", lambda yt, yp: f1_score(yt, yp, zero_division=0))]:
        point_diff = metric_fn(y_true, gemma_pred) - metric_fn(y_true, xgb_pred)
        diffs = paired_bootstrap_diff(y_true, gemma_pred, xgb_pred, metric_fn)
        lo90, hi90, equiv = tost(diffs, DELTA)
        lo95, hi95 = np.percentile(diffs, [2.5, 97.5])
        print(f"\n  [{name}]")
        print(f"    Point estimate (Gemma - XGBoost): {point_diff:+.4f}")
        print(f"    90% CI (TOST): [{lo90:+.4f}, {hi90:+.4f}]  ->  "
              f"{'EQUIVALENT (within +-'+str(DELTA)+')' if equiv else 'NOT equivalent at this margin'}")
        print(f"    95% CI (two-sided, for reference): [{lo95:+.4f}, {hi95:+.4f}]")
        rows.append({"metric": name, "point_diff": point_diff, "delta": DELTA,
                      "tost_ci_lo": lo90, "tost_ci_hi": hi90, "equivalent": equiv,
                      "ci95_lo": lo95, "ci95_hi": hi95})

    print(f"\n  Non-inferiority (Gemma not worse than XGBoost by more than {DELTA}):")
    for r in rows:
        non_inferior = r["tost_ci_lo"] > -DELTA
        print(f"    [{r['metric']}] lower bound {r['tost_ci_lo']:+.4f} > -{DELTA}  ->  "
              f"{'non-inferior' if non_inferior else 'NOT established'}")
        r["non_inferior"] = non_inferior

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "equivalence_tests.csv"), index=False)
    print(f"\n  Saved -> {os.path.join(OUT_DIR, 'equivalence_tests.csv')}")

    print("\n  Caveat (R3.4's second point, stands regardless of the result above):")
    print("  Gemma is supplied XGBoost's own prediction in the full-panel prompt (arm A5), and the")
    print("  ablation grid (new/outputs/analysis/decomposition.csv) shows A3 (verdict-only) = A5 exactly")
    print("  for Gemma. An equivalence result here is therefore a statement about faithful narration")
    print("  of a supplied verdict, not independent predictive ability.")


if __name__ == "__main__":
    main()
