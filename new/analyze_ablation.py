from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_CSV = os.path.join(BASE_DIR, "new", "outputs", "analysis", "ablation_merged.csv")
OUT_DIR = os.path.join(BASE_DIR, "new", "outputs", "analysis")

ARMS_ORDER = ["A0", "A1", "A2", "A3", "A4", "A6", "A7", "A5", "A8"]
N_BOOT = 2000
SEED = 42


def weighted_metrics(y_true, y_pred, weight) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=weight),
        "precision": precision_score(y_true, y_pred, sample_weight=weight, zero_division=0),
        "recall": recall_score(y_true, y_pred, sample_weight=weight, zero_division=0),
        "f1": f1_score(y_true, y_pred, sample_weight=weight, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred, sample_weight=weight),
    }


def stratified_bootstrap_ci(df: pd.DataFrame, metric_fn, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    strata = {s: g.reset_index(drop=True) for s, g in df.groupby("stratum")}
    values = []
    for _ in range(n_boot):
        parts = []
        for s, g in strata.items():
            idx = rng.integers(0, len(g), size=len(g))
            parts.append(g.iloc[idx])
        resampled = pd.concat(parts, ignore_index=True)
        values.append(metric_fn(resampled))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return lo, hi


def main() -> None:
    merged = pd.read_csv(MERGED_CSV)
    merged["y_pred_bin"] = (merged["decision"].astype(str).str.strip().str.lower() == "churn").astype(int)
    merged["y_true"] = merged["y_true"].astype(int)

    rows = []
    for arm in ARMS_ORDER:
        for model in ("qwen", "gemma"):
            sub = merged[(merged["arm"] == arm) & (merged["model"] == model)]
            if sub.empty:
                continue
            m = weighted_metrics(sub["y_true"], sub["y_pred_bin"], sub["weight"])

            def f1_fn(d):
                return f1_score(d["y_true"], (d["decision"].str.strip().str.lower() == "churn").astype(int),
                                 sample_weight=d["weight"], zero_division=0)

            def bacc_fn(d):
                return balanced_accuracy_score(
                    d["y_true"], (d["decision"].str.strip().str.lower() == "churn").astype(int),
                    sample_weight=d["weight"],
                )

            f1_lo, f1_hi = stratified_bootstrap_ci(sub, f1_fn)
            bacc_lo, bacc_hi = stratified_bootstrap_ci(sub, bacc_fn)

            rows.append({
                "arm": arm, "model": model, "n": len(sub),
                "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "f1_lo": f1_lo, "f1_hi": f1_hi,
                "balanced_accuracy": m["balanced_accuracy"], "bacc_lo": bacc_lo, "bacc_hi": bacc_hi,
            })

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(OUT_DIR, "arm_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("=" * 100)
    print("  Per-arm weighted metrics (reconstructs full-7,501-row test-set estimates)")
    print("=" * 100)
    for model in ("qwen", "gemma"):
        print(f"\n  [{model.upper()}]")
        print(f"  {'arm':<4}{'n':>6}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}{'  [95% CI]':<18}"
              f"{'bal.acc':>9}{'  [95% CI]':<18}")
        sub = metrics_df[metrics_df["model"] == model].set_index("arm").reindex(ARMS_ORDER).dropna(how="all")
        for arm, r in sub.iterrows():
            print(f"  {arm:<4}{int(r['n']):>6}{r['accuracy']:>8.4f}{r['precision']:>8.4f}{r['recall']:>8.4f}"
                  f"{r['f1']:>8.4f}  [{r['f1_lo']:.3f},{r['f1_hi']:.3f}]"
                  f"{r['balanced_accuracy']:>9.4f}  [{r['bacc_lo']:.3f},{r['bacc_hi']:.3f}]")

    print("\n" + "=" * 100)
    print("  Contribution decomposition (balanced accuracy, weighted)")
    print("=" * 100)
    decomp_rows = []
    for model in ("qwen", "gemma"):
        get = lambda a: metrics_df[(metrics_df["arm"] == a) & (metrics_df["model"] == model)]["balanced_accuracy"]
        a0 = get("A0").iloc[0] if not get("A0").empty else None
        a1 = get("A1").iloc[0] if not get("A1").empty else None
        a2 = get("A2").iloc[0] if not get("A2").empty else None
        a3 = get("A3").iloc[0] if not get("A3").empty else None
        a4 = get("A4").iloc[0] if not get("A4").empty else None
        a5 = get("A5").iloc[0] if not get("A5").empty else None
        a6 = get("A6").iloc[0] if not get("A6").empty else None
        a7 = get("A7").iloc[0] if not get("A7").empty else None

        print(f"\n  [{model.upper()}]")
        print(f"    A0 raw features alone                    : {a0:.4f}")
        print(f"    A1 (+benchmarks) - A0  [benchmark value]  : {a1 - a0:+.4f}")
        print(f"    A2 (+SHAP)      - A0  [SHAP value]        : {a2 - a0:+.4f}")
        print(f"    A3 verdict-only alone                     : {a3:.4f}")
        print(f"    A5 full panel                             : {a5:.4f}")
        print(f"    A5 - A3  [gap: full panel vs. verdict-only]: {a5 - a3:+.4f}")
        print(f"    A5 - A4  [override-guard value]           : {a5 - a4:+.4f}")
        print(f"    A5 - A6  [RF panel value (vs XGB-only)]   : {a5 - a6:+.4f}")
        print(f"    A5 - A7  [RF global-importances value]    : {a5 - a7:+.4f}")

        gate1_gap = abs(a5 - a3)
        gate1_verdict = "A3 ~ A5 (imitation-dominant)" if gate1_gap < 0.02 else "A3 clearly != A5 (independent signal beyond imitation)"
        print(f"    >>> GATE 1: |A5 - A3| = {gate1_gap:.4f}  ->  {gate1_verdict}")

        decomp_rows.append({
            "model": model, "A0": a0, "A1": a1, "A2": a2, "A3": a3, "A4": a4,
            "A5": a5, "A6": a6, "A7": a7,
            "benchmark_value": a1 - a0, "shap_value": a2 - a0,
            "override_guard_value": a5 - a4, "rf_panel_value": a5 - a6,
            "rf_importances_value": a5 - a7, "gate1_gap_A5_minus_A3": a5 - a3,
        })
    pd.DataFrame(decomp_rows).to_csv(os.path.join(OUT_DIR, "decomposition.csv"), index=False)

    print("\n" + "=" * 100)
    print("  A8 Anchoring Index: P(SLM decision == deliberately INVERTED panel verdict)")
    print("=" * 100)
    a8 = merged[merged["arm"] == "A8"].copy()
    a8["shown_inverted"] = a8["xgb_decision"].astype(str).str.upper().map({"CHURN": "RETAIN", "RETAIN": "CHURN"})
    a8["followed_inverted"] = (a8["decision"].str.strip().str.lower() == a8["shown_inverted"].str.lower())
    ambiguous_mask = a8["risk_tier"].isin(["Medium", "High"])

    anchor_rows = []
    for model in ("qwen", "gemma"):
        sub = a8[a8["model"] == model]
        overall = np.average(sub["followed_inverted"], weights=sub["weight"])
        amb = sub[ambiguous_mask.loc[sub.index]]
        amb_rate = (np.average(amb["followed_inverted"], weights=amb["weight"]) if len(amb) else float("nan"))
        print(f"  [{model.upper()}] overall (n={len(sub)}): {overall:.1%}   |   "
              f"ambiguous-zone only (Medium/High tier, n={len(amb)}): {amb_rate:.1%}")
        anchor_rows.append({"model": model, "n_overall": len(sub), "anchoring_index_overall": overall,
                             "n_ambiguous": len(amb), "anchoring_index_ambiguous": amb_rate})
    pd.DataFrame(anchor_rows).to_csv(os.path.join(OUT_DIR, "anchoring_index.csv"), index=False)

    print(f"\n  Saved -> {metrics_path}")
    print(f"  Saved -> {os.path.join(OUT_DIR, 'decomposition.csv')}")
    print(f"  Saved -> {os.path.join(OUT_DIR, 'anchoring_index.csv')}")


if __name__ == "__main__":
    main()
