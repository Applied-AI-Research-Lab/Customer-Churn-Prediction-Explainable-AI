from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = min((n - rank) * pvals[i], 1.0)
        running_max = max(running_max, adj)
        adjusted[i] = running_max
    return adjusted

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(BASE_DIR, "outputs", "llm_predictions", "test_llm_predictions.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

BINS = [
    ("[0.0, 0.2)", 0.0, 0.2),
    ("[0.2, 0.4)", 0.2, 0.4),
    ("[0.4, 0.6)", 0.4, 0.6),
    ("[0.6, 0.8)", 0.6, 0.8),
    ("[0.8, 1.0]", 0.8, 1.0 + 1e-9),
]


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def main() -> None:
    df = pd.read_csv(PRED_CSV)
    df["y_true"] = df["y_true"].astype(int)
    df["xgb_pred"] = (df["churn_decision"].str.upper() == "CHURN").astype(int)

    print("=" * 100)
    print("  Per-bin XGBoost accuracy and SLM override behaviour")
    print("=" * 100)
    bin_rows = []
    for label, lo, hi in BINS:
        mask = (df["y_proba"] >= lo) & (df["y_proba"] < hi)
        sub = df[mask]
        n = len(sub)
        xgb_acc = (sub["xgb_pred"] == sub["y_true"]).mean()
        row = {"bin": label, "n": n, "xgb_acc": xgb_acc}
        for model in ("qwen", "gemma"):
            pred = (sub[f"{model}_decision"].str.strip().str.lower() == "churn").astype(int)
            override = (pred != sub["xgb_pred"])
            override_rate = override.mean()
            model_acc = (pred == sub["y_true"]).mean()
            row[f"{model}_override_rate"] = override_rate
            row[f"{model}_acc"] = model_acc
        bin_rows.append(row)
        print(f"  {label:<12} n={n:>5}  XGB acc={xgb_acc:.4f}  "
              f"Qwen override={row['qwen_override_rate']:.1%} acc={row['qwen_acc']:.4f}  "
              f"Gemma override={row['gemma_override_rate']:.1%} acc={row['gemma_acc']:.4f}")
    pd.DataFrame(bin_rows).to_csv(os.path.join(OUT_DIR, "override_bins.csv"), index=False)

    print("\n" + "=" * 100)
    print("  Per-interval override analysis (Qwen), corrected: exact one-sample binomial test")
    print("=" * 100)
    interval_results = []
    for label, lo, hi in [("[0.4, 0.6)", 0.4, 0.6), ("[0.6, 0.8)", 0.6, 0.8)]:
        mask = (df["y_proba"] >= lo) & (df["y_proba"] < hi)
        sub = df[mask].copy()
        qwen_pred = (sub["qwen_decision"].str.strip().str.lower() == "churn").astype(int)
        overridden = sub[qwen_pred != sub["xgb_pred"]].copy()
        overridden["qwen_pred_bin"] = (overridden["qwen_decision"].str.strip().str.lower() == "churn").astype(int)
        n_override = len(overridden)
        n_correct = int((overridden["qwen_pred_bin"] == overridden["y_true"]).sum())
        acc = n_correct / n_override if n_override else float("nan")
        ci = clopper_pearson(n_correct, n_override)
        binom = stats.binomtest(n_correct, n_override, 0.5, alternative="two-sided")
        xgb_acc_bin = (sub["xgb_pred"] == sub["y_true"]).mean()

        churn_to_retain = int(((overridden["xgb_pred"] == 1) & (overridden["qwen_pred_bin"] == 0)).sum())
        retain_to_churn = int(((overridden["xgb_pred"] == 0) & (overridden["qwen_pred_bin"] == 1)).sum())

        interval_results.append({
            "interval": label, "n": len(sub), "xgb_acc": xgb_acc_bin,
            "n_override": n_override, "override_rate": n_override / len(sub),
            "n_correct": n_correct, "override_acc": acc,
            "ci_lo": ci[0], "ci_hi": ci[1], "p_value": binom.pvalue,
            "churn_to_retain": churn_to_retain, "retain_to_churn": retain_to_churn,
            "actual_churn_rate": sub["y_true"].mean(),
        })

    pvals = [r["p_value"] for r in interval_results]
    p_adj = holm_bonferroni(pvals)
    for r, padj in zip(interval_results, p_adj):
        r["p_holm"] = padj
        r["significant_at_0.05_holm"] = bool(padj < 0.05)

    for r in interval_results:
        print(f"\n  {r['interval']}  (n={r['n']}, XGB acc={r['xgb_acc']:.4f}, "
              f"actual churn rate={r['actual_churn_rate']:.1%})")
        print(f"    Overrides: {r['n_override']} ({r['override_rate']:.1%} of bin)  "
              f"[churn->retain: {r['churn_to_retain']}, retain->churn: {r['retain_to_churn']}]")
        print(f"    Override accuracy: {r['override_acc']:.4f}  "
              f"95% CI (Clopper-Pearson) [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
        print(f"    Exact binomial test vs 0.5: p={r['p_value']:.4f}  "
              f"Holm-adjusted p={r['p_holm']:.4f}  significant={r['significant_at_0.05_holm']}")

    n1, k1 = interval_results[0]["n_override"], interval_results[0]["n_correct"]
    n2, k2 = interval_results[1]["n_override"], interval_results[1]["n_correct"]
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    chi2, p_chi2, _, _ = stats.chi2_contingency(table, correction=True)
    _, p_fisher = stats.fisher_exact(table)
    print(f"\n  Heterogeneity between intervals: chi2={chi2:.4f}, p={p_chi2:.4f}  "
          f"(Fisher exact p={p_fisher:.4f})")

    n_pool = n1 + n2
    k_pool = k1 + k2
    pooled_binom = stats.binomtest(k_pool, n_pool, 0.5, alternative="two-sided")
    print(f"\n  [Pooled, demoted to footnote] n={n_pool}, override acc={k_pool/n_pool:.4f}, "
          f"p={pooled_binom.pvalue:.4f}")

    out = pd.DataFrame(interval_results)
    out.to_csv(os.path.join(OUT_DIR, "override_interval_analysis.csv"), index=False)
    print(f"\n  Saved -> {os.path.join(OUT_DIR, 'override_bins.csv')}")
    print(f"  Saved -> {os.path.join(OUT_DIR, 'override_interval_analysis.csv')}")

    with open(os.path.join(OUT_DIR, "override_heterogeneity.txt"), "w") as f:
        f.write(f"chi2={chi2:.4f}\np_chi2={p_chi2:.6f}\np_fisher={p_fisher:.6f}\n")
        f.write(f"pooled_n={n_pool}\npooled_acc={k_pool/n_pool:.4f}\npooled_p={pooled_binom.pvalue:.6f}\n")


if __name__ == "__main__":
    main()
