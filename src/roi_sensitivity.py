from __future__ import annotations

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(BASE_DIR, "outputs", "llm_predictions", "test_llm_predictions.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "XGBoost": "churn_decision",
    "Random Forest": None,
    "Qwen3.5-4B": "qwen_decision",
    "Gemma3-4B": "gemma_decision",
}

BASE_RETENTION_COST = 25.0
BASE_ACQUISITION_COST = 125.0
BASE_SUCCESS_RATE = 0.30


def load():
    df = pd.read_csv(PRED_CSV)
    rf = pd.read_csv(os.path.join(BASE_DIR, "outputs", "predictions", "test_predictions_random_forest.csv"))
    df["rf_pred_bin"] = rf["y_pred"].astype(int).to_numpy()
    df["y_true"] = df["y_true"].astype(int)
    df["xgb_pred_bin"] = (df["churn_decision"].str.upper() == "CHURN").astype(int)
    df["qwen_pred_bin"] = (df["qwen_decision"].str.strip().str.lower() == "churn").astype(int)
    df["gemma_pred_bin"] = (df["gemma_decision"].str.strip().str.lower() == "churn").astype(int)
    return df


def net_value(df, pred_col, retention_cost, acquisition_cost, success_rate, rng=None):
    pred = df[pred_col]
    y = df["y_true"]
    ltv = df["Lifetime_Value"]

    tp_mask = (pred == 1) & (y == 1)
    fp_mask = (pred == 1) & (y == 0)
    fn_mask = (pred == 0) & (y == 1)

    n_flagged = int((tp_mask | fp_mask).sum())
    spend = retention_cost * n_flagged

    if rng is None:
        value_saved = success_rate * ltv[tp_mask].sum()
        acquisition_avoided = success_rate * int(tp_mask.sum()) * acquisition_cost
    else:
        tp_idx = np.where(tp_mask)[0]
        n_saved = rng.binomial(len(tp_idx), success_rate)
        saved_idx = rng.choice(tp_idx, size=n_saved, replace=False) if n_saved else np.array([], dtype=int)
        value_saved = ltv.iloc[saved_idx].sum()
        acquisition_avoided = n_saved * acquisition_cost

    missed_loss = ltv[fn_mask].sum()
    return value_saved + acquisition_avoided - spend - missed_loss


def main() -> None:
    df = load()
    pred_cols = {"XGBoost": "xgb_pred_bin", "Random Forest": "rf_pred_bin",
                 "Qwen3.5-4B": "qwen_pred_bin", "Gemma3-4B": "gemma_pred_bin"}

    print("=" * 90)
    print("  Base-case validation (must match published Table 11)")
    print("=" * 90)
    base_values = {}
    for name, col in pred_cols.items():
        v = net_value(df, col, BASE_RETENTION_COST, BASE_ACQUISITION_COST, BASE_SUCCESS_RATE)
        base_values[name] = v
        print(f"  {name:<15} net value = ${v:,.0f}")

    print("\n" + "=" * 90)
    print("  One-way sensitivity (tornado), Gemma3-4B")
    print("=" * 90)
    tornado_rows = []
    for param, lo, hi in [("retention_cost", 5, 100), ("acquisition_multiplier", 2, 10), ("success_rate", 0.05, 0.60)]:
        for bound, val in [("low", lo), ("high", hi)]:
            rc, ac, sr = BASE_RETENTION_COST, BASE_ACQUISITION_COST, BASE_SUCCESS_RATE
            if param == "retention_cost":
                rc = val
            elif param == "acquisition_multiplier":
                ac = val * rc
            elif param == "success_rate":
                sr = val
            v = net_value(df, "gemma_pred_bin", rc, ac, sr)
            tornado_rows.append({"param": param, "bound": bound, "value": val, "net_value": v})
            print(f"  {param:<22} {bound:<5} = {val:<6}  ->  net value = ${v:,.0f}")
    pd.DataFrame(tornado_rows).to_csv(os.path.join(OUT_DIR, "roi_tornado.csv"), index=False)

    print("\n" + "=" * 90)
    print("  Two-way heatmap grid (success_rate x retention_cost) — saved for plotting")
    print("=" * 90)
    heatmap_rows = []
    success_grid = np.round(np.arange(0.05, 0.61, 0.05), 2)
    cost_grid = np.arange(5, 101, 5)
    for name, col in pred_cols.items():
        for sr in success_grid:
            for rc in cost_grid:
                v = net_value(df, col, rc, BASE_ACQUISITION_COST, sr)
                heatmap_rows.append({"model": name, "success_rate": sr, "retention_cost": rc, "net_value": v})
    heatmap_df = pd.DataFrame(heatmap_rows)
    heatmap_df.to_csv(os.path.join(OUT_DIR, "roi_heatmap.csv"), index=False)
    print(f"  Grid: {len(success_grid)} success rates x {len(cost_grid)} costs x {len(pred_cols)} models"
          f" = {len(heatmap_df)} rows")

    pivot = heatmap_df.pivot_table(index=["success_rate", "retention_cost"], columns="model", values="net_value")
    pivot["qwen_wins"] = pivot["Qwen3.5-4B"] > pivot[["XGBoost", "Gemma3-4B"]].max(axis=1)
    frontier = pivot[pivot["qwen_wins"]].reset_index()[["success_rate", "retention_cost"]]
    frontier.to_csv(os.path.join(OUT_DIR, "roi_breakeven_frontier.csv"), index=False)
    print(f"  Qwen overtakes Gemma/XGBoost in {len(frontier)}/{len(pivot)} grid cells "
          f"(low success rate and/or high retention cost)")
    if len(frontier):
        print(f"    e.g. success_rate<={frontier['success_rate'].max()}, "
              f"retention_cost>={frontier['retention_cost'].min()}")

    print("\n" + "=" * 90)
    print("  Monte Carlo (10,000 draws): P(model A > model B), assumption uncertainty")
    print("=" * 90)
    rng = np.random.default_rng(42)
    n_mc = 10_000
    mc_values = {name: np.empty(n_mc) for name in pred_cols}
    for i in range(n_mc):
        rc = rng.uniform(10, 50)
        mult = rng.uniform(3, 7)
        sr = rng.uniform(0.15, 0.45)
        trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        for name, col in pred_cols.items():
            mc_values[name][i] = net_value(df, col, rc, mult * rc, sr, rng=trial_rng)

    mc_df = pd.DataFrame(mc_values)
    mc_df.to_csv(os.path.join(OUT_DIR, "roi_montecarlo.csv"), index=False)
    pairs = [("Gemma3-4B", "XGBoost"), ("XGBoost", "Random Forest"), ("Gemma3-4B", "Qwen3.5-4B")]
    for a, b in pairs:
        p_a_gt_b = (mc_df[a] > mc_df[b]).mean()
        print(f"  P({a} > {b}) = {p_a_gt_b:.1%}")

    print("\n" + "=" * 90)
    print(f"  Base-case Gemma - XGBoost margin: ${base_values['Gemma3-4B'] - base_values['XGBoost']:,.0f}"
          f"  ({(base_values['Gemma3-4B'] - base_values['XGBoost']) / base_values['XGBoost']:.2%} of XGBoost's net value)")
    p_gemma_wins = (mc_df["Gemma3-4B"] > mc_df["XGBoost"]).mean()
    print(f"  Under assumption uncertainty: P(Gemma > XGBoost) = {p_gemma_wins:.1%}  "
          f"-> read as statistical parity, not systematic superiority.")


if __name__ == "__main__":
    main()
