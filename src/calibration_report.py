from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_CSV = os.path.join(BASE_DIR, "new", "outputs", "calibration", "test_calibrated_probabilities.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def reliability_bins(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        conf = float(y_prob[mask].mean()) if n else float("nan")
        acc = float(y_true[mask].mean()) if n else float("nan")
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": n, "mean_predicted": conf, "observed_freq": acc})
    return pd.DataFrame(rows)


def ece_mce(bins_df: pd.DataFrame, n_total: int):
    valid = bins_df.dropna(subset=["mean_predicted", "observed_freq"])
    gaps = (valid["mean_predicted"] - valid["observed_freq"]).abs()
    ece = (valid["n"] / n_total * gaps).sum()
    mce = gaps.max() if len(gaps) else float("nan")
    return ece, mce


def main() -> None:
    df = pd.read_csv(CALIB_CSV)
    y_true = df["y_true"].to_numpy()
    n = len(df)

    print("=" * 90)
    print("  Calibration report: Brier / ECE / MCE, pre- and post-calibration")
    print("=" * 90)
    summary = []
    for label, col in [("raw", "y_proba_raw"), ("isotonic", "y_proba_isotonic"), ("platt", "y_proba_platt")]:
        probs = df[col].to_numpy()
        bins_df = reliability_bins(y_true, probs)
        bins_df.insert(0, "scale", label)
        bins_df.to_csv(os.path.join(OUT_DIR, f"reliability_bins_{label}.csv"), index=False)

        brier = brier_score_loss(y_true, probs)
        ece, mce = ece_mce(bins_df, n)
        n_ambig = int(((probs >= 0.4) & (probs < 0.8)).sum())
        summary.append({"scale": label, "brier": brier, "ece": ece, "mce": mce, "n_ambiguous_0.4_0.8": n_ambig})
        print(f"  {label:<10} Brier={brier:.4f}  ECE={ece:.4f}  MCE={mce:.4f}  n[0.4,0.8)={n_ambig}")

    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "calibration_summary.csv"), index=False)

    raw = df["y_proba_raw"].to_numpy()
    mask = (raw >= 0.6) & (raw < 0.8)
    print(f"\n  Raw XGBoost in [0.6,0.8): predicts {raw[mask].mean():.3f}, observes {y_true[mask].mean():.3f}"
          f"  (over-confident by {raw[mask].mean() - y_true[mask].mean():+.3f})")
    print(f"  Saved reliability bins + summary -> {OUT_DIR}")


if __name__ == "__main__":
    main()
