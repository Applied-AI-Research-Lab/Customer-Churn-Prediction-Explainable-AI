"""
Merge parallel Qwen and Gemma prediction CSVs into a single output file.

Usage (after both GPU runs complete):
    python src/merge_llm_predictions.py

Or with custom paths:
    python src/merge_llm_predictions.py \
        --qwen  outputs/llm_predictions/qwen_predictions.csv \
        --gemma outputs/llm_predictions/gemma_predictions.csv \
        --out   outputs/llm_predictions/test_llm_predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report,
)


QWEN_COLS  = ["qwen_decision",  "qwen_explanation",  "qwen_recommendation",  "qwen_time_sec"]
GEMMA_COLS = ["gemma_decision", "gemma_explanation", "gemma_recommendation", "gemma_time_sec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge parallel LLM prediction CSVs")
    parser.add_argument("--qwen",  default="outputs/llm_predictions/qwen_predictions.csv")
    parser.add_argument("--gemma", default="outputs/llm_predictions/gemma_predictions.csv")
    parser.add_argument("--out",   default="outputs/llm_predictions/test_llm_predictions.csv")
    return parser.parse_args()


def compute_metrics(df: pd.DataFrame, prefix: str) -> dict:
    col = f"{prefix}_decision"
    mask = df[col].notna() & ~df[col].str.startswith("[parse_error]", na=False)
    sub  = df[mask]
    if sub.empty:
        return {}
    y_pred = (sub[col].str.strip().str.lower() == "churn").astype(int)
    y_true = sub["y_true"].astype(int)
    return {
        "model":        prefix,
        "n_evaluated":  len(sub),
        "accuracy":     round(accuracy_score(y_true, y_pred), 4),
        "precision":    round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":       round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":           round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":      round(roc_auc_score(y_true, y_pred), 4),
        "avg_time_sec": round(df[f"{prefix}_time_sec"].mean(), 2),
    }


def main() -> None:
    args = parse_args()
    qwen_path  = Path(args.qwen)
    gemma_path = Path(args.gemma)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load both files ────────────────────────────────────────────────────
    if not qwen_path.exists():
        raise FileNotFoundError(f"Qwen predictions not found: {qwen_path}")
    if not gemma_path.exists():
        raise FileNotFoundError(f"Gemma predictions not found: {gemma_path}")

    qwen_df  = pd.read_csv(qwen_path)
    gemma_df = pd.read_csv(gemma_path)

    print(f"Qwen  file : {qwen_path}  ({len(qwen_df)} rows)")
    print(f"Gemma file : {gemma_path}  ({len(gemma_df)} rows)")

    if len(qwen_df) != len(gemma_df):
        raise ValueError(
            f"Row count mismatch: Qwen={len(qwen_df)}, Gemma={len(gemma_df)}. "
            "Both must be run on the same test set."
        )

    # ── Verify row alignment ───────────────────────────────────────────────
    if not qwen_df["y_true"].equals(gemma_df["y_true"]):
        raise ValueError("y_true columns differ — files are not from the same test set.")

    # ── Merge: base columns from Qwen file + Gemma LLM columns ────────────
    merged = qwen_df.copy()
    for col in GEMMA_COLS:
        if col in gemma_df.columns:
            merged[col] = gemma_df[col].values
        else:
            merged[col] = None

    merged.to_csv(out_path, index=False)
    print(f"\nMerged file saved → {out_path}")
    print(f"Shape: {merged.shape[0]} rows × {merged.shape[1]} columns")

    # ── Metrics ────────────────────────────────────────────────────────────
    rows = []
    print(f"\n{'='*60}")
    print("  FINAL METRICS — Both Models")
    print(f"{'='*60}")
    for prefix in ("qwen", "gemma"):
        m = compute_metrics(merged, prefix)
        if not m:
            continue
        rows.append(m)
        print(f"\n  [{prefix.upper()}]")
        for k, v in m.items():
            if k != "model":
                print(f"    {k:>14}: {v}")
        print(f"\n  Classification Report ({prefix.upper()}):")
        col = f"{prefix}_decision"
        mask = merged[col].notna() & ~merged[col].str.startswith("[parse_error]", na=False)
        sub  = merged[mask]
        y_pred = (sub[col].str.strip().str.lower() == "churn").astype(int)
        y_true = sub["y_true"].astype(int)
        print(classification_report(y_true, y_pred,
                                    target_names=["Retained", "Churned"], digits=4))

    if rows:
        metrics_df = pd.DataFrame(rows)
        metrics_path = out_path.parent / "llm_zs_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n  Metrics saved → {metrics_path}")

    print("\n✅ Merge complete.")


if __name__ == "__main__":
    main()
