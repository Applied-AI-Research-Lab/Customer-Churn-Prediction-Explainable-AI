from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

NEW_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(NEW_DIR)
sys.path.insert(0, NEW_DIR)

import arms

SAMPLE_CSV = os.path.join(BASE_DIR, "new", "outputs", "ablation_sample.csv")
FULL_RUN_CSV = os.path.join(BASE_DIR, "outputs", "llm_predictions", "test_llm_predictions.csv")
LLM_READY_CSV = os.path.join(BASE_DIR, "outputs", "llm_ready", "llm_ready_xgboost.csv")
OUT_PATH = os.path.join(BASE_DIR, "new", "outputs", "analysis", "ablation_merged.csv")


def _long_rows(row_id, arm, model, decision, explanation, recommendation, time_sec):
    return {
        "arm": arm, "model": model, "row_id": row_id,
        "decision": decision, "explanation": explanation,
        "recommendation": recommendation, "time_sec": time_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default=os.path.join(BASE_DIR, "new", "outputs", "predictions"))
    args = parser.parse_args()

    sample = pd.read_csv(SAMPLE_CSV)
    llm_ready = pd.read_csv(LLM_READY_CSV).reset_index(drop=True)
    llm_ready["row_id"] = np.arange(len(llm_ready))

    long_rows = []

    full = pd.read_csv(FULL_RUN_CSV).reset_index(drop=True)
    full["row_id"] = np.arange(len(full))
    a5 = full[full["row_id"].isin(sample["row_id"])]
    for _, r in a5.iterrows():
        for model, dec_col, exp_col, rec_col, t_col in [
            ("qwen", "qwen_decision", "qwen_explanation", "qwen_recommendation", "qwen_time_sec"),
            ("gemma", "gemma_decision", "gemma_explanation", "gemma_recommendation", "gemma_time_sec"),
        ]:
            long_rows.append(_long_rows(
                r["row_id"], "A5", model, r.get(dec_col), r.get(exp_col), r.get(rec_col), r.get(t_col),
            ))
    print(f"  A5: {len(a5)} rows pulled from existing full run (no GPU job)")

    for arm in arms.GPU_ARMS:
        arm_dir = os.path.join(args.predictions_dir, arm)
        path = os.path.join(arm_dir, "both.csv")
        if not os.path.exists(path):
            qwen_path = os.path.join(arm_dir, "qwen.csv")
            gemma_path = os.path.join(arm_dir, "gemma.csv")
            if os.path.exists(qwen_path) and os.path.exists(gemma_path):
                q = pd.read_csv(qwen_path)
                g = pd.read_csv(gemma_path)
                pdf = q.merge(
                    g[["row_id", "gemma_decision", "gemma_explanation", "gemma_recommendation", "gemma_time_sec"]],
                    on="row_id", how="outer",
                )
            else:
                print(f"  [{arm}] MISSING — expected {path} (or qwen.csv + gemma.csv) — skipping.")
                continue
        else:
            pdf = pd.read_csv(path)

        for _, r in pdf.iterrows():
            for model, dec_col, exp_col, rec_col, t_col in [
                ("qwen", "qwen_decision", "qwen_explanation", "qwen_recommendation", "qwen_time_sec"),
                ("gemma", "gemma_decision", "gemma_explanation", "gemma_recommendation", "gemma_time_sec"),
            ]:
                if dec_col not in pdf.columns:
                    continue
                long_rows.append(_long_rows(
                    r["row_id"], arm, model, r.get(dec_col), r.get(exp_col), r.get(rec_col), r.get(t_col),
                ))
        print(f"  {arm}: {len(pdf)} rows loaded from {path if os.path.exists(path) else arm_dir}")

    merged = pd.DataFrame(long_rows)

    meta = sample[["row_id", "stratum", "weight", "y_true", "risk_tier"]]
    merged = merged.merge(meta, on="row_id", how="left")
    merged = merged.merge(
        llm_ready[["row_id", "churn_decision"]].rename(columns={"churn_decision": "xgb_decision"}),
        on="row_id", how="left",
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"\n  Merged table saved -> {OUT_PATH}  ({len(merged)} rows, {merged['arm'].nunique()} arms)")
    print("\n  Rows per arm x model:")
    print(merged.groupby(["arm", "model"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
