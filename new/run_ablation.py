from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NEW_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(NEW_DIR)
sys.path.insert(0, NEW_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import arms
import prompt_blocks
from zero_shot_llm_predictions import (
    ChurnZeroShotConfig,
    compute_and_save_metrics,
    load_and_merge,
    run_gemma,
    run_qwen,
    verify_gpu,
)

DEFAULT_SAMPLE_CSV = os.path.join(BASE_DIR, "new", "outputs", "ablation_sample.csv")


def _load_support_files(llm_ready_csv: str) -> None:
    llm_dir = Path(llm_ready_csv).parent
    rf_importances, benchmarks = {}, {}

    rf_path = llm_dir / "rf_feature_importances.json"
    if rf_path.exists():
        with open(rf_path) as f:
            rf_importances = json.load(f)
        print(f"  Loaded RF importances ({len(rf_importances)} features) from {rf_path}")
    else:
        print(f"  WARNING: {rf_path} not found")

    bench_path = llm_dir / "population_benchmarks.json"
    if bench_path.exists():
        with open(bench_path) as f:
            benchmarks = json.load(f)
        print(f"  Loaded population benchmarks ({len(benchmarks)} features) from {bench_path}")
    else:
        print(f"  WARNING: {bench_path} not found")

    prompt_blocks.set_support_files(rf_importances, benchmarks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ablation arm (Qwen and/or Gemma) on the frozen sample")
    parser.add_argument("--arm", required=True, choices=arms.GPU_ARMS,
                        help="Which ablation arm to run (A5 needs no GPU job — see new/README.md)")
    parser.add_argument("--model", choices=["qwen", "gemma", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows of the frozen sample, in row_id order "
                             "(pilot mode — same N rows across every arm for comparability)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-csv", default=DEFAULT_SAMPLE_CSV)
    parser.add_argument("--llm-ready-csv", default=None)
    parser.add_argument("--rf-pred-csv", default=None)
    parser.add_argument("--output-csv", default=None,
                        help="Default: new/outputs/predictions/{arm}/{model}.csv "
                             "(one subdirectory per arm — compute_and_save_metrics() "
                             "always writes '<dir>/llm_zs_metrics.csv', so arms must "
                             "not share an output directory or concurrent runs collide)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ChurnZeroShotConfig()
    if args.llm_ready_csv:
        cfg.llm_ready_csv = args.llm_ready_csv
    if args.rf_pred_csv:
        cfg.rf_pred_csv = args.rf_pred_csv

    output_csv = args.output_csv or os.path.join(
        BASE_DIR, "new", "outputs", "predictions", args.arm, f"{args.model}.csv"
    )
    cfg.output_csv = output_csv
    cfg.output_dir = str(Path(output_csv).parent)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Ablation arm {args.arm} — model={args.model}  host={socket.gethostname()}"
          f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")
    print("=" * 70)

    verify_gpu()
    print("\nLoading support files (RF importances + benchmarks)...")
    _load_support_files(cfg.llm_ready_csv)

    output_path = Path(cfg.output_csv)
    if args.resume and output_path.exists():
        df = pd.read_csv(output_path)
        print(f"\nResuming from checkpoint: {output_path} ({len(df)} rows)")
    else:
        print("\nLoading full merged test set + frozen ablation sample...")
        full = load_and_merge(cfg)
        full["row_id"] = np.arange(len(full))

        sample = pd.read_csv(args.sample_csv)
        row_ids = sample["row_id"].sort_values()
        if args.limit:
            row_ids = row_ids.head(args.limit)

        df = full[full["row_id"].isin(row_ids)].sort_values("row_id").reset_index(drop=True)
        df = df.merge(sample[["row_id", "stratum", "weight"]], on="row_id", how="left")
        print(f"Sample rows selected: {len(df)}"
              + (f" (--limit {args.limit}, pilot mode)" if args.limit else ""))

    print(f"\nTotal rows to process: {len(df)}")
    print(f"Churn distribution (y_true): {df['y_true'].value_counts().to_dict()}")

    print(f"\nBuilding '{args.arm}' prompts...")
    prompts = [arms.render_prompt(row, args.arm) for _, row in df.iterrows()]
    print(f"Built {len(prompts)} prompts. Example (row_id={df['row_id'].iloc[0]}):\n")
    print(prompts[0][:600] + ("..." if len(prompts[0]) > 600 else ""))

    if args.model in ("qwen", "both"):
        df = run_qwen(df, prompts, cfg)
    if args.model in ("gemma", "both"):
        df = run_gemma(df, prompts, cfg)

    df.to_csv(output_path, index=False)
    print(f"\nFinal output saved -> {output_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    compute_and_save_metrics(df, cfg)
    print(f"\n[OK] Arm {args.arm} ({args.model}) complete.")


if __name__ == "__main__":
    main()
