from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NEW_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(NEW_DIR)
sys.path.insert(0, NEW_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import arms
from run_ablation import _load_support_files
from zero_shot_llm_predictions import (
    ChurnZeroShotConfig,
    compute_and_save_metrics,
    load_and_merge,
    run_gemma,
    run_qwen,
    verify_gpu,
)

ROW_IDS_CSV = os.path.join(BASE_DIR, "new", "outputs", "a9_followup_row_ids.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "new", "outputs", "predictions", "A9", "both.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "gemma", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test on the first N rows of whichever sample is active")
    parser.add_argument("--full", action="store_true",
                        help="Run on the full 7,501-row test set instead of the 50-customer "
                             "faithfulness-audit subset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-csv", default=None,
                        help="Default: new/outputs/predictions/A9/both.csv (subset) or "
                             "new/outputs/predictions/A9_full/both.csv (--full)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ChurnZeroShotConfig()
    default_output = os.path.join(BASE_DIR, "new", "outputs", "predictions",
                                   "A9_full" if args.full else "A9", f"{args.model}.csv")
    cfg.output_csv = args.output_csv or default_output
    cfg.output_dir = str(Path(cfg.output_csv).parent)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  A9 follow-up: explicit evidence/context separation instruction"
          f"  [{'FULL 7,501-row test set' if args.full else '50-customer audit subset'}]")
    print("=" * 70)

    verify_gpu()
    _load_support_files(cfg.llm_ready_csv)

    output_path = Path(cfg.output_csv)
    if args.resume and output_path.exists():
        df = pd.read_csv(output_path)
        print(f"Resuming from checkpoint: {output_path} ({len(df)} rows)")
    else:
        full = load_and_merge(cfg)
        full["row_id"] = np.arange(len(full))

        if args.full:
            df = full.reset_index(drop=True)
            print(f"Sample: {len(df)} customers (full test set)")
        else:
            row_ids = pd.read_csv(ROW_IDS_CSV)["row_id"].sort_values().unique()
            df = full[full["row_id"].isin(row_ids)].sort_values("row_id").reset_index(drop=True)
            print(f"Sample: {len(df)} customers (same set as the faithfulness audit)")

        if args.limit:
            df = df.head(args.limit).copy()

    print(f"\nBuilding A9 prompts...")
    prompts = [arms.render_prompt(row, "A9") for _, row in df.iterrows()]
    print(f"Built {len(prompts)} prompts. Example:\n{prompts[0][:500]}...\n")

    if args.model in ("qwen", "both"):
        df = run_qwen(df, prompts, cfg)
    if args.model in ("gemma", "both"):
        df = run_gemma(df, prompts, cfg)

    df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    compute_and_save_metrics(df, cfg)
    print("\n[OK] A9 follow-up complete.")


if __name__ == "__main__":
    main()
