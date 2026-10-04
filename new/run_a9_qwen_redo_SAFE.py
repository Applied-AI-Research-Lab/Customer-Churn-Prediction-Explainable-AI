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
    run_qwen,
    verify_gpu,
)

OUTPUT_CSV = os.path.join(BASE_DIR, "new", "outputs", "predictions", "A9_qwen_redo", "qwen_only.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test on the first N rows")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ChurnZeroShotConfig()
    cfg.output_csv = OUTPUT_CSV
    cfg.output_dir = str(Path(cfg.output_csv).parent)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  QWEN-ONLY REDO (arm A9, full 7,501-row test set)")
    print(f"  Output: {cfg.output_csv}")
    print("  This path is dedicated to this script only -- it cannot collide")
    print("  with the Gemma A9_full job, which writes to a different directory.")
    print("=" * 70)

    verify_gpu()
    _load_support_files(cfg.llm_ready_csv)

    output_path = Path(cfg.output_csv)
    if args.resume and output_path.exists():
        df = pd.read_csv(output_path)
        print(f"Resuming from checkpoint: {output_path} ({len(df)} rows)")
    else:
        df = load_and_merge(cfg)
        df["row_id"] = np.arange(len(df))
        print(f"Sample: {len(df)} customers (full test set)")
        if args.limit:
            df = df.head(args.limit).copy()

    print("\nBuilding A9 prompts...")
    prompts = [arms.render_prompt(row, "A9") for _, row in df.iterrows()]
    print(f"Built {len(prompts)} prompts.\n")

    df = run_qwen(df, prompts, cfg)

    df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    compute_and_save_metrics(df, cfg)
    print("\n[OK] Qwen-only A9 redo complete.")


if __name__ == "__main__":
    main()
