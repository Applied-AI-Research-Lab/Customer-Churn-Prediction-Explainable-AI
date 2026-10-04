from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

NEW_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(NEW_DIR)
sys.path.insert(0, NEW_DIR)

import arms
import prompt_blocks
import _reference_build_prompt as reference

LLM_READY_CSV = os.path.join(BASE_DIR, "outputs", "llm_ready", "llm_ready_xgboost.csv")
RF_PRED_CSV = os.path.join(BASE_DIR, "outputs", "predictions", "test_predictions_random_forest.csv")
PILOT_DIR = os.path.join(BASE_DIR, "new", "outputs", "pilot")


def _load_and_merge() -> pd.DataFrame:
    llm_ready = pd.read_csv(LLM_READY_CSV)
    rf_preds = pd.read_csv(RF_PRED_CSV)
    rf_preds = rf_preds.rename(columns={"y_pred": "rf_pred", "y_proba": "rf_proba"}).drop(
        columns=["y_true"], errors="ignore"
    )
    return pd.concat([llm_ready.reset_index(drop=True), rf_preds.reset_index(drop=True)], axis=1)


def _load_support_files_for_check() -> None:
    llm_dir = Path(LLM_READY_CSV).parent
    rf_importances, benchmarks = {}, {}
    rf_path = llm_dir / "rf_feature_importances.json"
    if rf_path.exists():
        rf_importances = json.load(open(rf_path))
    bench_path = llm_dir / "population_benchmarks.json"
    if bench_path.exists():
        benchmarks = json.load(open(bench_path))
    prompt_blocks.set_support_files(rf_importances, benchmarks)
    reference.set_support_files(rf_importances, benchmarks)


def check_parity(n_rows: int = 100) -> bool:
    print("=" * 70)
    print("  Prompt-parity check: arms.render_prompt(row, 'A5') vs. reference build_prompt()")
    print(f"  ({n_rows} sample rows, CPU only, no GPU)")
    print("=" * 70)

    _load_support_files_for_check()
    df = _load_and_merge()
    sample = df.sample(n=min(n_rows, len(df)), random_state=42)

    mismatches = []
    for idx, row in sample.iterrows():
        original = reference.build_prompt(row)
        refactored = arms.render_prompt(row, "A5")
        if original != refactored:
            mismatches.append(idx)

    if not mismatches:
        print(f"  PASS — all {len(sample)} sampled rows byte-identical.")
        return True

    print(f"  FAIL — {len(mismatches)}/{len(sample)} rows differ. First mismatch (row {mismatches[0]}):")
    o = reference.build_prompt(df.loc[mismatches[0]])
    r = arms.render_prompt(df.loc[mismatches[0]], "A5")
    for i, (co, cr) in enumerate(zip(o, r)):
        if co != cr:
            print(f"    First differing char at position {i}:")
            print(f"    original  : ...{o[max(0, i-40):i+40]!r}...")
            print(f"    refactored: ...{r[max(0, i-40):i+40]!r}...")
            break
    else:
        print(f"    Length mismatch: original={len(o)} chars, refactored={len(r)} chars")
    return False


def check_pilot() -> bool:
    print("=" * 70)
    print("  Post-pilot Gate 0 checks (50-row pilot, all 8 arms)")
    print("=" * 70)

    all_pass = True
    per_arm_decisions = {}

    for arm in arms.GPU_ARMS:
        path = os.path.join(PILOT_DIR, arm, "both.csv")
        if not os.path.exists(path):
            print(f"  [{arm}] MISSING — expected {path}")
            all_pass = False
            continue
        pdf = pd.read_csv(path)
        for prefix in ("qwen", "gemma"):
            col = f"{prefix}_decision"
            if col not in pdf.columns:
                continue
            n = len(pdf)
            n_parsed = int((pdf[col].notna() & ~pdf[col].astype(str).str.startswith("[parse_error]")).sum())
            rate = n_parsed / n if n else 0.0
            status = "OK" if rate >= 0.95 else "FAIL"
            if rate < 0.95:
                all_pass = False
            print(f"  [{arm}/{prefix}] parse rate = {rate:.1%} ({n_parsed}/{n})  [{status}]")
            per_arm_decisions[(arm, prefix)] = pdf.set_index("row_id")[col] if "row_id" in pdf.columns else pdf[col]

    print()
    if ("A0", "qwen") in per_arm_decisions and ("A4", "qwen") in per_arm_decisions:
        a0 = per_arm_decisions[("A0", "qwen")]
        a4 = per_arm_decisions[("A4", "qwen")]
        common = a0.index.intersection(a4.index) if hasattr(a0, "index") else range(min(len(a0), len(a4)))
        disagree = sum(a0[i] != a4[i] for i in common)
        print(f"  A0 vs A4 (Qwen) disagreement on pilot: {disagree}/{len(list(common))} rows "
              f"({'OK — arms differ' if disagree > 0 else 'FAIL — arms produce identical output'})")
        if disagree == 0:
            all_pass = False

    print("\n  " + ("PASS — safe to launch the full ~15.6h run." if all_pass
                     else "FAIL — investigate before spending the full GPU budget."))
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=["parity", "pilot"], required=True)
    parser.add_argument("--n-rows", type=int, default=100)
    args = parser.parse_args()

    ok = check_parity(args.n_rows) if args.check == "parity" else check_pilot()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
