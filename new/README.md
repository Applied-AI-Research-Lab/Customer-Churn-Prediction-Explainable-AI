# Ablation grid

Nine-condition (A0–A9) zero-shot prompt-ablation grid testing whether SLM
decisions reflect independent reasoning or imitation of the supplied
classifier verdict, run on a frozen, stratified sample of the test set.

This directory does not modify `src/` — the production pipeline that
generated the paper's published numbers stays untouched. Code here imports
the model-loading/generation/parsing logic from
`src/zero_shot_llm_predictions.py` rather than duplicating it.

## Files

| File | Role |
|---|---|
| `prompt_blocks.py` | Decomposed prompt fragments (profile, behaviour, benchmarks, SHAP, XGB/RF panels, agreement, override guard, instructions) |
| `arms.py` | The arm registry + `render_prompt(row, arm_id)` |
| `calibrate.py` | Fits isotonic/Platt calibrators on the validation split, applies to test |
| `build_ablation_sample.py` | Freezes the stratified sample + a built-in validation check |
| `run_ablation.py` | Runner for one arm (imports `run_qwen`/`run_gemma` from `src/`) |
| `validate_pilot.py` | Prompt-parity check + post-pilot checks |
| `merge_ablation_results.py` | Merges all arms' outputs into one long table |
| `run_a9_followup.py`, `run_a9_qwen_redo_SAFE.py` | A9 fabrication-mitigation arm and its recovery/redo run |
| `analyze_ablation.py`, `analyze_a9_followup.py` | Decomposition, anchoring-index, and fabrication-rate analysis |

## Arm reference

| Arm | Tests |
|---|---|
| A0 | raw features only |
| A1 | features + population benchmarks |
| A2 | features + SHAP evidence, no classifier verdict |
| A3 | classifier verdict + probability only |
| A4 | full panel, no override guard |
| A5 | full panel (the published prompt) |
| A6 | XGBoost-only panel, no RF |
| A7 | full panel, RF global importances removed |
| A8 | full panel, both classifiers' verdicts deliberately inverted (anchoring control) |
| A9 | full panel, plus one instruction restricting the model to its numbered SHAP evidence (fabrication mitigation) |
