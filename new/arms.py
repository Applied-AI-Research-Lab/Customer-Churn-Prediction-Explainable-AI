
from __future__ import annotations

import pandas as pd

from prompt_blocks import (
    JSON_SPEC_GENERIC,
    JSON_SPEC_PANEL,
    SYSTEM_PREAMBLE,
    SYSTEM_PREAMBLE_NO_PANEL,
    agreement_block,
    behaviour_block,
    benchmarks_block,
    instruction_block,
    override_guard_block,
    profile_block,
    rf_panel_block,
    shap_block,
    xgb_panel_block,
    xgb_verdict_only_block,
)

GPU_ARMS = ["A0", "A1", "A2", "A3", "A4", "A6", "A7", "A8", "A9"]
ALL_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]

PANEL_HEADER = "━━━ ADVISORY PANEL SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
RULE = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"


def _panel_style_prompt(
    row: pd.Series,
    *,
    include_rf: bool,
    include_rf_importances: bool,
    include_override_guard: bool,
    invert: bool,
    instruction_variant: str = "panel",
) -> str:
    parts = [SYSTEM_PREAMBLE]
    parts.append(profile_block(row) + "\n")
    parts.append(behaviour_block(row) + "\n")
    bench = benchmarks_block(row)
    if bench:
        parts.append(bench + "\n")
    parts.append(PANEL_HEADER)
    parts.append(xgb_panel_block(row, invert=invert) + "\n")
    if include_rf:
        parts.append(rf_panel_block(row, include_importances=include_rf_importances, invert=invert) + "\n")
        parts.append(agreement_block(row, invert=invert))
    parts.append(RULE)
    parts.append(instruction_block(instruction_variant))
    if include_override_guard:
        parts.append(override_guard_block(row, invert=invert))
    parts.append("\n" + JSON_SPEC_PANEL)
    return "".join(parts)


def _no_panel_prompt(row: pd.Series, arm_id: str) -> str:
    parts = [SYSTEM_PREAMBLE_NO_PANEL]
    if arm_id == "A3":
        parts.append(xgb_verdict_only_block(row) + "\n")
        parts.append(instruction_block("prediction_only"))
        parts.append("\n" + JSON_SPEC_GENERIC)
        return "".join(parts)

    parts.append(profile_block(row) + "\n")
    parts.append(behaviour_block(row) + "\n")
    if arm_id in ("A1", "A2"):
        bench = benchmarks_block(row)
        if bench:
            parts.append(bench + "\n")
    if arm_id == "A2":
        parts.append(shap_block(row) + "\n")

    variant = {"A0": "features_only", "A1": "features_benchmarks", "A2": "features_shap"}[arm_id]
    parts.append(instruction_block(variant))
    parts.append("\n" + JSON_SPEC_GENERIC)
    return "".join(parts)


def render_prompt(row: pd.Series, arm_id: str) -> str:
    if arm_id not in ALL_ARMS:
        raise ValueError(f"Unknown arm '{arm_id}' — must be one of {ALL_ARMS}")

    if arm_id in ("A0", "A1", "A2", "A3"):
        return _no_panel_prompt(row, arm_id)

    if arm_id == "A4":
        return _panel_style_prompt(
            row, include_rf=True, include_rf_importances=True,
            include_override_guard=False, invert=False,
        )
    if arm_id == "A5":
        return _panel_style_prompt(
            row, include_rf=True, include_rf_importances=True,
            include_override_guard=True, invert=False,
        )
    if arm_id == "A6":
        return _panel_style_prompt(
            row, include_rf=False, include_rf_importances=False,
            include_override_guard=True, invert=False,
        )
    if arm_id == "A7":
        return _panel_style_prompt(
            row, include_rf=True, include_rf_importances=False,
            include_override_guard=True, invert=False,
        )
    if arm_id == "A8":
        return _panel_style_prompt(
            row, include_rf=True, include_rf_importances=True,
            include_override_guard=True, invert=True,
        )
    if arm_id == "A9":
        return _panel_style_prompt(
            row, include_rf=True, include_rf_importances=True,
            include_override_guard=True, invert=False,
            instruction_variant="panel_strict",
        )
    raise AssertionError(f"unhandled arm {arm_id}")
