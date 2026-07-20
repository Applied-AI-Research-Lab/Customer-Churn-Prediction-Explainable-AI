#!/usr/bin/env python3
"""
Zero-Shot LLM Churn Explainability Layer
=========================================
Reads the LLM-ready test file (XGBoost SHAP signals + raw features) and
the Random Forest test predictions, builds a structured advisory-panel
prompt per customer row, and passes it to both SLMs in zero-shot mode.

Each SLM returns a JSON object:
  • decision      — "churn" or "retain"
  • explanation   — 2-3 sentences referencing panel signals and SHAP reasons
  • recommendation — 1-2 sentences of actionable advice for the business owner

Models (Unsloth, 4-bit):
  • unsloth/Qwen3.5-4B      → FastLanguageModel  (thinking mode disabled)
  • unsloth/gemma-3-4b-it   → FastModel

New columns written to outputs/llm_predictions/test_llm_predictions.csv:
  qwen_decision, qwen_explanation, qwen_recommendation, qwen_time_sec
  gemma_decision, gemma_explanation, gemma_recommendation, gemma_time_sec

Usage (GPU server venv):
    python src/zero_shot_llm_predictions.py
    python src/zero_shot_llm_predictions.py --limit 100          # smoke test
    python src/zero_shot_llm_predictions.py --model qwen         # only Qwen
    python src/zero_shot_llm_predictions.py --model gemma        # only Gemma
    python src/zero_shot_llm_predictions.py --resume             # continue from checkpoint

Requires: unsloth, torch (CUDA), transformers, pandas, tqdm
"""

from __future__ import annotations

# ── Force-set HF env vars BEFORE any huggingface import ──────────────────────
# HF_HUB_DISABLE_XET=1  : disables the XET protocol which stalls on some servers
# HF_TOKEN              : set via setenv in tcsh before running the script:
#   setenv HF_TOKEN <your_hf_token>
#   setenv HF_HUB_DISABLE_XET 1
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

# ── Disable torch.compile / dynamo for inference ─────────────────────────────
# Qwen3.5-4B uses a hybrid GatedDeltaNet architecture whose causal_conv1d_update
# layer triggers endless recompilations as prompt lengths vary across rows,
# eventually hitting accumulated_cache_size_limit.
# torch.compile provides zero benefit for inference — disabling it is safe.
import torch._dynamo
torch._dynamo.config.disable = True

from tqdm import tqdm

# Support-file globals (populated in main via _load_support_files)
_RF_IMPORTANCES: dict = {}
_BENCHMARKS: dict = {}

ALLOWED_DECISIONS = {"churn", "retain"}
SAVE_EVERY = 10          # checkpoint to disk every N processed rows


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (mirrors old-project ZeroShotConfig)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChurnZeroShotConfig:
    # Input paths (relative to project root)
    llm_ready_csv: str  = "outputs/llm_ready/llm_ready_xgboost.csv"
    rf_pred_csv: str    = "outputs/predictions/test_predictions_random_forest.csv"

    # Output
    output_dir: str     = "outputs/llm_predictions"
    output_csv: str     = "outputs/llm_predictions/test_llm_predictions.csv"

    # Model identifiers — same as old project
    qwen_model_name: str  = "unsloth/Qwen3.5-4B"
    gemma_model_name: str = "unsloth/gemma-3-4b-it"

    max_seq_length: int  = 2048
    load_in_4bit: bool   = True

    # Generation — Qwen
    # 512 → 350: smoke test shows responses complete well within 350 tokens
    qwen_max_new_tokens: int  = 350
    qwen_temperature: float   = 0.3
    qwen_top_p: float         = 0.9

    # Generation — Gemma
    # 300 tokens: smoke test responses need ~225 tokens (explanation ~550 chars +
    # recommendation ~300 chars + JSON overhead). 180 caused truncation mid-JSON.
    gemma_max_new_tokens: int = 300
    gemma_temperature: float  = 0.7
    gemma_top_p: float        = 0.95
    gemma_top_k: int          = 64

    generation_retry_attempts: int = 4


# ─────────────────────────────────────────────────────────────────────────────
# GPU verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_gpu() -> None:
    print("=" * 60)
    print("  GPU Verification")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("  CUDA NOT AVAILABLE — this script requires a GPU.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"  GPU:          {gpu_name}")
    print(f"  VRAM:         {gpu_mem:.1f} GB")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  PyTorch:      {torch.__version__}")
    # Smoke test
    _ = torch.matmul(torch.randn(512, 512, device="cuda"),
                     torch.randn(512, 512, device="cuda"))
    torch.cuda.synchronize()
    print("  GPU smoke test: OK")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder  (advisory-panel style — same pattern as old project)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 2) -> str:
    """Format a possibly-NaN numeric value for display."""
    try:
        f = float(val)
        if f != f:          # NaN check
            return "N/A"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(val) if pd.notna(val) else "N/A"


def build_prompt(row: pd.Series) -> str:
    """Construct the advisory-panel prompt for a single customer row."""
    # ── Customer profile ──────────────────────────────────────────────────────
    age        = _fmt(row.get("Age"), 0)
    gender     = str(row.get("Gender", "N/A"))
    country    = str(row.get("Country", "N/A"))
    mem_yrs    = _fmt(row.get("Membership_Years"))
    ltv        = _fmt(row.get("Lifetime_Value"))
    credit_bal = _fmt(row.get("Credit_Balance"))
    signup_q   = str(row.get("Signup_Quarter", "N/A"))

    # ── Behavioral signals ────────────────────────────────────────────────────
    login_freq = _fmt(row.get("Login_Frequency"), 0)
    session    = _fmt(row.get("Session_Duration_Avg"))
    cart_ab    = _fmt(row.get("Cart_Abandonment_Rate"))
    days_last  = _fmt(row.get("Days_Since_Last_Purchase"), 0)
    cs_calls   = _fmt(row.get("Customer_Service_Calls"), 0)
    email_open = _fmt(row.get("Email_Open_Rate"))
    soc_media  = _fmt(row.get("Social_Media_Engagement_Score"))
    returns    = _fmt(row.get("Returns_Rate"))
    total_pur  = _fmt(row.get("Total_Purchases"), 0)
    avg_order  = _fmt(row.get("Average_Order_Value"))
    mobile_app = _fmt(row.get("Mobile_App_Usage"))
    discount   = _fmt(row.get("Discount_Usage_Rate"))

    # ── Model predictions ─────────────────────────────────────────────────────
    xgb_pred   = str(row.get("churn_decision", "N/A")).upper()
    xgb_prob   = _fmt(float(row.get("y_proba", 0)) * 100, 1)

    rf_pred_raw = row.get("rf_pred", None)
    rf_proba    = row.get("rf_proba", None)
    if pd.notna(rf_pred_raw):
        rf_decision = "CHURN" if int(rf_pred_raw) == 1 else "RETAIN"
        rf_prob     = _fmt(float(rf_proba) * 100, 1) if pd.notna(rf_proba) else "N/A"
    else:
        rf_decision, rf_prob = "N/A", "N/A"

    # ── SHAP top-5 reasons with magnitudes ────────────────────────────────────
    reasons = [str(row.get(f"reason_{k}", "N/A")) for k in range(1, 6)]
    r_lines = "\n".join(f"    SHAP reason {k}   : {r}" for k, r in enumerate(reasons, 1) if r and r != "N/A")

    # ── Risk tier ─────────────────────────────────────────────────────────────
    risk_tier = str(row.get("risk_tier", "N/A"))

    # ── RF global feature importances (top 5) ─────────────────────────────────
    if _RF_IMPORTANCES:
        rf_top5 = list(_RF_IMPORTANCES.items())[:5]
        rf_imp_lines = "  ".join(
            f"{feat.split('_')[0] if len(feat) > 25 else feat}={imp:.3f}"
            for feat, imp in rf_top5
        )
    else:
        rf_imp_lines = "N/A"

    # ── Population benchmarks for top SHAP features ───────────────────────────
    bench_lines = ""
    if _BENCHMARKS:
        # Find which original features correspond to the top SHAP reasons
        bench_feats = []
        for r in reasons[:3]:
            for feat in _BENCHMARKS:
                desc = feat.replace("_", " ").lower()
                if desc in r.lower() or feat.lower() in r.lower():
                    if feat not in bench_feats:
                        bench_feats.append(feat)
                    break
        if not bench_feats:          # fallback: top-3 by overall importance
            bench_feats = list(_BENCHMARKS.keys())[:3]
        bench_parts = []
        for feat in bench_feats[:3]:
            b = _BENCHMARKS[feat]
            cust_val = row.get(feat, None)
            cust_str = _fmt(cust_val) if pd.notna(cust_val) else "N/A"
            bench_parts.append(
                f"  {feat}: customer={cust_str} | avg churner={b['churned_mean']} "
                f"| avg retained={b['retained_mean']}"
            )
        bench_lines = "\n".join(bench_parts)

    # ── Agreement status ──────────────────────────────────────────────────────
    preds = [xgb_pred, rf_decision]
    unique_preds = set(p for p in preds if p not in ("N/A",))
    if len(unique_preds) == 1:
        agreement = f"Full agreement — both models predict {list(unique_preds)[0]}"
    else:
        agreement = f"Disagreement — XGBoost predicts {xgb_pred}, Random Forest predicts {rf_decision}"

    bench_section = (
        "━━━ POPULATION BENCHMARKS (training set) ━━━━━━━━━━━━━━━━━\n"
        f"{bench_lines}\n\n"
    ) if bench_lines else ""

    return (
        "You are a senior customer retention analyst reviewing the output of a "
        "two-model advisory panel that has already analyzed the customer below.\n\n"
        "━━━ CUSTOMER PROFILE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Age              : {age}\n"
        f"  Gender           : {gender}\n"
        f"  Country          : {country}\n"
        f"  Signup quarter   : {signup_q}\n"
        f"  Membership years : {mem_yrs}\n"
        f"  Lifetime value   : ${ltv}\n"
        f"  Credit balance   : ${credit_bal}\n\n"
        "━━━ BEHAVIORAL SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Login frequency             : {login_freq} logins/period\n"
        f"  Avg session duration        : {session} min\n"
        f"  Days since last purchase    : {days_last}\n"
        f"  Total purchases             : {total_pur}\n"
        f"  Avg order value             : ${avg_order}\n"
        f"  Cart abandonment rate       : {cart_ab}%\n"
        f"  Returns rate                : {returns}%\n"
        f"  Discount usage rate         : {discount}%\n"
        f"  Email open rate             : {email_open}%\n"
        f"  Social media engagement     : {soc_media}\n"
        f"  Mobile app usage            : {mobile_app}\n"
        f"  Customer service calls      : {cs_calls}\n\n"
        f"{bench_section}"
        "━━━ ADVISORY PANEL SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "[1] XGBOOST  (gradient-boosted tree expert — best model, ROC-AUC 0.927)\n"
        f"    Prediction       : {xgb_pred}\n"
        f"    Churn probability : {xgb_prob}%   |   Risk tier: {risk_tier}\n"
        f"{r_lines}\n\n"
        "[2] RANDOM FOREST  (bagging ensemble expert — ROC-AUC 0.921)\n"
        f"    Prediction       : {rf_decision}\n"
        f"    Churn probability : {rf_prob}%\n"
        f"    Top global features (importance): {rf_imp_lines}\n\n"
        f"Panel status : {agreement}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Based on all panel signals, determine the correct final churn decision, "
        "explain your reasoning clearly by referencing the model predictions, SHAP "
        "signals (magnitude labels indicate signal strength), population benchmarks, "
        "and behavioral data, and provide an actionable recommendation for "
        "the business owner on how to retain this customer or confirm churn.\n"
        + (
            "IMPORTANT: Both models agree on CHURN with ≥80% confidence. "
            "Override to 'retain' ONLY if there is clear overwhelming counter-evidence "
            "in the behavioral signals.\n"
            if (xgb_pred == "CHURN" and rf_decision == "CHURN"
                and float(row.get("y_proba", 0)) >= 0.80)
            else ""
        )
        + "\nRespond with ONLY a JSON object — no other text, no markdown fences:\n"
        '{"decision": "<churn|retain>", '
        '"explanation": "<2-3 sentences referencing panel signals, SHAP magnitudes and benchmarks>", '
        '"recommendation": "<1-2 sentences of actionable advice for the business owner>"}'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response parser  (same logic as old project — strips Qwen3 <think> blocks)
# ─────────────────────────────────────────────────────────────────────────────

class FormatError(RuntimeError):
    pass


def parse_response(raw: str) -> tuple[str, str, str]:
    """Extract decision / explanation / recommendation from a JSON model response.

    Handles:
    - Standard single-line JSON
    - Pretty-printed JSON (with newlines)
    - Qwen3 <think>...</think> preamble
    - Markdown code fences
    - Truncated JSON: tries regex fallback to salvage at least the decision
    """
    text = str(raw).strip()

    # Strip Qwen3 <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    # ── Attempt 1: full valid JSON ────────────────────────────────────────────
    json_m = re.search(r"\{.*\}", text, re.DOTALL)
    if json_m:
        try:
            data = json.loads(json_m.group())
            decision       = str(data.get("decision", "")).strip().lower()
            explanation    = str(data.get("explanation", "")).strip()
            recommendation = str(data.get("recommendation", "")).strip()
            if decision in ALLOWED_DECISIONS and explanation and recommendation:
                return decision, explanation, recommendation
            if decision in ALLOWED_DECISIONS:
                raise FormatError(
                    f"Empty explanation or recommendation in: {data!r}"
                )
            raise FormatError(f"Invalid decision '{decision}' — must be 'churn' or 'retain'")
        except json.JSONDecodeError:
            pass  # fall through to truncation recovery

    # ── Attempt 2: truncation recovery ───────────────────────────────────────
    # The response was cut mid-JSON (max_new_tokens reached). Try to recover
    # the decision and whatever partial explanation/recommendation exists.
    dec_m  = re.search(r'"decision"\s*:\s*"(churn|retain)"', text, re.IGNORECASE)
    expl_m = re.search(r'"explanation"\s*:\s*"([^"]{20,})', text, re.DOTALL)
    rec_m  = re.search(r'"recommendation"\s*:\s*"([^"]{10,})', text, re.DOTALL)

    if dec_m:
        decision = dec_m.group(1).lower()
        explanation    = (expl_m.group(1).strip() + " [truncated]") if expl_m else "[truncated]"
        recommendation = (rec_m.group(1).strip()  + " [truncated]") if rec_m  else "[truncated]"
        return decision, explanation, recommendation

    raise FormatError(f"No JSON object found in: {text!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Chat template helper  (copied verbatim from old project model_utils)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_chat_template_compat(tokenizer, messages: list[dict],
                                enable_thinking: bool = False, **kwargs):
    """Handle Qwen models that require content as block dicts and
    may or may not accept `enable_thinking`."""
    call_kwargs = dict(enable_thinking=enable_thinking, **kwargs)
    try:
        return tokenizer.apply_chat_template(messages, **call_kwargs)
    except TypeError as exc:
        err = str(exc)
        if "string indices must be integers" not in err and "enable_thinking" not in err:
            raise
        if "enable_thinking" in err:
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError as exc2:
                if "string indices must be integers" not in str(exc2):
                    raise
        # Block-format content fallback
        block_msgs = [
            {
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}]
                if isinstance(m.get("content"), str) else m.get("content"),
            }
            for m in messages
        ]
        try:
            return tokenizer.apply_chat_template(block_msgs, **call_kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(block_msgs, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Qwen inference  (FastLanguageModel, thinking disabled)
# ─────────────────────────────────────────────────────────────────────────────

def load_qwen(cfg: ChurnZeroShotConfig):
    from unsloth import FastLanguageModel
    print(f"  Loading {cfg.qwen_model_name} (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.qwen_model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def infer_qwen(model, tokenizer, prompt: str,
               cfg: ChurnZeroShotConfig) -> tuple[str, str, str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer retention analyst. "
                "Follow the output format exactly. Respond only with the JSON object."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    inputs = _apply_chat_template_compat(
        tokenizer, messages,
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    total_time = 0.0
    last_raw   = ""
    for attempt in range(1, cfg.generation_retry_attempts + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=cfg.qwen_max_new_tokens,
                do_sample=True,
                temperature=cfg.qwen_temperature,
                top_p=cfg.qwen_top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        generated = output[0][inputs["input_ids"].shape[1]:]
        last_raw  = tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            decision, explanation, recommendation = parse_response(last_raw)
            return decision, explanation, recommendation, total_time
        except FormatError as e:
            print(f"    [Qwen] attempt {attempt}/{cfg.generation_retry_attempts} — {e}")

    raise FormatError(last_raw)


def run_qwen(df: pd.DataFrame, prompts: list[str],
             cfg: ChurnZeroShotConfig) -> pd.DataFrame:
    print("\nRunning Qwen3.5-4B zero-shot...")
    model, tokenizer = load_qwen(cfg)

    for col in ("qwen_decision", "qwen_explanation", "qwen_recommendation", "qwen_time_sec"):
        if col not in df.columns:
            df[col] = None

    output_path = Path(cfg.output_csv)
    n = len(df)

    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=n, desc="Qwen")):
        if pd.notna(df.at[idx, "qwen_decision"]):
            continue  # resume skip

        try:
            dec, expl, rec, elapsed = infer_qwen(model, tokenizer, prompts[i], cfg)
        except FormatError as exc:
            raw = str(exc)
            print(
                f"\n{'='*70}\n"
                f"[Qwen] ALL RETRIES FAILED — row {i} / index {idx}\n"
                f"{'='*70}\n"
                f"PROMPT:\n{prompts[i]}\n"
                f"LAST RESPONSE:\n{raw}\n"
                f"{'='*70}\n"
                f"Falling back to XGBoost decision.\n"
            )
            dec     = str(row.get("churn_decision", "retain")).lower()
            expl    = f"[parse_error] {raw[:300]}"
            rec     = "Unable to generate recommendation due to a format error."
            elapsed = 0.0

        df.at[idx, "qwen_decision"]       = dec
        df.at[idx, "qwen_explanation"]    = expl
        df.at[idx, "qwen_recommendation"] = rec
        df.at[idx, "qwen_time_sec"]       = elapsed

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n:
            df.to_csv(output_path, index=False)
            print(f"  Checkpoint: {i + 1}/{n} rows")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Qwen zero-shot complete.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Gemma inference  (FastModel)
# ─────────────────────────────────────────────────────────────────────────────

def load_gemma(cfg: ChurnZeroShotConfig):
    from unsloth import FastModel
    print(f"  Loading {cfg.gemma_model_name} (4-bit)...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.gemma_model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )
    FastModel.for_inference(model)
    return model, tokenizer


def infer_gemma(model, tokenizer, prompt: str,
                cfg: ChurnZeroShotConfig) -> tuple[str, str, str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    total_time = 0.0
    last_raw   = ""
    for attempt in range(1, cfg.generation_retry_attempts + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=cfg.gemma_max_new_tokens,
                do_sample=True,
                temperature=cfg.gemma_temperature,
                top_p=cfg.gemma_top_p,
                top_k=cfg.gemma_top_k,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        generated = output[0][inputs["input_ids"].shape[1]:]
        last_raw  = tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            decision, explanation, recommendation = parse_response(last_raw)
            return decision, explanation, recommendation, total_time
        except FormatError as e:
            print(f"    [Gemma] attempt {attempt}/{cfg.generation_retry_attempts} — {e}")

    raise FormatError(last_raw)


def run_gemma(df: pd.DataFrame, prompts: list[str],
              cfg: ChurnZeroShotConfig) -> pd.DataFrame:
    print("\nRunning Gemma3-4B zero-shot...")
    model, tokenizer = load_gemma(cfg)

    for col in ("gemma_decision", "gemma_explanation", "gemma_recommendation", "gemma_time_sec"):
        if col not in df.columns:
            df[col] = None

    output_path = Path(cfg.output_csv)
    n = len(df)

    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=n, desc="Gemma")):
        if pd.notna(df.at[idx, "gemma_decision"]):
            continue

        try:
            dec, expl, rec, elapsed = infer_gemma(model, tokenizer, prompts[i], cfg)
        except FormatError as exc:
            raw = str(exc)
            print(
                f"\n{'='*70}\n"
                f"[Gemma] ALL RETRIES FAILED — row {i} / index {idx}\n"
                f"{'='*70}\n"
                f"PROMPT:\n{prompts[i]}\n"
                f"LAST RESPONSE:\n{raw}\n"
                f"{'='*70}\n"
                f"Falling back to XGBoost decision.\n"
            )
            dec     = str(row.get("churn_decision", "retain")).lower()
            expl    = f"[parse_error] {raw[:300]}"
            rec     = "Unable to generate recommendation due to a format error."
            elapsed = 0.0

        df.at[idx, "gemma_decision"]       = dec
        df.at[idx, "gemma_explanation"]    = expl
        df.at[idx, "gemma_recommendation"] = rec
        df.at[idx, "gemma_time_sec"]       = elapsed

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n:
            df.to_csv(output_path, index=False)
            print(f"  Checkpoint: {i + 1}/{n} rows")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Gemma zero-shot complete.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_save_metrics(df: pd.DataFrame, cfg: ChurnZeroShotConfig) -> None:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
        classification_report,
    )

    results_dir = Path(cfg.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_true_bin = df["y_true"].astype(int)

    rows = []
    print(f"\n{'='*60}")
    print("  LLM ZERO-SHOT — TEST SET METRICS")
    print(f"{'='*60}")

    for prefix in ("qwen", "gemma"):
        col = f"{prefix}_decision"
        if col not in df.columns:
            continue
        mask = df[col].notna() & ~df[col].str.startswith("[parse_error]", na=False)
        sub  = df[mask].copy()
        if sub.empty:
            continue

        y_pred_bin = (sub[col].str.strip().str.lower() == "churn").astype(int)
        y_true_sub = sub["y_true"].astype(int)

        acc  = accuracy_score(y_true_sub, y_pred_bin)
        prec = precision_score(y_true_sub, y_pred_bin, zero_division=0)
        rec  = recall_score(y_true_sub, y_pred_bin, zero_division=0)
        f1   = f1_score(y_true_sub, y_pred_bin, zero_division=0)
        # AUC from decision (binary, no proba available)
        auc  = roc_auc_score(y_true_sub, y_pred_bin)
        avg_t = df[f"{prefix}_time_sec"].mean()

        print(f"\n  [{prefix.upper()}]")
        print(f"    n evaluated : {len(sub)}")
        print(f"    Accuracy    : {acc:.4f}")
        print(f"    Precision   : {prec:.4f}")
        print(f"    Recall      : {rec:.4f}")
        print(f"    F1          : {f1:.4f}")
        print(f"    ROC-AUC     : {auc:.4f}")
        print(f"    Avg time/row: {avg_t:.2f}s")
        print(f"\n  Classification Report ({prefix.upper()}):")
        print(classification_report(
            y_true_sub, y_pred_bin,
            target_names=["Retained", "Churned"],
            digits=4,
        ))

        rows.append({
            "model": prefix,
            "n_evaluated": len(sub),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "avg_time_sec": round(avg_t, 2),
        })

    if rows:
        metrics_df = pd.DataFrame(rows)
        metrics_path = results_dir / "llm_zs_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n  Metrics saved → {metrics_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge(cfg: ChurnZeroShotConfig) -> pd.DataFrame:
    """Merge the XGBoost LLM-ready file with RF predictions."""
    llm_ready = pd.read_csv(cfg.llm_ready_csv)
    rf_preds  = pd.read_csv(cfg.rf_pred_csv)

    print(f"  LLM-ready (XGBoost): {len(llm_ready)} rows × {llm_ready.shape[1]} cols")
    print(f"  RF predictions     : {len(rf_preds)} rows")

    if len(llm_ready) != len(rf_preds):
        raise ValueError(
            f"Row count mismatch: llm_ready={len(llm_ready)}, rf_preds={len(rf_preds)}\n"
            "Both files must correspond to the same test set."
        )

    # Rename RF columns to avoid collision
    rf_preds = rf_preds.rename(columns={
        "y_pred": "rf_pred",
        "y_proba": "rf_proba",
    }).drop(columns=["y_true"], errors="ignore")

    # Row-wise merge (both are already in test-set order)
    df = pd.concat([llm_ready.reset_index(drop=True),
                    rf_preds.reset_index(drop=True)], axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot churn explainability via Qwen3.5-4B and Gemma3-4B"
    )
    parser.add_argument("--model",  choices=["qwen", "gemma", "both"], default="both",
                        help="Which model(s) to run (default: both)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Process only the first N rows (for smoke testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output CSV (skip already-done rows)")
    parser.add_argument("--llm-ready-csv", type=str, default=None,
                        help="Override path to LLM-ready CSV")
    parser.add_argument("--rf-pred-csv",   type=str, default=None,
                        help="Override path to RF predictions CSV")
    parser.add_argument("--output-csv",    type=str, default=None,
                        help="Override output CSV path (use separate files when "
                             "running Qwen and Gemma in parallel on different GPUs)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _load_support_files(cfg: ChurnZeroShotConfig) -> None:
    """Load RF importances and population benchmarks into module-level dicts."""
    global _RF_IMPORTANCES, _BENCHMARKS
    llm_dir = Path(cfg.llm_ready_csv).parent

    rf_path = llm_dir / "rf_feature_importances.json"
    if rf_path.exists():
        with open(rf_path) as f:
            _RF_IMPORTANCES = json.load(f)
        print(f"  Loaded RF importances ({len(_RF_IMPORTANCES)} features) from {rf_path}")
    else:
        print(f"  WARNING: {rf_path} not found — RF importances will be omitted from prompt")

    bench_path = llm_dir / "population_benchmarks.json"
    if bench_path.exists():
        with open(bench_path) as f:
            _BENCHMARKS = json.load(f)
        print(f"  Loaded population benchmarks ({len(_BENCHMARKS)} features) from {bench_path}")
    else:
        print(f"  WARNING: {bench_path} not found — benchmarks will be omitted from prompt")


def main() -> None:
    args = parse_args()
    cfg  = ChurnZeroShotConfig()

    if args.llm_ready_csv:
        cfg.llm_ready_csv = args.llm_ready_csv
    if args.rf_pred_csv:
        cfg.rf_pred_csv = args.rf_pred_csv
    if args.output_csv:
        cfg.output_csv = args.output_csv
        cfg.output_dir = str(Path(args.output_csv).parent)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    verify_gpu()

    # ── Load RF importances + population benchmarks ────────────────────────
    print("\nLoading support files (RF importances + benchmarks)...")
    _load_support_files(cfg)

    # ── Load / resume ──────────────────────────────────────────────────────
    output_path = Path(cfg.output_csv)
    if args.resume and output_path.exists():
        df = pd.read_csv(output_path)
        print(f"\nResuming from checkpoint: {output_path} ({len(df)} rows)")
    else:
        print("\nLoading and merging input files...")
        df = load_and_merge(cfg)

    if args.limit:
        df = df.head(args.limit).copy()
        print(f"Limiting to {args.limit} rows (smoke test)")

    print(f"\nTotal rows to process: {len(df)}")
    print(f"Churn distribution (y_true): {df['y_true'].value_counts().to_dict()}")

    # ── Build prompts (CPU, cheap) ─────────────────────────────────────────
    print("\nBuilding advisory-panel prompts...")
    prompts = [build_prompt(row) for _, row in df.iterrows()]
    print(f"Built {len(prompts)} prompts.")

    # ── Inference ─────────────────────────────────────────────────────────
    run_qwen_flag   = args.model in ("qwen",  "both")
    run_gemma_flag  = args.model in ("gemma", "both")

    if run_qwen_flag:
        df = run_qwen(df, prompts, cfg)

    if run_gemma_flag:
        df = run_gemma(df, prompts, cfg)

    # ── Final save ─────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)
    print(f"\nFinal output saved → {output_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Metrics ────────────────────────────────────────────────────────────
    compute_and_save_metrics(df, cfg)

    print("\n✅ Zero-shot LLM explainability complete.")


if __name__ == "__main__":
    main()
