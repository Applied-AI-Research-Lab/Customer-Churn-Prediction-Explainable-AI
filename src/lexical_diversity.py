from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(BASE_DIR, "outputs", "llm_predictions", "test_llm_predictions.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

TTR_THRESHOLD = 0.72


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def _mtld_pass(tokens: list[str], threshold: float = TTR_THRESHOLD) -> float:
    if len(tokens) < 2:
        return float(len(tokens))
    factors = 0.0
    types: set[str] = set()
    token_count = 0
    for tok in tokens:
        types.add(tok)
        token_count += 1
        ttr = len(types) / token_count
        if ttr <= threshold:
            factors += 1.0
            types = set()
            token_count = 0
    if token_count > 0:
        ttr = len(types) / token_count
        partial = (1 - ttr) / (1 - threshold) if ttr < 1.0 else 0.0
        factors += min(partial, 1.0)
    return len(tokens) / factors if factors > 0 else float(len(tokens))


def mtld(text: str) -> float:
    tokens = tokenize(text)
    if len(tokens) < 10:
        return float("nan")
    forward = _mtld_pass(tokens)
    backward = _mtld_pass(list(reversed(tokens)))
    return (forward + backward) / 2.0


def main() -> None:
    df = pd.read_csv(PRED_CSV)
    rows = []
    for model in ("qwen", "gemma"):
        col = f"{model}_explanation"
        mtld_scores = df[col].apply(mtld)
        unique_ratio = df[col].apply(
            lambda t: len(set(tokenize(t))) / len(tokenize(t)) if len(tokenize(t)) > 0 else float("nan")
        )
        length = df[col].str.len()
        rows.append({
            "model": model, "n": mtld_scores.notna().sum(),
            "mtld_mean": mtld_scores.mean(), "mtld_sd": mtld_scores.std(),
            "unique_word_ratio_mean": unique_ratio.mean(),
            "mean_length_chars": length.mean(),
            "corr_unique_ratio_vs_length": unique_ratio.corr(length),
            "corr_mtld_vs_length": mtld_scores.corr(length),
        })
        print(f"[{model.upper()}] n={rows[-1]['n']}  "
              f"MTLD={rows[-1]['mtld_mean']:.1f} (SD {rows[-1]['mtld_sd']:.1f})  "
              f"raw unique-word-ratio={rows[-1]['unique_word_ratio_mean']:.3f}  "
              f"mean length={rows[-1]['mean_length_chars']:.1f} chars  "
              f"corr(unique-ratio, length)={rows[-1]['corr_unique_ratio_vs_length']:+.3f}  "
              f"corr(MTLD, length)={rows[-1]['corr_mtld_vs_length']:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "lexical_diversity.csv"), index=False)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'lexical_diversity.csv')}")


if __name__ == "__main__":
    main()
