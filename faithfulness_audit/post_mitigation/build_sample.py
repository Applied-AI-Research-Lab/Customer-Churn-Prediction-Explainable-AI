from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "blank_sheets")
sys.path.insert(0, os.path.join(BASE, "new"))
sys.path.insert(0, os.path.join(BASE, "src"))
import prompt_blocks
from arms import render_prompt

A5_CSV = os.path.join(BASE, "outputs", "llm_predictions", "test_llm_predictions.csv")
A9_CSV = os.path.join(BASE, "new", "outputs", "predictions", "A9_full_MERGED.csv")
EXCLUDE = [os.path.join(HERE, "excluded_row_ids.csv")]
STRATA_N = {"Critical": 12, "High": 16, "Medium": 14, "Low": 18}
N_PRACTICE = 10
SEED = 44
DIMENSIONS = ["fabricated", "reversed", "magnitude", "causal", "recommendation"]

def evidence_block(row: pd.Series) -> str:
    prompt = render_prompt(row, "A5")
    start = prompt.index("━━━ CUSTOMER PROFILE")
    status = prompt.index("Panel status")
    end = prompt.index("\n", prompt.index("━━━━", status)) if "━━━━" in prompt[status:] else len(prompt)
    return prompt[start:end].strip()

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_blocks.set_support_files(
        json.load(open(os.path.join(BASE, "outputs", "llm_ready", "rf_feature_importances.json"))),
        json.load(open(os.path.join(BASE, "outputs", "llm_ready", "population_benchmarks.json"))))
    a5 = pd.read_csv(A5_CSV).reset_index(drop=True)
    a5["row_id"] = np.arange(len(a5))
    a9 = pd.read_csv(A9_CSV).set_index("row_id")
    return a5, a9

def texts_for(r5: pd.Series, r9: pd.Series) -> list[dict]:
    out = []
    for model in ("qwen", "gemma"):
        for cond, r in (("A5", r5), ("A9", r9)):
            out.append(dict(model=model, condition=cond, decision=r[f"{model}_decision"],
                            explanation=r[f"{model}_explanation"], recommendation=r[f"{model}_recommendation"]))
    return out

def blank_codes() -> dict:
    d = {"faithful": ""}
    d.update({f"dim_{d_}": "" for d_ in DIMENSIONS})
    d["notes"] = ""
    return d

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    a5, a9 = load()
    excluded = set()
    for path in EXCLUDE:
        excluded |= set(pd.read_csv(path)["row_id"])
    pool = a5[~a5["row_id"].isin(excluded)]
    rng = np.random.default_rng(SEED)

    parts, weights = [], {}
    for tier, n in STRATA_N.items():
        tier_pool = pool[pool["risk_tier"] == tier]
        ids = rng.choice(tier_pool["row_id"].values, size=n, replace=False)
        parts.extend(ids)
        w = (a5["risk_tier"] == tier).sum() / n
        weights.update({int(i): float(w) for i in ids})
    sample_ids = rng.permutation(parts)

    sheet, key = [], []
    for c, row_id in enumerate(sample_ids, 1):
        r5, r9 = a5.set_index("row_id").loc[row_id], a9.loc[row_id]
        ev = evidence_block(a5.loc[row_id])
        texts = texts_for(r5, r9)
        for slot, t in zip("ABCD", rng.permutation(texts)):
            text_id = f"P{c:03d}-{slot}"
            sheet.append(dict(text_id=text_id, customer=f"P{c:03d}", evidence_shown_to_model=ev,
                              decision=t["decision"], explanation=t["explanation"],
                              recommendation=t["recommendation"], **blank_codes()))
            key.append(dict(text_id=text_id, customer=f"P{c:03d}", row_id=int(row_id),
                            risk_tier=a5.loc[row_id, "risk_tier"], weight=weights[int(row_id)],
                            model=t["model"], condition=t["condition"]))

    sheet, key = pd.DataFrame(sheet), pd.DataFrame(key)
    for coder in ("coderA", "coderB"):
        sheet.to_csv(os.path.join(OUT, f"coding_sheet_{coder}.csv"), index=False)
    key.to_csv(os.path.join(OUT, "_answer_key_DO_NOT_SHARE.csv"), index=False)

    practice_pool = pool[~pool["row_id"].isin(sample_ids)]
    p_ids = rng.choice(practice_pool["row_id"].values, size=N_PRACTICE, replace=False)
    practice = []
    for k, row_id in enumerate(p_ids, 1):
        r5, r9 = a5.set_index("row_id").loc[row_id], a9.loc[row_id]
        t = texts_for(r5, r9)[rng.integers(0, 4)]
        practice.append(dict(text_id=f"PRACTICE-{k:02d}", evidence_shown_to_model=evidence_block(a5.loc[row_id]),
                             decision=t["decision"], explanation=t["explanation"],
                             recommendation=t["recommendation"], **blank_codes()))
    pd.DataFrame(practice).to_csv(os.path.join(OUT, "calibration_practice_sheet.csv"), index=False)

    print(f"Customers: {len(sample_ids)}  texts per coder: {len(sheet)}")
    print("Tier composition:", key.drop_duplicates('customer')['risk_tier'].value_counts().to_dict())
    print("Overlap with excluded samples:", len(set(sample_ids) & excluded))

if __name__ == "__main__":
    main()
