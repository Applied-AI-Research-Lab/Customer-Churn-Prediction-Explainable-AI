import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
FULL_A5_CSV = os.path.join(BASE_DIR, "outputs", "llm_predictions", "test_llm_predictions.csv")
A9_MERGED_CSV = os.path.join(BASE_DIR, "new", "outputs", "predictions", "A9_full_MERGED.csv")
ORIGINAL_ROW_IDS_CSV = os.path.join(BASE_DIR, "new", "outputs", "a9_followup_row_ids.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

N = 25
SEED = 43


def top_reasons_text(row, k=5):
    lines = []
    for i in range(1, k + 1):
        r = row.get(f"reason_{i}")
        if isinstance(r, str) and r:
            lines.append(f"  {i}. {r}")
    return "\n".join(lines)


def main():
    a5 = pd.read_csv(FULL_A5_CSV).reset_index(drop=True)
    a5["row_id"] = np.arange(len(a5))
    a9 = pd.read_csv(A9_MERGED_CSV)

    excluded = set(pd.read_csv(ORIGINAL_ROW_IDS_CSV)["row_id"])
    pool = a5[~a5["row_id"].isin(excluded)]

    rng = np.random.default_rng(SEED)
    sample_ids = rng.choice(pool["row_id"].values, size=N, replace=False)
    sample_ids.sort()

    a5_sub = a5[a5["row_id"].isin(sample_ids)].set_index("row_id")
    a9_sub = a9.set_index("row_id").loc[sample_ids]

    rows, key_rows = [], []
    for i, row_id in enumerate(sample_ids):
        item_id = f"V{i+1:03d}"
        r5 = a5_sub.loc[row_id]
        r9 = a9_sub.loc[row_id]

        candidates = [
            ("qwen", "A5", r5["qwen_explanation"]),
            ("qwen", "A9", r9["qwen_explanation"]),
            ("gemma", "A5", r5["gemma_explanation"]),
            ("gemma", "A9", r9["gemma_explanation"]),
        ]
        order = rng.permutation(4)
        slots = ["A", "B", "C", "D"]

        row_out = {
            "item_id": item_id,
            "row_id": row_id,
            "risk_tier": r5["risk_tier"],
            "xgb_probability_pct": round(float(r5["y_proba"]) * 100, 1),
            "xgb_decision": r5["churn_decision"],
            "top_5_shap_reasons": top_reasons_text(r5, 5),
        }
        key_row = {"item_id": item_id, "row_id": row_id}
        for slot_idx, cand_idx in zip(slots, order):
            model, cond, text = candidates[cand_idx]
            row_out[f"text_{slot_idx}"] = text
            row_out[f"fabricated_{slot_idx}"] = ""
            key_row[f"text_{slot_idx}_model"] = model
            key_row[f"text_{slot_idx}_condition"] = cond
        row_out["notes"] = ""
        rows.append(row_out)
        key_rows.append(key_row)

    sheet = pd.DataFrame(rows)
    key = pd.DataFrame(key_rows)

    for coder in ("coderA", "coderB"):
        sheet.to_csv(os.path.join(OUT_DIR, f"spotcheck_{coder}.csv"), index=False)
    key.to_csv(os.path.join(OUT_DIR, "_answer_key_DO_NOT_SHARE.csv"), index=False)

    print(f"Sample built: {N} customers x 4 texts = {4*N} quick fabrication judgments per coder.")
    print(f"Risk-tier composition: {a5_sub['risk_tier'].value_counts().to_dict()}")
    print("Saved -> spotcheck_coderA.csv, spotcheck_coderB.csv (identical, blinded)")
    print("Saved -> _answer_key_DO_NOT_SHARE.csv (keep yourself)")


if __name__ == "__main__":
    main()
