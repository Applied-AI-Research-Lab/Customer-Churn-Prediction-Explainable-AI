import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PRED_CSV = os.path.join(BASE_DIR, "outputs", "llm_predictions", "test_llm_predictions.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

STRATA_N = {"Critical": 10, "High": 13, "Medium": 12, "Low": 15}
SEED = 42


def top_reasons_text(row, k=5):
    lines = []
    for i in range(1, k + 1):
        r = row.get(f"reason_{i}")
        if isinstance(r, str) and r:
            lines.append(f"  {i}. {r}")
    return "\n".join(lines)


def main():
    df = pd.read_csv(PRED_CSV)
    df["row_id"] = np.arange(len(df))

    rng = np.random.default_rng(SEED)
    parts = []
    for tier, n in STRATA_N.items():
        pool = df[df["risk_tier"] == tier]
        idx = rng.choice(pool.index.values, size=min(n, len(pool)), replace=False)
        parts.append(df.loc[idx])
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    a_is_qwen = rng.integers(0, 2, size=len(sample)).astype(bool)

    rows = []
    key_rows = []
    for i, r in sample.iterrows():
        item_id = f"F{i+1:03d}"
        text_a_model = "qwen" if a_is_qwen[i] else "gemma"
        text_b_model = "gemma" if a_is_qwen[i] else "qwen"

        rows.append({
            "item_id": item_id,
            "row_id": r["row_id"],
            "risk_tier": r["risk_tier"],
            "xgb_probability_pct": round(float(r["y_proba"]) * 100, 1),
            "xgb_decision": r["churn_decision"],
            "top_5_shap_reasons": top_reasons_text(r, 5),
            "text_A_decision": r[f"{text_a_model}_decision"],
            "text_A_explanation": r[f"{text_a_model}_explanation"],
            "text_A_recommendation": r[f"{text_a_model}_recommendation"],
            "text_B_decision": r[f"{text_b_model}_decision"],
            "text_B_explanation": r[f"{text_b_model}_explanation"],
            "text_B_recommendation": r[f"{text_b_model}_recommendation"],
            "faithful_A": "", "faithful_B": "",
            "failure_tags_A": "", "failure_tags_B": "",
            "notes": "",
        })
        key_rows.append({"item_id": item_id, "row_id": r["row_id"], "text_A_model": text_a_model, "text_B_model": text_b_model})

    sheet = pd.DataFrame(rows)
    key = pd.DataFrame(key_rows)

    for coder in ("coderA", "coderB"):
        sheet.to_csv(os.path.join(OUT_DIR, f"coding_sheet_{coder}.csv"), index=False)
    key.to_csv(os.path.join(OUT_DIR, "_answer_key_DO_NOT_SHARE.csv"), index=False)

    print(f"Sample built: {len(sheet)} customers x 2 models = {2*len(sheet)} texts to judge per coder.")
    print(f"Risk-tier composition: {sample['risk_tier'].value_counts().to_dict()}")
    print(f"Saved -> coding_sheet_coderA.csv, coding_sheet_coderB.csv (identical, blinded)")
    print(f"Saved -> _answer_key_DO_NOT_SHARE.csv (keep this yourself; needed only for analysis afterwards)")


if __name__ == "__main__":
    main()
