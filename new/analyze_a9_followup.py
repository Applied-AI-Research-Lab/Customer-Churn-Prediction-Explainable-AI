from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_RUN_CSV = os.path.join(BASE_DIR, "outputs", "llm_predictions", "test_llm_predictions.csv")
A9_CSV = os.path.join(BASE_DIR, "new", "outputs", "predictions", "A9", "both.csv")
ANSWER_KEY = os.path.join(BASE_DIR, "Paper", "Round1", "faithfulness_audit", "_answer_key_DO_NOT_SHARE.csv")
ADJUDICATION_CSV = os.path.join(BASE_DIR, "Paper", "Round1", "faithfulness_audit", "final_resolved_faithfulness.csv")

FEATURE_PATTERNS = {
    "Age": r"\bage\b",
    "Membership_Years": r"membership (duration|years|tenure)|years? (of )?tenure|\d+(\.\d+)? years? (with|tenure)",
    "Login_Frequency": r"login frequency|logs? in (frequently|often)",
    "Session_Duration_Avg": r"session duration",
    "Pages_Per_Session": r"pages (per|browsed)|page views",
    "Cart_Abandonment_Rate": r"cart abandonment",
    "Wishlist_Items": r"wishlist",
    "Total_Purchases": r"total purchases",
    "Average_Order_Value": r"(average|avg\.?) order value",
    "Days_Since_Last_Purchase": r"days? since.{0,20}last purchase|recent (purchase|inactivity|activity)|\d+-day (gap|inactivity)",
    "Discount_Usage_Rate": r"discount usage|discount (rate|reliance|dependency)",
    "Returns_Rate": r"returns? rate",
    "Email_Open_Rate": r"email open|email engagement",
    "Customer_Service_Calls": r"(customer )?service calls?|support calls?",
    "Product_Reviews_Written": r"product reviews?|reviews written",
    "Social_Media_Engagement_Score": r"social media|social engagement",
    "Mobile_App_Usage": r"mobile app",
    "Payment_Method_Diversity": r"payment method",
    "Lifetime_Value": r"lifetime value|\bltv\b",
    "Credit_Balance": r"credit balance",
}


def detect_features(text: str) -> set[str]:
    text = str(text).lower()
    return {feat for feat, pat in FEATURE_PATTERNS.items() if re.search(pat, text)}


def top5_features(row: pd.Series) -> set[str]:
    found = set()
    for i in range(1, 6):
        reason = str(row.get(f"reason_{i}", ""))
        for feat, pat in FEATURE_PATTERNS.items():
            if re.search(pat, reason.lower()):
                found.add(feat)
    return found


def fabrication_rate(df: pd.DataFrame, explanation_col: str) -> tuple[float, pd.Series]:
    flags = []
    fabricated_lists = []
    for _, row in df.iterrows():
        top5 = top5_features(row)
        cited = detect_features(row[explanation_col])
        fabricated = cited - top5
        flags.append(len(fabricated) > 0)
        fabricated_lists.append(",".join(sorted(fabricated)))
    return np.mean(flags), pd.Series(fabricated_lists, index=df.index)


def main() -> None:
    key = pd.read_csv(ANSWER_KEY)
    row_ids = key["row_id"].sort_values().unique()

    full = pd.read_csv(FULL_RUN_CSV).reset_index(drop=True)
    full["row_id"] = np.arange(len(full))
    a5 = full[full["row_id"].isin(row_ids)].sort_values("row_id").reset_index(drop=True)

    print("=" * 80)
    print("  Sanity check: does the automated proxy track the real human-adjudicated result?")
    print("=" * 80)
    if os.path.exists(ADJUDICATION_CSV):
        resolved = pd.read_csv(ADJUDICATION_CSV)
        rate_auto, fab_lists = fabrication_rate(a5, "qwen_explanation")
        print(f"  Automated proxy fabrication rate (A5, Qwen): {rate_auto:.1%}")
        rate_auto_g, _ = fabrication_rate(a5, "gemma_explanation")
        print(f"  Automated proxy fabrication rate (A5, Gemma): {rate_auto_g:.1%}")
        print("  (compare by eye against the real adjudicated ~42-44% faithful / ~36-40% fabricated-tagged rate;")
        print("   this proxy is necessarily rougher -- it flags ANY mention of a non-top-5 feature, regardless")
        print("   of whether it's framed as load-bearing evidence or a passing caveat -- so expect it to run higher.)")
    else:
        print("  (adjudication file not found -- skipping sanity check)")

    if not os.path.exists(A9_CSV):
        print(f"\n  {A9_CSV} not found yet -- run new/run_a9_followup.py on the GPU server first.")
        return

    a9 = pd.read_csv(A9_CSV).sort_values("row_id").reset_index(drop=True)
    assert list(a5["row_id"]) == list(a9["row_id"]), "A5 and A9 must cover the identical 50 customers"

    print("\n" + "=" * 80)
    print("  A5 (original) vs A9 (evidence/context instruction) -- automated fabrication proxy")
    print("=" * 80)
    for model, col in [("qwen", "qwen_explanation"), ("gemma", "gemma_explanation")]:
        rate_a5, fab_a5 = fabrication_rate(a5, col)
        rate_a9, fab_a9 = fabrication_rate(a9, col)
        print(f"\n  [{model.upper()}]")
        print(f"    A5 (original):        {rate_a5:.1%} of explanations cite a non-top-5 feature")
        print(f"    A9 (new instruction): {rate_a9:.1%} of explanations cite a non-top-5 feature")
        print(f"    Change: {rate_a9 - rate_a5:+.1%}")

        from collections import Counter
        counts_a5 = Counter(f for lst in fab_a5 if lst for f in lst.split(","))
        counts_a9 = Counter(f for lst in fab_a9 if lst for f in lst.split(","))
        print(f"    Most-fabricated features (A5): {counts_a5.most_common(5)}")
        print(f"    Most-fabricated features (A9): {counts_a9.most_common(5)}")

    out = pd.DataFrame({
        "row_id": a5["row_id"],
        "qwen_fabricated_A5": fabrication_rate(a5, "qwen_explanation")[1],
        "qwen_fabricated_A9": fabrication_rate(a9, "qwen_explanation")[1],
        "gemma_fabricated_A5": fabrication_rate(a5, "gemma_explanation")[1],
        "gemma_fabricated_A9": fabrication_rate(a9, "gemma_explanation")[1],
    })
    out_path = os.path.join(BASE_DIR, "new", "outputs", "analysis", "a9_followup_comparison.csv")
    out.to_csv(out_path, index=False)
    print(f"\n  Saved per-row detail -> {out_path}")


if __name__ == "__main__":
    main()
