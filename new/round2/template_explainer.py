from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_CSV = os.path.join(BASE, "outputs", "llm_predictions", "test_llm_predictions.csv")
BENCH_JSON = os.path.join(BASE, "outputs", "llm_ready", "population_benchmarks.json")
OUT = os.path.join(BASE, "new", "outputs", "round2", "template_explanations.csv")

REASON_RE = re.compile(
    r"^(?P<phrase>.*?) \((?P<value>[^)]*)\) (?P<dir>increases|reduces) churn risk "
    r"\[SHAP=(?P<shap>[+-][0-9.]+), (?P<label>[^\]]+)\]$")
CATEGORICAL_RE = re.compile(
    r"^(?P<phrase>[^=]+?) = (?P<value>.+?) (?P<dir>increases|reduces) churn risk "
    r"\[SHAP=(?P<shap>[+-][0-9.]+), (?P<label>[^\]]+)\]$")

DESC_TO_FEATURE = {
    "age": "Age", "membership duration (years)": "Membership_Years", "login frequency": "Login_Frequency",
    "average session duration": "Session_Duration_Avg", "pages browsed per session": "Pages_Per_Session",
    "cart abandonment rate": "Cart_Abandonment_Rate", "wishlist items count": "Wishlist_Items",
    "total purchases": "Total_Purchases", "average order value": "Average_Order_Value",
    "days since last purchase": "Days_Since_Last_Purchase", "discount usage rate": "Discount_Usage_Rate",
    "returns rate": "Returns_Rate", "email open rate": "Email_Open_Rate",
    "customer service calls": "Customer_Service_Calls", "product reviews written": "Product_Reviews_Written",
    "social media engagement score": "Social_Media_Engagement_Score", "mobile app usage": "Mobile_App_Usage",
    "payment method diversity": "Payment_Method_Diversity", "lifetime value": "Lifetime_Value",
    "credit balance": "Credit_Balance",
}

ACTIONS = {
    "Days_Since_Last_Purchase": "send a time-limited re-engagement offer to prompt a new purchase",
    "Customer_Service_Calls": "assign a dedicated support contact to resolve the customer's open issues",
    "Cart_Abandonment_Rate": "send cart-recovery reminders with a checkout incentive",
    "Returns_Rate": "review the customer's recent returns and follow up on product fit or quality",
    "Email_Open_Rate": "move retention messages to a channel the customer engages with (app or SMS)",
    "Login_Frequency": "run a reactivation campaign highlighting new or relevant products",
    "Session_Duration_Avg": "improve onsite personalisation of recommendations for this customer",
    "Pages_Per_Session": "improve onsite personalisation of recommendations for this customer",
    "Average_Order_Value": "offer bundles or loyalty points that reward consistent order values",
    "Lifetime_Value": "enrol the customer in a loyalty tier proportional to their value",
    "Age": "use age-appropriate messaging in the retention offer",
    "Discount_Usage_Rate": "offer a targeted discount on categories the customer buys",
    "Total_Purchases": "offer a purchase-milestone reward",
    "Membership_Years": "send an onboarding or tenure-recognition message",
    "Credit_Balance": "remind the customer of their available store credit",
    "Mobile_App_Usage": "promote app-exclusive offers",
    "Social_Media_Engagement_Score": "invite the customer to community or social campaigns",
    "Wishlist_Items": "send price-drop alerts on wishlist items",
    "Product_Reviews_Written": "invite the customer to review recent purchases",
    "Payment_Method_Diversity": "offer an additional convenient payment option",
}

def parse_reasons(row: pd.Series) -> list[dict]:
    out = []
    for k in range(1, 6):
        text = str(row.get(f"reason_{k}", "")).strip()
        m = REASON_RE.match(text) or CATEGORICAL_RE.match(text)
        if not m:
            continue
        phrase = m["phrase"]
        desc = re.sub(r"^(High|Low)\s+", "", phrase).lower()
        feat = DESC_TO_FEATURE.get(desc, re.sub(r"\s*=.*$", "", phrase).strip())
        out.append(dict(phrase=phrase, value=m["value"], increases=m["dir"] == "increases",
                        shap=float(m["shap"]), label=m["label"], feature=feat))
    return out

def fmt_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]

def explain(row: pd.Series, bench: dict) -> tuple[str, str, str]:
    decision = str(row["churn_decision"]).upper()
    rf_decision = "CHURN" if int(row["rf_pred"]) == 1 else "RETAIN"
    reasons = parse_reasons(row)
    up = [r for r in reasons if r["increases"]]
    down = [r for r in reasons if not r["increases"]]

    name = lambda r: re.sub(r"^(High|Low)\s+", "", r["phrase"]).lower()
    item = lambda r: f"{name(r)} ({r['value']}; {r['label']})"

    s = [f"XGBoost predicts {decision} with a {row['y_proba'] * 100:.1f}% churn probability "
         f"({row['risk_tier']} risk); Random Forest predicts {rf_decision} "
         f"({row['rf_proba'] * 100:.1f}%)."]
    if up:
        s.append(f"Factors increasing churn risk: {fmt_list([item(r) for r in up])}.")
    if down:
        s.append(f"Factors reducing churn risk: {fmt_list([item(r) for r in down])}.")
    top = max(up, key=lambda r: r["shap"]) if up else None
    if top and top["feature"] in bench:
        b = bench[top["feature"]]
        s.append(f"For comparison, churners average {b['churned_mean']} and retained customers "
                 f"{b['retained_mean']} on {name(top)}.")
    explanation = " ".join(s)

    if decision == "CHURN" and top:
        rec = f"Prioritise this customer for retention: {ACTIONS.get(top['feature'], 'contact the customer with a personalised retention offer')}."
    elif decision == "CHURN":
        rec = "Prioritise this customer for retention with a personalised offer."
    elif top and top["label"] in ("dominant driver", "strong signal"):
        rec = f"No retention spend needed now; monitor {name(top)}."
    else:
        rec = "No retention action needed; keep the customer in the standard engagement programme."
    return decision.lower(), explanation, rec

def main() -> None:
    df = pd.read_csv(TEST_CSV)
    bench = json.load(open(BENCH_JSON))
    rows = []
    for i, r in df.iterrows():
        d, e, rec = explain(r, bench)
        rows.append(dict(row_id=i, template_decision=d, template_explanation=e, template_recommendation=rec))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    unparsed = sum(len(parse_reasons(r)) < 5 for _, r in df.iterrows())
    print(f"Wrote {len(out)} template explanations -> {OUT}  (rows with <5 parsed reasons: {unparsed})")
    for i in (8, 17):
        print("\n", out.loc[i, "template_explanation"], "\n  REC:", out.loc[i, "template_recommendation"])

if __name__ == "__main__":
    main()
