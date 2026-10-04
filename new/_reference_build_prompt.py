from __future__ import annotations

import pandas as pd

_RF_IMPORTANCES: dict = {}
_BENCHMARKS: dict = {}


def set_support_files(rf_importances: dict, benchmarks: dict) -> None:
    global _RF_IMPORTANCES, _BENCHMARKS
    _RF_IMPORTANCES = rf_importances
    _BENCHMARKS = benchmarks


def _fmt(val, decimals: int = 2) -> str:
    try:
        f = float(val)
        if f != f:
            return "N/A"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(val) if pd.notna(val) else "N/A"


def build_prompt(row: pd.Series) -> str:
    age        = _fmt(row.get("Age"), 0)
    gender     = str(row.get("Gender", "N/A"))
    country    = str(row.get("Country", "N/A"))
    mem_yrs    = _fmt(row.get("Membership_Years"))
    ltv        = _fmt(row.get("Lifetime_Value"))
    credit_bal = _fmt(row.get("Credit_Balance"))
    signup_q   = str(row.get("Signup_Quarter", "N/A"))

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

    xgb_pred   = str(row.get("churn_decision", "N/A")).upper()
    xgb_prob   = _fmt(float(row.get("y_proba", 0)) * 100, 1)

    rf_pred_raw = row.get("rf_pred", None)
    rf_proba    = row.get("rf_proba", None)
    if pd.notna(rf_pred_raw):
        rf_decision = "CHURN" if int(rf_pred_raw) == 1 else "RETAIN"
        rf_prob     = _fmt(float(rf_proba) * 100, 1) if pd.notna(rf_proba) else "N/A"
    else:
        rf_decision, rf_prob = "N/A", "N/A"

    reasons = [str(row.get(f"reason_{k}", "N/A")) for k in range(1, 6)]
    r_lines = "\n".join(f"    SHAP reason {k}   : {r}" for k, r in enumerate(reasons, 1) if r and r != "N/A")

    risk_tier = str(row.get("risk_tier", "N/A"))

    if _RF_IMPORTANCES:
        rf_top5 = list(_RF_IMPORTANCES.items())[:5]
        rf_imp_lines = "  ".join(
            f"{feat.split('_')[0] if len(feat) > 25 else feat}={imp:.3f}"
            for feat, imp in rf_top5
        )
    else:
        rf_imp_lines = "N/A"

    bench_lines = ""
    if _BENCHMARKS:
        bench_feats = []
        for r in reasons[:3]:
            for feat in _BENCHMARKS:
                desc = feat.replace("_", " ").lower()
                if desc in r.lower() or feat.lower() in r.lower():
                    if feat not in bench_feats:
                        bench_feats.append(feat)
                    break
        if not bench_feats:
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
