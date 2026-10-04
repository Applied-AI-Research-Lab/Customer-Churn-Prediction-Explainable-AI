
from __future__ import annotations

import pandas as pd

RF_IMPORTANCES: dict = {}
BENCHMARKS: dict = {}


def set_support_files(rf_importances: dict, benchmarks: dict) -> None:
    global RF_IMPORTANCES, BENCHMARKS
    RF_IMPORTANCES = rf_importances
    BENCHMARKS = benchmarks


def _fmt(val, decimals: int = 2) -> str:
    try:
        f = float(val)
        if f != f:
            return "N/A"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(val) if pd.notna(val) else "N/A"


def profile_block(row: pd.Series) -> str:
    age = _fmt(row.get("Age"), 0)
    gender = str(row.get("Gender", "N/A"))
    country = str(row.get("Country", "N/A"))
    mem_yrs = _fmt(row.get("Membership_Years"))
    ltv = _fmt(row.get("Lifetime_Value"))
    credit_bal = _fmt(row.get("Credit_Balance"))
    signup_q = str(row.get("Signup_Quarter", "N/A"))
    return (
        "━━━ CUSTOMER PROFILE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Age              : {age}\n"
        f"  Gender           : {gender}\n"
        f"  Country          : {country}\n"
        f"  Signup quarter   : {signup_q}\n"
        f"  Membership years : {mem_yrs}\n"
        f"  Lifetime value   : ${ltv}\n"
        f"  Credit balance   : ${credit_bal}\n"
    )


def behaviour_block(row: pd.Series) -> str:
    login_freq = _fmt(row.get("Login_Frequency"), 0)
    session = _fmt(row.get("Session_Duration_Avg"))
    cart_ab = _fmt(row.get("Cart_Abandonment_Rate"))
    days_last = _fmt(row.get("Days_Since_Last_Purchase"), 0)
    cs_calls = _fmt(row.get("Customer_Service_Calls"), 0)
    email_open = _fmt(row.get("Email_Open_Rate"))
    soc_media = _fmt(row.get("Social_Media_Engagement_Score"))
    returns = _fmt(row.get("Returns_Rate"))
    total_pur = _fmt(row.get("Total_Purchases"), 0)
    avg_order = _fmt(row.get("Average_Order_Value"))
    mobile_app = _fmt(row.get("Mobile_App_Usage"))
    discount = _fmt(row.get("Discount_Usage_Rate"))
    return (
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
        f"  Customer service calls      : {cs_calls}\n"
    )


def benchmarks_block(row: pd.Series) -> str:
    if not BENCHMARKS:
        return ""
    reasons = [str(row.get(f"reason_{k}", "N/A")) for k in range(1, 6)]
    bench_feats: list[str] = []
    for r in reasons[:3]:
        for feat in BENCHMARKS:
            desc = feat.replace("_", " ").lower()
            if desc in r.lower() or feat.lower() in r.lower():
                if feat not in bench_feats:
                    bench_feats.append(feat)
                break
    if not bench_feats:
        bench_feats = list(BENCHMARKS.keys())[:3]
    bench_parts = []
    for feat in bench_feats[:3]:
        b = BENCHMARKS[feat]
        cust_val = row.get(feat, None)
        cust_str = _fmt(cust_val) if pd.notna(cust_val) else "N/A"
        bench_parts.append(
            f"  {feat}: customer={cust_str} | avg churner={b['churned_mean']} "
            f"| avg retained={b['retained_mean']}"
        )
    bench_lines = "\n".join(bench_parts)
    return (
        "━━━ POPULATION BENCHMARKS (training set) ━━━━━━━━━━━━━━━━━\n"
        f"{bench_lines}\n"
    )


def shap_block(row: pd.Series) -> str:
    reasons = [str(row.get(f"reason_{k}", "N/A")) for k in range(1, 6)]
    r_lines = "\n".join(f"    SHAP reason {k}   : {r}" for k, r in enumerate(reasons, 1) if r and r != "N/A")
    return (
        "━━━ SHAP EXPLANATION SIGNALS (feature attributions, no verdict) ━━━━━━━━━\n"
        f"{r_lines}\n"
    )


def _xgb_fields(row: pd.Series, invert: bool = False) -> tuple[str, str, str]:
    xgb_pred = str(row.get("churn_decision", "N/A")).upper()
    proba = float(row.get("y_proba", 0))
    risk_tier = str(row.get("risk_tier", "N/A"))
    if invert:
        xgb_pred = "RETAIN" if xgb_pred == "CHURN" else "CHURN"
        proba = 1.0 - proba
        if proba >= 0.80:
            risk_tier = "Critical"
        elif proba >= 0.60:
            risk_tier = "High"
        elif proba >= 0.40:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"
    return xgb_pred, _fmt(proba * 100, 1), risk_tier


def xgb_panel_block(row: pd.Series, invert: bool = False) -> str:
    xgb_pred, xgb_prob, risk_tier = _xgb_fields(row, invert=invert)
    reasons = [str(row.get(f"reason_{k}", "N/A")) for k in range(1, 6)]
    r_lines = "\n".join(f"    SHAP reason {k}   : {r}" for k, r in enumerate(reasons, 1) if r and r != "N/A")
    return (
        "[1] XGBOOST  (gradient-boosted tree expert — best model, ROC-AUC 0.927)\n"
        f"    Prediction       : {xgb_pred}\n"
        f"    Churn probability : {xgb_prob}%   |   Risk tier: {risk_tier}\n"
        f"{r_lines}\n"
    )


def xgb_verdict_only_block(row: pd.Series) -> str:
    xgb_pred, xgb_prob, _ = _xgb_fields(row, invert=False)
    return (
        "A prior predictive model has already analyzed this customer and "
        f"reached the following verdict:\n"
        f"    Prediction        : {xgb_pred}\n"
        f"    Churn probability : {xgb_prob}%\n"
    )


def _rf_fields(row: pd.Series, invert: bool = False) -> tuple[str, str]:
    rf_pred_raw = row.get("rf_pred", None)
    rf_proba = row.get("rf_proba", None)
    if pd.notna(rf_pred_raw):
        rf_decision = "CHURN" if int(rf_pred_raw) == 1 else "RETAIN"
        prob = float(rf_proba) if pd.notna(rf_proba) else None
    else:
        rf_decision, prob = "N/A", None
    if invert and prob is not None:
        rf_decision = "RETAIN" if rf_decision == "CHURN" else "CHURN"
        prob = 1.0 - prob
    rf_prob = _fmt(prob * 100, 1) if prob is not None else "N/A"
    return rf_decision, rf_prob


def rf_panel_block(row: pd.Series, include_importances: bool = True, invert: bool = False) -> str:
    rf_decision, rf_prob = _rf_fields(row, invert=invert)
    if include_importances:
        if RF_IMPORTANCES:
            rf_top5 = list(RF_IMPORTANCES.items())[:5]
            rf_imp_lines = "  ".join(
                f"{feat.split('_')[0] if len(feat) > 25 else feat}={imp:.3f}"
                for feat, imp in rf_top5
            )
        else:
            rf_imp_lines = "N/A"
    else:
        rf_imp_lines = None
    lines = (
        "[2] RANDOM FOREST  (bagging ensemble expert — ROC-AUC 0.921)\n"
        f"    Prediction       : {rf_decision}\n"
        f"    Churn probability : {rf_prob}%\n"
    )
    if rf_imp_lines is not None:
        lines += f"    Top global features (importance): {rf_imp_lines}\n"
    return lines


def agreement_block(row: pd.Series, invert: bool = False) -> str:
    xgb_pred, _, _ = _xgb_fields(row, invert=invert)
    rf_decision, _ = _rf_fields(row, invert=invert)
    preds = [xgb_pred, rf_decision]
    unique_preds = set(p for p in preds if p not in ("N/A",))
    if len(unique_preds) == 1:
        agreement = f"Full agreement — both models predict {list(unique_preds)[0]}"
    else:
        agreement = f"Disagreement — XGBoost predicts {xgb_pred}, Random Forest predicts {rf_decision}"
    return f"Panel status : {agreement}\n"


def override_guard_block(row: pd.Series, invert: bool = False) -> str:
    xgb_pred, _, _ = _xgb_fields(row, invert=invert)
    rf_decision, _ = _rf_fields(row, invert=invert)
    proba = float(row.get("y_proba", 0))
    if invert:
        proba = 1.0 - proba
    if xgb_pred == "CHURN" and rf_decision == "CHURN" and proba >= 0.80:
        return (
            "IMPORTANT: Both models agree on CHURN with ≥80% confidence. "
            "Override to 'retain' ONLY if there is clear overwhelming counter-evidence "
            "in the behavioral signals.\n"
        )
    return ""


JSON_SPEC_PANEL = (
    'Respond with ONLY a JSON object — no other text, no markdown fences:\n'
    '{"decision": "<churn|retain>", '
    '"explanation": "<2-3 sentences referencing panel signals, SHAP magnitudes and benchmarks>", '
    '"recommendation": "<1-2 sentences of actionable advice for the business owner>"}'
)

JSON_SPEC_GENERIC = (
    'Respond with ONLY a JSON object — no other text, no markdown fences:\n'
    '{"decision": "<churn|retain>", '
    '"explanation": "<2-3 sentences referencing the evidence available to you>", '
    '"recommendation": "<1-2 sentences of actionable advice for the business owner>"}'
)

_INSTRUCTIONS = {
    "panel": (
        "Based on all panel signals, determine the correct final churn decision, "
        "explain your reasoning clearly by referencing the model predictions, SHAP "
        "signals (magnitude labels indicate signal strength), population benchmarks, "
        "and behavioral data, and provide an actionable recommendation for "
        "the business owner on how to retain this customer or confirm churn.\n"
    ),
    "panel_strict": (
        "Based on all panel signals, determine the correct final churn decision, "
        "explain your reasoning clearly by referencing the model predictions, SHAP "
        "signals (magnitude labels indicate signal strength), population benchmarks, "
        "and behavioral data, and provide an actionable recommendation for "
        "the business owner on how to retain this customer or confirm churn. "
        "IMPORTANT: only the five numbered SHAP reasons listed under [1] XGBOOST "
        "are attributed evidence for why THIS customer was predicted churn or "
        "retain. The customer profile and behavioral signals sections above are "
        "background context, not attributed drivers -- do not cite any behavioral "
        "signal or profile field as a reason for the decision unless it also "
        "appears in the numbered SHAP reasons list.\n"
    ),
    "features_only": (
        "You are given only this customer's raw profile and behavioral data — no "
        "model prediction, no SHAP evidence, and no population benchmarks. Based "
        "solely on this data, independently decide whether this customer is likely "
        "to churn, explain your reasoning by referencing the specific data points "
        "above, and provide an actionable recommendation for the business owner.\n"
    ),
    "features_benchmarks": (
        "You are given this customer's raw profile, behavioral data, and population "
        "benchmarks (average values for churned vs. retained customers) — no model "
        "prediction and no SHAP evidence. Based solely on this data, independently "
        "decide whether this customer is likely to churn, explain your reasoning by "
        "referencing the specific data points and benchmarks above, and provide an "
        "actionable recommendation for the business owner.\n"
    ),
    "features_shap": (
        "You are given this customer's raw profile, behavioral data, population "
        "benchmarks, and SHAP feature-attribution signals showing which features "
        "push toward or away from churn and by how much — but you are NOT told what "
        "any predictive model decided. Based solely on this evidence, independently "
        "decide whether this customer is likely to churn, explain your reasoning by "
        "referencing the specific SHAP signals and benchmarks above, and provide an "
        "actionable recommendation for the business owner.\n"
    ),
    "prediction_only": (
        "You have no access to this customer's underlying data, features, or "
        "supporting evidence — only the prior model's verdict above. State whether "
        "you agree with this prediction, explain that you are relying solely on the "
        "model's stated confidence (acknowledge you have no independent evidence), "
        "and provide an actionable recommendation for the business owner consistent "
        "with the stated decision.\n"
    ),
}


def instruction_block(variant: str) -> str:
    return _INSTRUCTIONS[variant]


SYSTEM_PREAMBLE = (
    "You are a senior customer retention analyst reviewing the output of a "
    "two-model advisory panel that has already analyzed the customer below.\n\n"
)

SYSTEM_PREAMBLE_NO_PANEL = (
    "You are a senior customer retention analyst assessing the customer below.\n\n"
)
