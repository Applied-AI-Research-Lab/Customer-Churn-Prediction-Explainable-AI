"""
Generate LLM-ready decision & explanation datasets for churn prediction.

For each test-set customer, we produce:
  1. All original customer features (for context)
  2. Model prediction (y_pred) and confidence (y_proba)
  3. SHAP values per feature (model's explanation of *why*)
  4. Top 5 churn-driving reasons with SHAP magnitude (for the LLM to use)
  5. Risk tier (Critical / High / Medium / Low) based on churn probability
  6. RF global feature importances + population benchmarks (separate JSON files)

Output:
  outputs/llm_ready/llm_ready_xgboost.csv
  outputs/llm_ready/rf_feature_importances.json
  outputs/llm_ready/population_benchmarks.json

These files are designed to be fed to an LLM prompt so it can:
  - Confirm/adjust the final churn decision
  - Write a natural-language explanation for the business owner
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    load_and_clean_data, split_data, build_preprocessor,
    SPLIT_DIR, MODEL_DIR, TARGET,
)

LLM_DIR = os.path.join(os.path.dirname(MODEL_DIR), "llm_ready")
os.makedirs(LLM_DIR, exist_ok=True)

# Human-readable feature descriptions for plain-English explanations
FEATURE_DESCRIPTIONS = {
    "Age": "Age",
    "Membership_Years": "Membership duration (years)",
    "Login_Frequency": "Login frequency",
    "Session_Duration_Avg": "Average session duration",
    "Pages_Per_Session": "Pages browsed per session",
    "Cart_Abandonment_Rate": "Cart abandonment rate",
    "Wishlist_Items": "Wishlist items count",
    "Total_Purchases": "Total purchases",
    "Average_Order_Value": "Average order value",
    "Days_Since_Last_Purchase": "Days since last purchase",
    "Discount_Usage_Rate": "Discount usage rate",
    "Returns_Rate": "Returns rate",
    "Email_Open_Rate": "Email open rate",
    "Customer_Service_Calls": "Customer service calls",
    "Product_Reviews_Written": "Product reviews written",
    "Social_Media_Engagement_Score": "Social media engagement score",
    "Mobile_App_Usage": "Mobile app usage",
    "Payment_Method_Diversity": "Payment method diversity",
    "Lifetime_Value": "Lifetime value",
    "Credit_Balance": "Credit balance",
    "Gender": "Gender",
    "Country": "Country",
    "Signup_Quarter": "Signup quarter",
}

# Direction: does a HIGH value push toward churn (positive) or retention (negative)?
# Used to phrase the reason as "high X" (churn driver) or "low X" (engagement signal)
CHURN_DIRECTION = {
    "Age": -1,                          # younger customers churn more
    "Membership_Years": -1,             # longer membership → less churn
    "Login_Frequency": -1,             # more logins → less churn
    "Session_Duration_Avg": -1,        # longer sessions → less churn
    "Pages_Per_Session": -1,           # more pages → less churn
    "Cart_Abandonment_Rate": 1,        # higher abandonment → more churn
    "Wishlist_Items": -1,              # more wishlist items → engaged → less churn
    "Total_Purchases": -1,             # more purchases → less churn
    "Average_Order_Value": 1,          # higher order value (erratic) → more churn
    "Days_Since_Last_Purchase": 1,     # more days since last purchase → more churn
    "Discount_Usage_Rate": -1,         # higher discount usage → less churn (engaged)
    "Returns_Rate": 1,                 # higher returns → more churn
    "Email_Open_Rate": -1,             # higher email opens → less churn
    "Customer_Service_Calls": 1,       # more service calls → more churn
    "Product_Reviews_Written": -1,     # more reviews → less churn
    "Social_Media_Engagement_Score": -1,  # higher social engagement → less churn
    "Mobile_App_Usage": -1,            # higher app usage → less churn
    "Payment_Method_Diversity": 0,     # negligible
    "Lifetime_Value": -1,              # higher LTV → less churn
    "Credit_Balance": -1,              # higher credit balance → less churn
}


def get_feature_names(preprocessor, df):
    """Extract the output feature names from the ColumnTransformer."""
    num_cols = df.drop(columns=[TARGET]).select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.drop(columns=[TARGET]).select_dtypes(include=["object"]).columns.tolist()

    num_names = num_cols  # scaler preserves names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()

    return num_names + cat_names


def generate_top_reasons(shap_values_row, feature_names, feature_values, top_n=3):
    """Generate plain-English top churn-driving reasons from SHAP values."""
    # Pair each feature with its SHAP value and original value
    pairs = list(zip(feature_names, shap_values_row, feature_values))

    # Sort by absolute SHAP value (most impactful first)
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, shap_val, raw_val in pairs[:top_n]:
        # Get base feature name (strip onehot prefix)
        base_feat = feat.split("_")[0] if "_" in feat and feat.split("_")[0] in FEATURE_DESCRIPTIONS else feat
        # For onehot features, use the full feature name
        if base_feat not in FEATURE_DESCRIPTIONS:
            for key in FEATURE_DESCRIPTIONS:
                if feat.startswith(key):
                    base_feat = key
                    break

        desc = FEATURE_DESCRIPTIONS.get(base_feat, feat)
        direction = CHURN_DIRECTION.get(base_feat, 0)

        # Determine if this feature pushes toward churn (positive SHAP) or retention (negative)
        if shap_val > 0:
            if direction == 1:
                reason = f"High {desc} ({raw_val}) increases churn risk"
            elif direction == -1:
                reason = f"Low {desc} ({raw_val}) increases churn risk"
            else:
                reason = f"{desc} = {raw_val} increases churn risk"
        else:
            if direction == 1:
                reason = f"Low {desc} ({raw_val}) reduces churn risk"
            elif direction == -1:
                reason = f"High {desc} ({raw_val}) reduces churn risk"
            else:
                reason = f"{desc} = {raw_val} reduces churn risk"

        reasons.append({"feature": feat, "shap_value": round(float(shap_val), 4),
                         "feature_value": raw_val, "reason": reason})

    return reasons


def main():
    # Load and prepare data
    df = load_and_clean_data()
    train_df, val_df, test_df = split_data(df)

    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Build and fit preprocessor fresh (avoids joblib version mismatch)
    preprocessor, _, _ = build_preprocessor(df)
    preprocessor.fit(X_train)
    X_test_p = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, df)

    # Explainer for tree models
    explainer = shap.TreeExplainer

    for model_name in ["xgboost"]:  # RF SHAP is prohibitively slow; RF importances extracted below
        print(f"\n{'='*60}")
        print(f"  Generating LLM-ready data for: {model_name}")
        print(f"{'='*60}")

        # Load trained model
        model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.joblib"))

        # Predictions
        y_pred = model.predict(X_test_p)
        y_proba = model.predict_proba(X_test_p)[:, 1]

        # SHAP values
        expl = explainer(model)
        shap_values = expl.shap_values(X_test_p)

        # For binary classifiers, shap_values may be a list [class0, class1] or a 3D array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # take churn class
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # (n_samples, n_features, 2) → take churn class

        print(f"  SHAP values shape: {shap_values.shape}")

        # Build enriched DataFrame
        enriched = X_test.copy().reset_index(drop=True)
        enriched["y_true"] = y_test.values
        enriched["y_pred"] = y_pred
        enriched["y_proba"] = y_proba.round(4)
        enriched["churn_decision"] = np.where(y_pred == 1, "CHURN", "RETAIN")

        # Add SHAP values per feature
        for i, feat in enumerate(feature_names):
            enriched[f"shap_{feat}"] = shap_values[:, i].round(4)

        # Generate top-5 plain-English reasons per customer (with SHAP magnitudes)
        orig_feat_names = X_test.columns.tolist()
        top_reasons_list = []
        for idx in range(len(enriched)):
            reasons = generate_top_reasons_simple(
                shap_values[idx], feature_names, X_test.iloc[idx], orig_feat_names
            )
            top_reasons_list.append(reasons[:5])

        # Risk tier based on churn probability
        def _risk_tier(p):
            if p >= 0.80: return "Critical"
            if p >= 0.60: return "High"
            if p >= 0.40: return "Medium"
            return "Low"
        enriched["risk_tier"] = [_risk_tier(p) for p in y_proba]

        # Top-5 reasons with SHAP magnitudes
        enriched["top_5_reasons"] = [
            " | ".join([r["reason_with_magnitude"] for r in reasons]) for reasons in top_reasons_list
        ]
        for k in range(1, 6):
            enriched[f"reason_{k}"] = [
                r[k-1]["reason_with_magnitude"] if len(r) >= k else "" for r in top_reasons_list
            ]

        # Reorder columns: features first, then predictions, then SHAP, then reasons
        feature_cols = X_test.columns.tolist()
        pred_cols = ["y_true", "y_pred", "y_proba", "churn_decision", "risk_tier"]
        shap_cols = [f"shap_{f}" for f in feature_names]
        reason_cols = ["top_5_reasons"] + [f"reason_{k}" for k in range(1, 6)]
        enriched = enriched[feature_cols + pred_cols + shap_cols + reason_cols]

        output_path = os.path.join(LLM_DIR, f"llm_ready_{model_name}.csv")
        enriched.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}")
        print(f"  Shape: {enriched.shape[0]} rows × {enriched.shape[1]} columns")
        print(f"  Columns: {enriched.shape[1]} total "
              f"({len(feature_cols)} features + {len(pred_cols)} predictions + "
              f"{len(shap_cols)} SHAP + {len(reason_cols)} reasons)")

        # Preview first 3 rows (selected columns)
        preview_cols = feature_cols[:5] + ["y_proba", "churn_decision", "risk_tier",
                                           "reason_1", "reason_2", "reason_3"]
        print(f"\n  Preview (first 3 rows, selected columns):")
        print(enriched[preview_cols].head(3).to_string(index=False))

    # ── RF global feature importances ────────────────────────────────────────
    print("\nExtracting RF global feature importances...")
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
    preprocessor2, _, _ = build_preprocessor(df)
    preprocessor2.fit(train_df.drop(columns=[TARGET]))
    feat_names_rf = get_feature_names(preprocessor2, df)
    importances = rf_model.feature_importances_
    rf_imp = sorted(zip(feat_names_rf, importances), key=lambda x: x[1], reverse=True)[:10]
    rf_imp_dict = {feat: round(float(imp), 4) for feat, imp in rf_imp}
    rf_imp_path = os.path.join(LLM_DIR, "rf_feature_importances.json")
    import json
    with open(rf_imp_path, "w") as f:
        json.dump(rf_imp_dict, f, indent=2)
    print(f"  Saved → {rf_imp_path}")
    for feat, imp in rf_imp[:5]:
        print(f"    {feat:40s}: {imp:.4f}")

    # ── Population benchmarks (mean per churn class, top features) ────────────
    print("\nComputing population benchmarks (mean per churn class)...")
    df_clean = load_and_clean_data()
    train_clean, _, _ = split_data(df_clean)
    num_feat_names = train_clean.drop(columns=[TARGET]).select_dtypes(include=[np.number]).columns.tolist()
    benchmarks = {}
    for feat in num_feat_names:
        churned_mean  = train_clean.loc[train_clean[TARGET] == 1, feat].mean()
        retained_mean = train_clean.loc[train_clean[TARGET] == 0, feat].mean()
        benchmarks[feat] = {
            "churned_mean":  round(float(churned_mean), 3),
            "retained_mean": round(float(retained_mean), 3),
        }
    benchmarks_path = os.path.join(LLM_DIR, "population_benchmarks.json")
    with open(benchmarks_path, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"  Saved → {benchmarks_path} ({len(benchmarks)} features)")

    print(f"\n✅ LLM-ready datasets generated in: {LLM_DIR}")


def generate_top_reasons_simple(shap_row, feature_names, original_row, original_feat_names):
    """Generate top reasons mapping SHAP values back to original feature names."""
    # For numerical features, SHAP feature name = original feature name
    # For categorical (onehot), SHAP feature name = "Feature_Value"
    # We map back to original feature for readability

    # Aggregate SHAP values for onehot features back to original categorical feature
    shap_by_orig = {}
    for shap_feat, sv in zip(feature_names, shap_row):
        matched = False
        for orig_feat in original_feat_names:
            if shap_feat == orig_feat or shap_feat.startswith(orig_feat + "_"):
                if orig_feat not in shap_by_orig:
                    shap_by_orig[orig_feat] = 0.0
                shap_by_orig[orig_feat] += abs(sv) if shap_feat.startswith(orig_feat + "_") else sv
                matched = True
                break
        if not matched:
            shap_by_orig[shap_feat] = sv

    # Build pairs with original feature values
    pairs = []
    for feat in original_feat_names:
        if feat in shap_by_orig:
            pairs.append((feat, shap_by_orig[feat], original_row[feat]))

    # Sort by absolute SHAP value
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, shap_val, raw_val in pairs[:5]:
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        direction = CHURN_DIRECTION.get(feat, 0)

        if pd.isna(raw_val):
            raw_val_str = "missing"
        else:
            raw_val_str = str(round(raw_val, 2) if isinstance(raw_val, (int, float)) else raw_val)

        if shap_val > 0:
            if direction == 1:
                reason = f"High {desc} ({raw_val_str}) increases churn risk"
            elif direction == -1:
                reason = f"Low {desc} ({raw_val_str}) increases churn risk"
            else:
                reason = f"{desc} = {raw_val_str} increases churn risk"
        else:
            if direction == 1:
                reason = f"Low {desc} ({raw_val_str}) reduces churn risk"
            elif direction == -1:
                reason = f"High {desc} ({raw_val_str}) reduces churn risk"
            else:
                reason = f"{desc} = {raw_val_str} reduces churn risk"

            # Add SHAP magnitude label so the LLM can prioritise dominant vs. minor signals
        magnitude = abs(float(shap_val))
        if magnitude >= 1.5:
            strength = "dominant driver"
        elif magnitude >= 0.5:
            strength = "strong signal"
        elif magnitude >= 0.15:
            strength = "moderate signal"
        else:
            strength = "weak signal"
        reason_with_mag = f"{reason} [SHAP={float(shap_val):+.2f}, {strength}]"

        reasons.append({
            "feature": feat,
            "shap_value": round(float(shap_val), 4),
            "feature_value": raw_val_str,
            "reason": reason,
            "reason_with_magnitude": reason_with_mag,
        })

    return reasons


if __name__ == "__main__":
    main()