from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_CSV = os.path.join(BASE, "outputs", "llm_predictions", "test_llm_predictions.csv")
OUT = os.path.join(BASE, "new", "outputs", "round2", "fairness_check.csv")
NUM = ["Membership_Years", "Login_Frequency", "Session_Duration_Avg", "Pages_Per_Session",
       "Cart_Abandonment_Rate", "Wishlist_Items", "Total_Purchases", "Average_Order_Value",
       "Days_Since_Last_Purchase", "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate",
       "Customer_Service_Calls", "Product_Reviews_Written", "Social_Media_Engagement_Score",
       "Mobile_App_Usage", "Payment_Method_Diversity", "Lifetime_Value", "Credit_Balance"]

def main() -> None:
    t = pd.read_csv(TEST_CSV)
    t = t[t["y_proba"] >= 0.5].reset_index(drop=True)
    rows = []
    for model in ("qwen", "gemma"):
        y = (t[f"{model}_decision"].str.strip().str.lower() == "retain").astype(int).to_numpy()
        rates = {f"rate_{g}": float(y[(t["Gender"] == g).to_numpy()].mean()) for g in ("Female", "Male", "Other")}
        if y.sum() == 0:
            rows.append(dict(model=model, n=len(y), overrides=0, **rates))
            continue
        lp = np.log(t["y_proba"] / (1 - t["y_proba"] + 1e-9))
        P = StandardScaler().fit_transform(np.column_stack([lp, lp ** 2, lp ** 3]))
        F = StandardScaler().fit_transform(t[NUM].fillna(t[NUM].median()))
        C = pd.get_dummies(t[["Country", "Signup_Quarter"]].astype(str), drop_first=True).astype(float).to_numpy()
        G = pd.get_dummies(t["Gender"], drop_first=True).astype(float).to_numpy()
        A = ((t["Age"] - t["Age"].mean()) / t["Age"].std()).fillna(0).to_numpy()[:, None]
        Am = t["Age"].isna().astype(float).to_numpy()[:, None]

        def ll(X: np.ndarray) -> float:
            m = LogisticRegression(C=1e6, max_iter=20000).fit(X, y)
            p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
            return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        full = ll(np.column_stack([P, F, C, Am, G, A]))
        lr_g = 2 * (full - ll(np.column_stack([P, F, C, Am, A])))
        lr_a = 2 * (full - ll(np.column_stack([P, F, C, Am, G])))
        tert = pd.qcut(t["Age"], 3)
        rows.append(dict(model=model, n=len(y), overrides=int(y.sum()), **rates,
                         gender_lr=lr_g, gender_p=float(chi2.sf(lr_g, G.shape[1])),
                         age_lr=lr_a, age_p=float(chi2.sf(lr_a, 1)),
                         **{f"rate_age_{k}": float(v) for k, v in pd.Series(y).groupby(tert.values).mean().items()}))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(4).T.to_string())

if __name__ == "__main__":
    main()
