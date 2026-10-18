import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIT = os.path.join(BASE, "faithfulness_audit")
OUT = os.path.join(BASE, "new", "outputs", "round2", "round1_audit_cis.csv")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    f = pd.read_csv(os.path.join(AUDIT, "final_resolved_faithfulness.csv"))
    items = pd.read_csv(os.path.join(AUDIT, "results", "coding_sheet_coderA.csv"))[["item_id", "risk_tier"]]
    tier = pd.read_csv(os.path.join(BASE, "outputs", "llm_predictions", "test_llm_predictions.csv"))["risk_tier"]
    items["w"] = items["risk_tier"].map(tier.value_counts() / items["risk_tier"].value_counts())
    f = f.merge(items, on="item_id")
    f["faith"] = (f["final"].str.lower() == "yes").astype(int)
    rows = []
    for m, g in f.groupby("model"):
        k, n = g["faith"].sum(), len(g)
        lo, hi = wilson(k, n)
        rows.append(dict(audit="Round-1 A5 holistic faithful", model=m, k=k, n=n, rate=k / n, ci_lo=lo, ci_hi=hi,
                         weighted=np.average(g["faith"], weights=g["w"])))
    s = pd.read_csv(os.path.join(AUDIT, "a9_validation", "spotcheck_results.csv"))
    for _, r in s.iterrows():
        k = int(round(r["fabrication_rate"] * r["n"]))
        lo, hi = wilson(k, int(r["n"]))
        rows.append(dict(audit=f"A9 spot-check fabrication ({r['coder']})", model=r["model"] + "-" + r["condition"],
                         k=k, n=int(r["n"]), rate=k / r["n"], ci_lo=lo, ci_hi=hi, weighted=np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
