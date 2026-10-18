from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = ["faithful", "dim_fabricated", "dim_reversed", "dim_magnitude", "dim_causal", "dim_recommendation"]
KEY = os.path.join(HERE, "_answer_key_DO_NOT_SHARE.csv")

def load_sheet(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(HERE, name))
    for c in CODES:
        df[c] = df[c].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    return df.set_index("text_id")

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h

def agreement() -> None:
    a, b = load_sheet("coding_sheet_coderA.csv"), load_sheet("coding_sheet_coderB.csv")
    missing = {n: int(s[CODES].isna().sum().sum()) for n, s in (("coderA", a), ("coderB", b))}
    if any(missing.values()):
        print("WARNING blank or invalid cells:", missing)
    for n, s in (("coderA", a), ("coderB", b)):
        bad = s[(s["faithful"] == 1) & (s[CODES[1:]].sum(axis=1) > 0)]
        if len(bad):
            print(f"WARNING {n}: {len(bad)} texts marked faithful but with a dimension = yes:", list(bad.index))
    rows = []
    for c in CODES:
        m = a[c].notna() & b[c].notna()
        rows.append(dict(code=c, n=int(m.sum()), raw_agreement=float((a.loc[m, c] == b.loc[m, c]).mean()),
                         kappa=cohen_kappa_score(a.loc[m, c], b.loc[m, c]),
                         prevalence_coderA=float(a.loc[m, c].mean()), prevalence_coderB=float(b.loc[m, c].mean())))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "kappa_results.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    dims = CODES[1:]
    disagree = a.index[(a[dims] != b[dims]).any(axis=1)]
    yn = {1: "yes", 0: "no"}
    adj = a.loc[disagree, ["customer", "evidence_shown_to_model", "decision", "explanation", "recommendation"]].copy()
    adj.insert(0, "decide", [", ".join(c.replace("dim_", "") for c in dims if a.at[t, c] != b.at[t, c]) for t in disagree])
    for c in dims:
        agree = a.loc[disagree, c] == b.loc[disagree, c]
        adj[f"coderA_{c}"] = a.loc[disagree, c].map(yn)
        adj[f"coderB_{c}"] = b.loc[disagree, c].map(yn)
        adj[f"final_{c}"] = a.loc[disagree, c].map(yn).where(agree, "")
    adj["adjudication_reasoning"] = ""
    adj_path = os.path.join(HERE, "adjudication_sheet.csv")
    if os.path.exists(adj_path):
        adj_path = os.path.join(HERE, "adjudication_sheet_blank.csv")
    adj.reset_index().to_csv(adj_path, index=False)
    n_cells = int((a.loc[disagree, dims] != b.loc[disagree, dims]).sum().sum())
    print(f"\n{len(disagree)} of {len(a)} texts have at least one dimension disagreement "
          f"({n_cells} decisions in total) -> adjudication_sheet.csv (no model/condition labels; "
          f"agreed dimensions pre-filled; column 'decide' lists what to rule on)")

def results() -> None:
    a = load_sheet("coding_sheet_coderA.csv")
    adj = pd.read_csv(os.path.join(HERE, "adjudication_sheet.csv")).set_index("text_id")
    final = a[CODES].copy()
    for c in CODES[1:]:
        col = adj[f"final_{c}"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        missing = adj.index[col.isna()]
        if len(missing):
            raise SystemExit(f"adjudication_sheet.csv: final_{c} is blank for {list(missing)}")
        final.loc[col.index, c] = col
    final["faithful"] = (final[CODES[1:]].sum(axis=1) == 0).astype(int)
    if os.path.exists(KEY):
        key = pd.read_csv(KEY).set_index("text_id")
    else:
        key = pd.read_csv(os.path.join(HERE, "final_resolved.csv")).set_index("text_id")[
            ["customer", "row_id", "risk_tier", "weight", "model", "condition"]]
    final = final.join(key)
    final.reset_index().to_csv(os.path.join(HERE, "final_resolved.csv"), index=False)

    rows = []
    for (model, cond), g in final.groupby(["model", "condition"]):
        for c in CODES:
            k, n = int(g[c].sum()), int(g[c].notna().sum())
            lo, hi = wilson(k, n)
            rows.append(dict(model=model, condition=cond, code=c, rate=k / n, ci_lo=lo, ci_hi=hi,
                             weighted_rate=float(np.average(g[c], weights=g["weight"])), n=n))
    table = pd.DataFrame(rows)

    paired = []
    for model, g in final.groupby("model"):
        w = g.pivot_table(index="customer", columns="condition", values=CODES)
        for c in CODES:
            x5, x9 = w[(c, "A5")], w[(c, "A9")]
            b, d = int(((x5 == 1) & (x9 == 0)).sum()), int(((x5 == 0) & (x9 == 1)).sum())
            p = binomtest(b, b + d, 0.5).pvalue if b + d else np.nan
            paired.append(dict(model=model, code=c, A5_only=b, A9_only=d, mcnemar_exact_p=p))
    table = table.merge(pd.DataFrame(paired), on=["model", "code"], how="left")
    table.to_csv(os.path.join(HERE, "results_by_model_condition.csv"), index=False)
    print(table.round(3).to_string(index=False))

if __name__ == "__main__":
    {"agreement": agreement, "results": results}[sys.argv[1] if len(sys.argv) > 1 else "agreement"]()
