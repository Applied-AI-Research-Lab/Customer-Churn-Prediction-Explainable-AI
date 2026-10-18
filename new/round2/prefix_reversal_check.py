from __future__ import annotations

import json
import os
import sys

import pandas as pd
from scipy.stats import fisher_exact

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "new"))
sys.path.insert(0, os.path.join(BASE, "new", "round2"))
from information_content import features_in
from template_explainer import parse_reasons

AUDIT = os.path.join(BASE, "faithfulness_audit", "post_mitigation")
OUT = os.path.join(BASE, "new", "outputs", "round2", "prefix_reversal_check.csv")

def contradictory_features(row: pd.Series, bench: dict) -> set[str]:
    out = set()
    for rs in parse_reasons(row):
        word, feat = rs["phrase"].split(" ", 1)[0], rs["feature"]
        if word not in ("High", "Low") or feat not in bench or pd.isna(row.get(feat)):
            continue
        pop_mean = bench[feat]["churned_mean"] * 0.289 + bench[feat]["retained_mean"] * 0.711
        if (word == "High") != (float(row[feat]) > pop_mean):
            out.add(feat)
    return out

def main() -> None:
    bench = json.load(open(os.path.join(BASE, "outputs", "llm_ready", "population_benchmarks.json")))
    a5 = pd.read_csv(os.path.join(BASE, "outputs", "llm_predictions", "test_llm_predictions.csv"))
    f = pd.read_csv(os.path.join(AUDIT, "final_resolved.csv")).merge(
        pd.read_csv(os.path.join(AUDIT, "coding_sheet_coderA.csv"))[["text_id", "explanation"]], on="text_id")
    f["cites_contradictory"] = [bool(features_in(e) & contradictory_features(a5.iloc[int(r)], bench))
                                for e, r in zip(f["explanation"], f["row_id"])]
    rows = []
    for name, g in [("all", f), ("gemma", f[f["model"] == "gemma"]), ("qwen", f[f["model"] == "qwen"])]:
        t = pd.crosstab(g["cites_contradictory"], g["dim_reversed"]).reindex(index=[True, False], columns=[1, 0], fill_value=0)
        rows.append(dict(group=name,
                         reversed_if_cites=t.loc[True, 1], n_cites=int(t.loc[True].sum()),
                         reversed_if_not=t.loc[False, 1], n_not=int(t.loc[False].sum()),
                         rate_if_cites=t.loc[True, 1] / t.loc[True].sum(),
                         rate_if_not=t.loc[False, 1] / t.loc[False].sum(),
                         fisher_p=fisher_exact(t.values).pvalue))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
