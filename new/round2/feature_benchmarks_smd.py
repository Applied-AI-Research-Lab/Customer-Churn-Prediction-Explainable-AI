import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
from preprocessing import load_and_clean_data, split_data, TARGET

OUT = os.path.join(BASE, "new", "outputs", "round2", "feature_benchmarks_smd.csv")


def main() -> None:
    df = load_and_clean_data()
    train, _, _ = split_data(df)
    bench = json.load(open(os.path.join(BASE, "outputs", "llm_ready", "population_benchmarks.json")))
    rows = []
    for f in bench:
        c = train.loc[train[TARGET] == 1, f].dropna()
        r = train.loc[train[TARGET] == 0, f].dropna()
        sp = np.sqrt(((len(c) - 1) * c.var() + (len(r) - 1) * r.var()) / (len(c) + len(r) - 2))
        rows.append(dict(f=f, churn=c.mean(), ret=r.mean(), bench_c=bench[f]["churned_mean"],
                         bench_r=bench[f]["retained_mean"], diff=c.mean() - r.mean(), d=(c.mean() - r.mean()) / sp))
    t = pd.DataFrame(rows)
    t["ad"] = t["d"].abs()
    t = t.sort_values("ad", ascending=False)
    t.to_csv(OUT, index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
