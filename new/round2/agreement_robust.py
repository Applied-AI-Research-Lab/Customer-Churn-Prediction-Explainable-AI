from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R1 = os.path.join(BASE, "faithfulness_audit")
R2 = os.path.join(BASE, "faithfulness_audit", "post_mitigation")
OUT = os.path.join(BASE, "new", "outputs", "round2", "agreement_robust.csv")
TAGS = ["fabricated", "reversed", "magnitude", "causal", "recommendation"]

def stats(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.astype(int), b.astype(int)
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    pi = (pa + pb) / 2
    pe_g = 2 * pi * (1 - pi)
    return dict(n=len(a), raw_agreement=po,
                kappa=(po - pe) / (1 - pe) if pe < 1 else np.nan,
                pabak=2 * po - 1,
                gwet_ac1=(po - pe_g) / (1 - pe_g) if pe_g < 1 else np.nan,
                prevalence_mean=float(pi))

def audit1() -> list[dict]:
    a, b = (pd.read_csv(os.path.join(R1, "results", f"coding_sheet_coder{c}.csv")) for c in "AB")
    rows = []

    def pooled(df, col):
        return pd.concat([df[f"{col}_A"], df[f"{col}_B"]], ignore_index=True).astype(str).str.strip().str.lower()

    fa, fb = pooled(a, "faithful"), pooled(b, "faithful")
    rows.append(dict(audit=1, code="faithful", **stats((fa == "yes").to_numpy(), (fb == "yes").to_numpy())))
    ta, tb = pooled(a, "failure_tags").replace("nan", ""), pooled(b, "failure_tags").replace("nan", "")
    for t in TAGS:
        xa = (ta.str.contains(t) & (fa == "no")).to_numpy()
        xb = (tb.str.contains(t) & (fb == "no")).to_numpy()
        rows.append(dict(audit=1, code=t, **stats(xa, xb)))
    return rows

def audit2() -> list[dict]:
    a, b = (pd.read_csv(os.path.join(R2, f"coding_sheet_coder{c}.csv")).set_index("text_id") for c in "AB")
    b = b.loc[a.index]
    rows = []
    for c in ["faithful"] + [f"dim_{t}" for t in TAGS]:
        xa = a[c].astype(str).str.strip().str.lower().eq("yes").to_numpy()
        xb = b[c].astype(str).str.strip().str.lower().eq("yes").to_numpy()
        rows.append(dict(audit=2, code=c.replace("dim_", ""), **stats(xa, xb)))
    return rows

def mdd(n: int = 60, alpha: float = 0.05, power: float = 0.8, discordant_frac: float = 0.4,
        n_sim: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for frac in (0.2, 0.25, discordant_frac):
        for diff in np.arange(0.05, 0.401, 0.05):
            if diff > frac:
                continue
            p_up = (frac + diff) / 2
            hits = 0
            for _ in range(n_sim):
                k = rng.binomial(n, frac)
                up = rng.binomial(k, p_up / frac) if k else 0
                hits += k > 0 and binomtest(up, k, 0.5).pvalue < alpha
            out.append(dict(n=n, discordant_frac=frac, difference=round(diff, 2), power=hits / n_sim))
    return pd.DataFrame(out)

def main() -> None:
    df = pd.DataFrame(audit1() + audit2())
    df.to_csv(OUT, index=False)
    pw = mdd()
    pw.to_csv(OUT.replace("agreement_robust", "mcnemar_power"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print()
    print(pw.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
