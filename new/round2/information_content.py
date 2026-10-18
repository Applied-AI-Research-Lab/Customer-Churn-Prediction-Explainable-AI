from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "new"))
sys.path.insert(0, os.path.join(BASE, "new", "round2"))
from analyze_a9_followup import FEATURE_PATTERNS
from template_explainer import parse_reasons

OUT_DIR = os.path.join(BASE, "new", "outputs", "round2")

def features_in(text: str) -> set[str]:
    t = str(text).lower()
    return {f for f, p in FEATURE_PATTERNS.items() if re.search(p, t)}

def top5_ranked(row: pd.Series) -> list[str]:
    out = []
    for k in range(1, 6):
        found = features_in(row.get(f"reason_{k}", ""))
        out.append(next(iter(found)) if found else None)
    return out

def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and len(groups) > 1 else 0)
    return max(1, n)

def flesch(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?]+", str(text))))
    words = re.findall(r"[A-Za-z]+", str(text))
    if not words:
        return np.nan
    return 206.835 - 1.015 * len(words) / sentences - 84.6 * sum(map(syllables, words)) / len(words)

def score(text: str, ranked: list[str]) -> dict:
    cited = features_in(text)
    top5 = {f for f in ranked if f}
    return dict(top1_cited=ranked[0] in cited if ranked[0] else np.nan,
                top5_coverage=len(cited & top5) / len(top5) if top5 else np.nan,
                fabricated=len(cited - top5) > 0,
                words=len(str(text).split()), flesch=flesch(text))

def main() -> None:
    a5 = pd.read_csv(os.path.join(BASE, "outputs", "llm_predictions", "test_llm_predictions.csv"))
    a9 = pd.read_csv(os.path.join(BASE, "new", "outputs", "predictions", "A9_full_MERGED.csv")).set_index("row_id")
    tpl = pd.read_csv(os.path.join(OUT_DIR, "template_explanations.csv")).set_index("row_id")

    sources = {"Template": tpl["template_explanation"],
               "Qwen A5": a5["qwen_explanation"], "Gemma A5": a5["gemma_explanation"],
               "Qwen A9": a9["qwen_explanation"], "Gemma A9": a9["gemma_explanation"]}
    ranked = [top5_ranked(r) for _, r in a5.iterrows()]
    rows = []
    for name, texts in sources.items():
        s = pd.DataFrame([score(texts.loc[i], ranked[i]) for i in range(len(a5))])
        rows.append(dict(source=name, top1_cited=s.top1_cited.mean(), top5_coverage=s.top5_coverage.mean(),
                         fabricated=s.fabricated.mean(), words_mean=s.words.mean(), flesch_mean=s.flesch.mean()))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "information_content.csv"), index=False)

    bench = json.load(open(os.path.join(BASE, "outputs", "llm_ready", "population_benchmarks.json")))
    checks = []
    for _, r in a5.iterrows():
        for rs in parse_reasons(r):
            word = rs["phrase"].split(" ", 1)[0]
            f = rs["feature"]
            if word not in ("High", "Low") or f not in bench or pd.isna(r.get(f)):
                continue
            pop_mean = (bench[f]["churned_mean"] * 0.289 + bench[f]["retained_mean"] * 0.711)
            actual_high = float(r[f]) > pop_mean
            checks.append(dict(feature=f, contradicts=(word == "High") != actual_high))
    chk = pd.DataFrame(checks)
    summary = chk.groupby("feature")["contradicts"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    summary.to_csv(os.path.join(OUT_DIR, "reason_wording_check.csv"))

    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print(f"\nReason lines whose High/Low word contradicts the value vs the population mean: "
          f"{chk.contradicts.mean():.1%} of {len(chk)} reason lines")
    print(summary.head(8).round(3))

if __name__ == "__main__":
    main()
