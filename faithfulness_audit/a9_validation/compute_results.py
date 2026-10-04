import argparse
import os

import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SLOTS = ["A", "B", "C", "D"]


def cohens_kappa(y1: pd.Series, y2: pd.Series) -> float:
    mask = y1.notna() & y2.notna() & (y1 != "") & (y2 != "")
    y1, y2 = y1[mask], y2[mask]
    n = len(y1)
    if n == 0:
        return float("nan")
    labels = sorted(set(y1) | set(y2))
    po = (y1.values == y2.values).mean()
    pe = sum((y1 == lab).mean() * (y2 == lab).mean() for lab in labels)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coderA", required=True)
    parser.add_argument("--coderB", required=True)
    args = parser.parse_args()

    a = pd.read_csv(args.coderA)
    b = pd.read_csv(args.coderB)
    key = pd.read_csv(os.path.join(OUT_DIR, "_answer_key_DO_NOT_SHARE.csv"))
    assert list(a["item_id"]) == list(b["item_id"])

    print("=" * 70)
    print("  Inter-rater reliability (pooled across all 4 slots)")
    print("=" * 70)
    ya = pd.concat([a[f"fabricated_{s}"] for s in SLOTS], ignore_index=True).astype(str).str.strip().str.lower()
    yb = pd.concat([b[f"fabricated_{s}"] for s in SLOTS], ignore_index=True).astype(str).str.strip().str.lower()
    k = cohens_kappa(ya, yb)
    print(f"  kappa={k:.3f}   raw agreement={(ya == yb).mean():.1%}   n={len(ya)}")

    print("\n" + "=" * 70)
    print("  Fabrication rate by model x condition (unblinded)")
    print("=" * 70)
    merged = a.merge(key, on="item_id")
    rows = []
    for coder_name, df in [("coderA", a), ("coderB", b)]:
        m = df.merge(key, on="item_id")
        for model in ("qwen", "gemma"):
            for cond in ("A5", "A9"):
                vals = []
                for _, r in m.iterrows():
                    for s in SLOTS:
                        if r[f"text_{s}_model"] == model and r[f"text_{s}_condition"] == cond:
                            vals.append(str(r[f"fabricated_{s}"]).strip().lower())
                rate = (pd.Series(vals) == "yes").mean()
                rows.append({"coder": coder_name, "model": model, "condition": cond,
                             "fabrication_rate": rate, "n": len(vals)})
                print(f"  [{coder_name}] {model:6s} {cond}: {rate:.1%}  (n={len(vals)})")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "spotcheck_results.csv"), index=False)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'spotcheck_results.csv')}")


if __name__ == "__main__":
    main()
