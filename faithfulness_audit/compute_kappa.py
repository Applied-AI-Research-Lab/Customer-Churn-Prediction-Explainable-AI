import argparse
import os

import pandas as pd

TAGS = ["fabricated", "reversed", "magnitude", "causal", "recommendation"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def cohens_kappa(y1: pd.Series, y2: pd.Series) -> float:
    mask = y1.notna() & y2.notna() & (y1 != "") & (y2 != "")
    y1, y2 = y1[mask], y2[mask]
    n = len(y1)
    if n == 0:
        return float("nan")
    labels = sorted(set(y1) | set(y2))
    po = (y1.values == y2.values).mean()
    pe = sum((y1 == lab).mean() * (y2 == lab).mean() for lab in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def pooled(df: pd.DataFrame, col_prefix: str) -> pd.Series:
    return pd.concat([df[f"{col_prefix}_A"], df[f"{col_prefix}_B"]], ignore_index=True).astype(str).str.strip().str.lower()


def tag_present(faithful: pd.Series, tags: pd.Series, tag: str) -> pd.Series:
    tags = tags.fillna("").astype(str).str.lower()
    present = tags.str.contains(tag)
    present = present & (faithful == "no")
    return present.map({True: "yes", False: "no"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coderA", required=True)
    parser.add_argument("--coderB", required=True)
    args = parser.parse_args()

    a = pd.read_csv(args.coderA)
    b = pd.read_csv(args.coderB)
    assert list(a["item_id"]) == list(b["item_id"]), "Item order/IDs don't match between the two sheets"

    print("=" * 80)
    print("  PRIMARY: inter-rater reliability on the holistic 'faithful' call")
    print("=" * 80)
    fa = pooled(a, "faithful")
    fb = pooled(b, "faithful")
    k_primary = cohens_kappa(fa, fb)
    print(f"  faithful (yes/no)   kappa={k_primary:.3f}   raw agreement={(fa == fb).mean():.1%}   n={len(fa)}")

    print("\n" + "=" * 80)
    print("  SECONDARY: per-failure-type binary kappa (present/absent)")
    print("=" * 80)
    tags_a = pooled(a, "failure_tags")
    tags_b = pooled(b, "failure_tags")
    kappa_rows = [{"dimension": "faithful_overall", "kappa": k_primary, "n": len(fa)}]
    for tag in TAGS:
        ta = tag_present(fa, tags_a, tag)
        tb = tag_present(fb, tags_b, tag)
        k = cohens_kappa(ta, tb)
        n_flagged = int(((ta == "yes") | (tb == "yes")).sum())
        print(f"  {tag:<15} kappa={k:.3f}   raw agreement={(ta == tb).mean():.1%}   "
              f"(flagged by >=1 coder: {n_flagged}/{len(ta)})")
        kappa_rows.append({"dimension": tag, "kappa": k, "n": len(ta), "n_flagged_by_either": n_flagged})

    pd.DataFrame(kappa_rows).to_csv(os.path.join(OUT_DIR, "kappa_results.csv"), index=False)

    item_ids = pd.concat([a["item_id"] + "_A", a["item_id"] + "_B"], ignore_index=True)
    disagreements = pd.DataFrame({
        "item_slot": item_ids, "coderA_faithful": fa, "coderB_faithful": fb,
        "coderA_tags": tags_a, "coderB_tags": tags_b,
    })
    disagreements = disagreements[disagreements["coderA_faithful"] != disagreements["coderB_faithful"]]
    disagreements.to_csv(os.path.join(OUT_DIR, "disagreements_for_adjudication.csv"), index=False)
    print(f"\n  {len(disagreements)}/{len(fa)} items disagree on the primary faithful call -> "
          f"saved for adjudication.")

    key_path = os.path.join(OUT_DIR, "_answer_key_DO_NOT_SHARE.csv")
    if os.path.exists(key_path):
        key = pd.read_csv(key_path)
        merged = a.merge(key, on="item_id")
        print("\n" + "=" * 80)
        print("  Per-model faithfulness rate (coder A's codes, unblinded)")
        print("=" * 80)
        for model in ("qwen", "gemma"):
            vals = []
            for _, r in merged.iterrows():
                slot = "A" if r["text_A_model"] == model else "B"
                vals.append(str(r[f"faithful_{slot}"]).strip().lower())
            rate = (pd.Series(vals) == "yes").mean()
            print(f"  {model:<8} faithful rate: {rate:.1%}  (n={len(vals)})")


if __name__ == "__main__":
    main()
