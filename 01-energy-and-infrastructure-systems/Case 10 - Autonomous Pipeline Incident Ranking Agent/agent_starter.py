"""Pipeline ranking: count-only vs count x consequence; then heavier consequence."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "cer_pipeline_incidents_alberta_2015.csv"
WEIGHT = {"high": 3.0, "medium": 1.5, "low": 1.0}
WEIGHT_REVISE = {"high": 6.0, "medium": 1.5, "low": 1.0}
TOP = 15


def top(g, score, label):
    ranked = g.assign(score=score).sort_values("score", ascending=False).head(TOP)
    high_n = int((ranked["consequence"] == "high").sum())
    print(
        f"{label:36s}  high_in_top{TOP}={high_n}  "
        f"incidents_in_list={int(ranked['n'].sum())}  "
        f"mean_count={ranked['n'].mean():.1f}"
    )
    return set(ranked.index), ranked


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    missing = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    print(f"Dropped {missing} rows with no date. Scoring {len(df)} incidents.")
    g = df.groupby("corridor").agg(
        n=("date", "size"),
        company=("company", lambda s: s.value_counts().index[0]),
        consequence=("consequence", lambda s: s.value_counts().index[0]),
    )
    g["w"] = g["consequence"].map(WEIGHT)
    g["w2"] = g["consequence"].map(WEIGHT_REVISE)

    a, _ = top(g, g["n"], "Baseline: count only")
    b, ranked = top(g, g["n"] * g["w"], "v1: count x consequence")
    c, _ = top(g, g["n"] * g["w2"], "revise: heavier high consequence")
    print(f"Overlap baseline vs v1: {len(a & b)}/{TOP}")
    print(f"Overlap v1 vs revise:   {len(b & c)}/{TOP}")
    print("\nTop of v1:")
    print(ranked[["company", "consequence", "n", "score"]].head(8).to_string())


if __name__ == "__main__":
    main()
