"""Results tab (step 12 skeleton): ladder table, tiles, cap slider stub, CI note, footer."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from softsignal.metrics import DEFAULT_CAP, EVAL_COLS

RESULTS = Path(__file__).resolve().parents[1] / "results"
EVAL_CSV = RESULTS / "eval.csv"
PLACEHOLDER_CSV = RESULTS / "eval_placeholder.csv"

BANNER = "PLACEHOLDER, projected, not measured. These are Combined Plan section 7 numbers, not results from this repo."
CI_NOTE = "95% CI on 900 test accounts: recall +/- 2.5 pts, false-teen +/- 3.3 pts. The cap is a band, not a line."
FOOTER = (
    "Text: 2004 public blogs (self-reported ages, label noise possible). "
    "Activity: synthetic_calibrated_demo, ~12% contrarians. "
    "Verification: simulated from the answer key. Likelihood, not legal age."
)


def load_ladder(results_dir: Path = RESULTS) -> tuple[pd.DataFrame, bool]:
    """(ladder, is_placeholder). The real eval.csv wins; the placeholder is used only when it is missing."""
    real, placeholder = results_dir / "eval.csv", results_dir / "eval_placeholder.csv"
    path, is_placeholder = (real, False) if real.exists() else (placeholder, True)
    df = pd.read_csv(path)
    missing = [c for c in EVAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    return df[EVAL_COLS], is_placeholder


def best_under_cap(ladder: pd.DataFrame, cap: float = DEFAULT_CAP) -> pd.Series | None:
    """Row with the highest recall among rows whose false-teen rate is within the cap, else None."""
    ok = ladder[ladder["ft"] <= cap + 1e-9].dropna(subset=["rec"])
    return None if ok.empty else ok.loc[ok["rec"].idxmax()]


def _pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def render_results_tab() -> None:
    ladder, is_placeholder = load_ladder()
    if is_placeholder:
        st.warning(BANNER)

    st.subheader("Results ladder")
    st.dataframe(ladder, hide_index=True, use_container_width=True)

    cap = st.slider("False-teen cap (stub, not wired yet)", 0.05, 0.30, DEFAULT_CAP, 0.01, disabled=True)
    best = best_under_cap(ladder, cap)
    c1, c2, c3 = st.columns(3)
    if best is None:
        c1.metric("Best recall under cap", "n/a")
    else:
        c1.metric("Best recall under cap", _pct(best["rec"]), help=str(best["stage"]))
        c2.metric("False-teen at that row", _pct(best["ft"]))
        c3.metric("Cap", _pct(cap))

    chart = (
        alt.Chart(ladder.dropna(subset=["rec", "ft"]))
        .mark_circle(size=90)
        .encode(x=alt.X("ft:Q", title="false-teen rate"), y=alt.Y("rec:Q", title="recall"), tooltip=["stage", "rec", "ft"])
    )
    cap_line = alt.Chart(pd.DataFrame({"cap": [cap]})).mark_rule(strokeDash=[4, 4]).encode(x="cap:Q")
    st.altair_chart(chart + cap_line, use_container_width=True)

    st.caption(CI_NOTE)
    st.divider()
    st.caption(FOOTER)
