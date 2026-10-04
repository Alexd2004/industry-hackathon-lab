"""Results tab (step 12): ladder table and tiles, the cap slider, the policy panel (tiles, ranked list,
account detail card), the Tier 3 crew panel (step 18), CI note, footer.

Every number comes from a results file. The ladder reads eval.csv (or the placeholder). The policy panel
reads ranked.csv, policy_grid.csv and contrib.csv (python -m softsignal.explain): the slider looks its
thresholds up in policy_grid.csv and re-bands ranked.csv with explain.apply_bands, so a move needs no
model, no cache and no label. The Tier 3 panel reads eval_tier3.csv and tier3_latency.csv
(python -m softsignal.tier3_report). The tab only reads files.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from softsignal.explain import ACTIONS, BANDS, FEATURE_NAMES, FRAME_COLS, SLIDER_CAPS, apply_bands, chip
from softsignal.features import ID_COL
from softsignal.metrics import CONTRIB_COLS, DEFAULT_CAP, EVAL_COLS, POLICY_GRID_COLS, RANKED_COLS
from softsignal.policy import POLICY_FILE, SOFT_CAPPED, load_policy
from softsignal.tier3_report import LATENCY_COLS, budget_line

RESULTS = Path(__file__).resolve().parents[1] / "results"
EVAL_CSV = RESULTS / "eval.csv"
PLACEHOLDER_CSV = RESULTS / "eval_placeholder.csv"
NUMERIC_COLS = [c for c in EVAL_COLS if c not in ("stage", "eval_set")]
TIER3_QUALITY_FILE, TIER3_LATENCY_FILE = "eval_tier3.csv", "tier3_latency.csv"
LATENCY_TEXT_COLS = ("row", "runs", "note")
NO_TIER3 = ("No Tier 3 rows yet. Run `python -m softsignal.tier3_report --make-rule` and then "
            "`python -m softsignal.tier3_report` to write results/eval_tier3.csv.")
NO_LATENCY = ("No latency table yet. `python -m softsignal.tier3_report --latency` writes results/tier3_latency.csv "
              "from the per-call log of a live run.")
POLICY_PATH = POLICY_FILE  # read once per rerun (tests point it elsewhere)
RANKED_FILE, GRID_FILE, CONTRIB_FILE = "ranked.csv", "policy_grid.csv", "contrib.csv"
N_SHOWN = 15
VIEWS = ("Top of the list", "Around the verify cutoff")
SENT, QUEUED = "sent now", "queued (over budget)"
BAND_BG = {"verify": "#f6c39b", "soft": "#fbe3a0", "none": ""}  # light tints, dark text: readable on a projector
NO_FILES = ("No ranked list yet. Run `python -m softsignal.stack`, then `python -m softsignal.explain`, "
            "to write results/ranked.csv, policy_grid.csv and contrib.csv.")

BANNER = "PLACEHOLDER, projected, not measured. These are Combined Plan section 7 numbers, not results from this repo."
FT_HUE, REC_HUE = "#d95f02", "#1b6ca8"  # false-teen and cap band share one hue; recall another (both tabs)
REC_CI, FT_CI = 0.025, 0.033  # the plan's 95% CI half-widths: recall at 0.92 on 450 teens, false-teen at 0.15 on 450 adults
CI_NOTE = (
    f"95% CI (Combined Plan section 7): recall +/- {REC_CI * 100:.1f} pts at 92% on 450 teens, "
    f"false-teen +/- {FT_CI * 100:.1f} pts at 15% on 450 adults (the 900 test accounts). "
    "The width changes with the rate, so it is only exact at those points. "
    "Rows marked frozen 600 were scored on a different set and are not covered. "
    "The cap is a band, not a line."
)
FOOTER = (
    "Text: 2004 public blogs (self-reported ages, label noise possible). "
    "Activity: synthetic_calibrated_demo, ~12% contrarians. "
    "Verification: simulated from the answer key. Likelihood, not legal age."
)


def load_ladder(results_dir: Path | None = None) -> tuple[pd.DataFrame, bool]:
    """(ladder, is_placeholder). The real eval.csv wins; the placeholder is used only when it is missing."""
    results_dir = RESULTS if results_dir is None else results_dir
    real, placeholder = results_dir / "eval.csv", results_dir / "eval_placeholder.csv"
    path, is_placeholder = (real, False) if real.exists() else (placeholder, True)
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as e:
        raise ValueError(f"{path.name} is not a readable CSV: {e}") from e
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"{path.name} is empty") from e
    except OSError as e:  # missing file, a folder named eval.csv, no read permission
        raise ValueError(f"{path.name} cannot be read: {e.strerror or e}") from e
    missing = [c for c in EVAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    df = df[EVAL_COLS].copy()
    for col in NUMERIC_COLS:
        num = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & num.isna()  # a non-number must fail loudly, not turn into a blank
        if bad.any():
            raise ValueError(f"{path.name}: column {col} has non-numeric values {df.loc[bad, col].tolist()[:3]}")
        df[col] = num
    return df, is_placeholder


def best_under_cap(ladder: pd.DataFrame, cap: float = DEFAULT_CAP) -> pd.Series | None:
    """Row with the highest recall among rows whose false-teen rate is within the cap, else None."""
    ok = ladder[ladder["ft"] <= cap + 1e-9].dropna(subset=["rec"])
    return None if ok.empty else ok.loc[ok["rec"].idxmax()]


def headline_rows(ladder: pd.DataFrame, is_placeholder: bool) -> pd.DataFrame:
    """Rows the tile and chart may use: all placeholder rows (all projected), else held-out test rows only."""
    return ladder if is_placeholder else ladder[ladder["eval_set"] == "test"]


def _pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:.1%}"


# --- Tier 3: crew against the rule loop (step 18) -----------------------------------------

def load_tier3(results_dir: Path | None = None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(quality rows, latency table); each None while its file is not written. A bad file raises ValueError."""
    results_dir = RESULTS if results_dir is None else results_dir
    q_path, l_path = results_dir / TIER3_QUALITY_FILE, results_dir / TIER3_LATENCY_FILE
    quality = _read(q_path, EVAL_COLS, NUMERIC_COLS) if q_path.exists() else None
    latency = None
    if l_path.exists():
        latency = _read_blanks_ok(l_path, LATENCY_COLS, [c for c in LATENCY_COLS if c not in LATENCY_TEXT_COLS])
        hit = latency[latency["row"] == "round_time"]
        if len(hit) != 1 or hit[["n", "p50_ms", "p95_ms"]].isna().any(axis=None):
            raise ValueError(f"{l_path.name} needs exactly one round_time row with n, p50_ms and p95_ms")
    return quality, latency


def _read_blanks_ok(path: Path, cols: list[str], numeric: list[str]) -> pd.DataFrame:
    """Like _read, but a blank number stays blank (an agent with no LIVE call has no p95); text must still parse."""
    try:
        df = pd.read_csv(path, dtype={c: str for c in LATENCY_TEXT_COLS}, keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"{path.name} is empty") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"{path.name} is not a readable CSV: {e}") from e
    except OSError as e:
        raise ValueError(f"{path.name} cannot be read: {e.strerror or e}") from e
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    df = df[cols].copy()
    for col in numeric:
        num = pd.to_numeric(df[col], errors="coerce")
        if (df[col].notna() & num.isna()).any() or (num.notna() & ~np.isfinite(num.fillna(0))).any():
            raise ValueError(f"{path.name}: column {col} must be numbers or blank")
        df[col] = num
    for col in ("runs", "note"):
        df[col] = df[col].fillna("")
    return df


def _stage_row(quality: pd.DataFrame, prefix: str) -> pd.Series | None:
    hit = quality[quality["stage"].str.startswith(prefix)]
    return None if hit.empty else hit.iloc[0]


def tier3_verdict(quality: pd.DataFrame) -> str | None:
    """One plain sentence comparing the live crew run with the rule loop at the last round, computed from the rows.
    None when either row is missing. It says lower, higher or equal on recall and never claims a win."""
    rule, live = _stage_row(quality, "loop_rule_"), _stage_row(quality, "loop_crew_live_")
    if rule is None or live is None:
        return None
    diff = live["rec"] - rule["rec"]
    rel = "lower than" if diff < -1e-9 else "higher than" if diff > 1e-9 else "equal to"
    return (f"At the last round the live crew run's recall is {_pct(live['rec'])} (false-teen {_pct(live['ft'])}), "
            f"{rel} the rule loop's {_pct(rule['rec'])} (false-teen {_pct(rule['ft'])}). "
            "One run of each, so this shows what happened once, not how often.")


def render_tier3_panel() -> None:
    st.subheader("Tier 3: agent crew against the rule loop")
    try:
        quality, latency = load_tier3()
    except ValueError as e:
        st.error(f"Cannot show the Tier 3 rows: {e}")
        return
    if quality is None:
        st.info(NO_TIER3)
    else:
        st.dataframe(quality, hide_index=True, width="stretch")
        verdict = tier3_verdict(quality)
        if verdict is not None:
            st.caption(verdict + " Rows are the final round on the frozen 900 test accounts (eval_tier3.csv).")
    if latency is None:
        st.info(NO_LATENCY)
        return
    st.dataframe(latency.drop(columns=["runs"]), hide_index=True, width="stretch")
    st.caption(budget_line(latency) + ". LIVE model calls only; with fewer than 20 values p95 is the maximum. "
               f"Measured on {' and '.join(sorted(set(' '.join(latency['runs']).split()))) or 'no run'} "
               "(tier3_latency.csv).")


# --- policy panel: files ------------------------------------------------------------------

@dataclass
class PolicyFiles:
    ranked: pd.DataFrame  # RANKED_COLS, rank order
    grid: pd.DataFrame  # POLICY_GRID_COLS, one row per cap
    contrib: pd.DataFrame | None  # CONTRIB_COLS; None when contrib.csv is missing (detail card only)


def _read(path: Path, cols: list[str], numeric: list[str]) -> pd.DataFrame:
    """A results CSV with these columns, numeric ones checked (finite), or a ValueError naming the problem."""
    try:
        # round_trip: the default parser can be 1 ulp off, and t_verify / t_budget sit 1 ulp above a score
        df = pd.read_csv(path, dtype={ID_COL: str, "flags": str}, keep_default_na=False, na_values=[""],
                         float_precision="round_trip")
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"{path.name} is empty") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"{path.name} is not a readable CSV: {e}") from e
    except OSError as e:  # a folder where the file should be, no read permission
        raise ValueError(f"{path.name} cannot be read: {e.strerror or e}") from e
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    df = df[cols].copy()
    for col in numeric:
        num = pd.to_numeric(df[col], errors="coerce")
        if num.isna().any() or not np.isfinite(num).all():
            raise ValueError(f"{path.name}: column {col} must be numbers in every row")
        df[col] = num
    return df


@st.cache_data(show_spinner=False, max_entries=8)
def _read_cached(path: str, stamp: tuple, cols: tuple, numeric: tuple) -> pd.DataFrame:
    return _read(Path(path), list(cols), list(numeric))


def _load(path: Path, cols: list[str], numeric: list[str]) -> pd.DataFrame:
    """Cached by path, size and modification time, so a slider move re-reads nothing."""
    try:
        st_ = path.stat()
    except OSError as e:
        raise ValueError(f"{path.name} cannot be read: {e.strerror or e}") from e
    return _read_cached(str(path), (st_.st_mtime_ns, st_.st_size), tuple(cols), tuple(numeric)).copy()


def load_policy_files(results_dir: Path | None = None) -> PolicyFiles | None:
    """The policy panel's files, or None while ranked.csv or policy_grid.csv is not written yet."""
    results_dir = RESULTS if results_dir is None else results_dir
    r_path, g_path, c_path = results_dir / RANKED_FILE, results_dir / GRID_FILE, results_dir / CONTRIB_FILE
    if not (r_path.exists() and g_path.exists()):
        return None
    ranked = _load(r_path, RANKED_COLS, ["rank", "score"])
    if ranked[ID_COL].duplicated().any():
        raise ValueError(f"{RANKED_FILE}: {ID_COL} must be unique")
    if ((ranked["score"] < 0) | (ranked["score"] > 1)).any():
        raise ValueError(f"{RANKED_FILE}: score must be between 0 and 1")
    ranked = ranked.sort_values("rank", kind="stable").reset_index(drop=True)
    for i in range(1, 4):  # an account with fewer than 3 contributions has blank chips (read as NaN)
        ranked[[f"c{i}", f"f{i}"]] = ranked[[f"c{i}", f"f{i}"]].fillna("")
    ranked["words"] = ranked["words"].fillna("")
    grid = _load(g_path, POLICY_GRID_COLS, [c for c in POLICY_GRID_COLS if c not in ("flags", "budget_binding")])
    grid["flags"] = grid["flags"].fillna("")
    grid["budget_binding"] = grid["budget_binding"].astype(str).str.lower() == "true"
    if grid["cap"].round(4).duplicated().any():
        raise ValueError(f"{GRID_FILE} must have one row per cap")
    contrib = _load(c_path, CONTRIB_COLS, ["score", "intercept", "raw", "z", "contrib"]) if c_path.exists() else None
    return PolicyFiles(ranked, grid, contrib)


def grid_row(grid: pd.DataFrame, cap: float) -> pd.Series | None:
    """The policy_grid.csv row for this cap, or None if the grid has no such cap."""
    hit = grid[(grid["cap"] - cap).abs() < 1e-9]
    return None if hit.empty else hit.iloc[0]


def reband(ranked: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """ranked.csv re-banded at a grid row's thresholds: band, action and reason from explain.apply_bands, plus
    the review queue (verify-band accounts at or above t_budget are sent now; the rest of the band is queued)."""
    out = apply_bands(ranked[FRAME_COLS], float(row["t_soft"]), float(row["t_verify"]))
    out.insert(0, "rank", ranked["rank"].to_numpy())
    verify = out["band"] == "verify"
    out["queue"] = np.where(verify & (out["score"] >= row["t_budget"]), SENT, np.where(verify, QUEUED, ""))
    return out


def grid_mismatches(banded: pd.DataFrame, row: pd.Series) -> list[str]:
    """Counts in the re-banded list that differ from the grid row (files written by different runs)."""
    got = {"n": len(banded), "n_flagged": int((banded["band"] == "verify").sum()),
           "n_soft": int((banded["band"] == "soft").sum()), "n_none": int((banded["band"] == "none").sum()),
           "n_verify": int((banded["queue"] == SENT).sum())}
    return [f"{k} {got[k]} vs {int(row[k])}" for k in got if got[k] != int(row[k])]


def shown_rows(banded: pd.DataFrame, view: str, n: int = N_SHOWN) -> pd.DataFrame:
    """The top n accounts, or n accounts centred on the last one in the verify band (where a cap move shows)."""
    if view == VIEWS[0]:
        return banded.head(n)
    n_verify = int((banded["band"] == "verify").sum())
    start = min(max(n_verify - n // 2, 0), max(len(banded) - n, 0))
    return banded.iloc[start:start + n]


def list_table(rows: pd.DataFrame) -> pd.DataFrame:
    """What the ranked list shows. Words only for flagged accounts: an unflagged account's words are not evidence."""
    words = rows["words"].where(rows["band"] != "none", "")
    return pd.DataFrame({
        "rank": rows["rank"], ID_COL: rows[ID_COL], "score": rows["score"].round(3), "band": rows["band"],
        "action": rows["action"], "review": rows["queue"], "signal 1": rows["c1"], "signal 2": rows["c2"],
        "signal 3": rows["c3"], "teen-leaning words": words,
    })


def _band_style(table: pd.DataFrame):
    color = table["band"].map(BAND_BG)
    return table.style.apply(
        lambda col: [f"background-color: {c}; color: #111" if c else "" for c in color], subset=["band", "action"]
    ).format({"score": "{:.3f}"})


def account_detail(contrib: pd.DataFrame, account: str) -> pd.DataFrame:
    """One account's contributions from contrib.csv, largest first, with its readable name and chip."""
    rows = contrib[contrib[ID_COL] == account].copy()
    rows["name"] = rows["feature"].map(FEATURE_NAMES).fillna(rows["feature"])
    rows["signal"] = [chip(f, c, z, r) if f in FEATURE_NAMES else "" for f, c, z, r
                      in zip(rows["feature"], rows["contrib"], rows["z"], rows["raw"])]
    rows["direction"] = np.where(rows["contrib"] >= 0, "toward teen", "toward adult")
    return rows.reindex(rows["contrib"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def detail_chart(detail: pd.DataFrame) -> alt.Chart:
    return alt.Chart(detail, title="Signed contributions to the score (logit scale)").mark_bar().encode(
        x=alt.X("contrib:Q", title="contribution (+ toward teen, - toward adult)"),
        y=alt.Y("name:N", sort=None, title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["toward teen", "toward adult"],
                                                       range=[FT_HUE, REC_HUE]), legend=alt.Legend(title=None)),
        tooltip=["name", "signal", alt.Tooltip("raw:Q", format=".3f"), alt.Tooltip("z:Q", format=".2f"),
                 alt.Tooltip("contrib:Q", format="+.2f")],
    )


def read_policy() -> tuple[dict | None, str | None]:
    """(policy.yaml, None), or (None, why) when it cannot be read: the tab then uses the default cap."""
    try:
        return load_policy(POLICY_PATH), None
    except (OSError, ValueError) as e:
        return None, str(e)


def slider_bounds(files: PolicyFiles | None) -> tuple[int, int]:
    """Slider range in whole percent: the caps policy_grid.csv holds, else the caps explain.py writes."""
    caps = files.grid["cap"] if files is not None else pd.Series(SLIDER_CAPS)
    return round(caps.min() * 100), round(caps.max() * 100)


# --- policy panel: render -----------------------------------------------------------------

def render_policy_panel(cap: float, files: PolicyFiles | None, error: str | None, budget: float | None) -> None:
    st.subheader("Policy and ranked likely-teen list")
    if error is not None:
        st.error(f"Cannot show the ranked list: {error}")
        return
    if files is None:
        st.info(NO_FILES)
        return
    row = grid_row(files.grid, cap)
    if row is None:
        st.warning(f"{GRID_FILE} has no row for a {cap:.0%} cap. Rerun python -m softsignal.explain.")
        return
    banded = reband(files.ranked, row)
    off = grid_mismatches(banded, row)
    if off:
        st.warning(f"{RANKED_FILE} and {GRID_FILE} disagree ({'; '.join(off)}): they were written by different "
                   "runs. Rerun python -m softsignal.explain.")

    n_queued = int(row["n_flagged"] - row["n_verify"])
    budget_txt = f"{budget:.0%} review budget" if budget is not None else "review budget"
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Flagged for verification", f"{int(row['n_flagged'])} ({row['flagged_share']:.0%})")
    t1.caption(f"score >= t_verify {row['t_verify']:.3f}")
    t2.metric("Recall, flagged", _pct(row["rec_flagged"]))
    t2.caption("Teens in the verify band")
    t3.metric("False-teen, flagged", _pct(row["ft_flagged"]))
    t3.caption(f"Training (OOF) {_pct(row['oof_ft_flagged'])}; cap {cap:.0%} +/- {FT_CI * 100:.1f} pts on held-out")
    t4.metric("Teen-safe defaults (soft band)", str(int(row["n_soft"])))
    t4.caption(f"t_soft {row['t_soft']:.3f} <= score < t_verify")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Sent to verification now", str(int(row["n_verify"])))
    u1.caption(f"{budget_txt}; score >= {row['t_budget']:.3f}")
    u2.metric("Recall, sent now", _pct(row["rec_sent"]))
    u3.metric("False-teen, sent now", _pct(row["ft_sent"]))
    u4.metric("Precision, sent now", _pct(row["prec_sent"]))
    notes = [f"Rates on the {int(row['n'])} held-out test accounts (labels used for these totals only; "
             "the list below carries none). They are for reporting: the cap is a policy choice (policy.yaml), "
             "never tuned on them."]
    if row["budget_binding"]:
        notes.append(f"The verify band is over the {budget_txt}: the top {int(row['n_verify'])} by score are sent "
                     f"now and the other {n_queued} stay flagged, queued for review (policy.py: the cap wins, "
                     "the band is never cut).")
    if SOFT_CAPPED in str(row["flags"]).split(","):
        notes.append("Soft band empty at this cap: t_soft would sit above t_verify, because the teens it was "
                     "meant to catch already score above t_verify (policy.py soft_capped).")
    st.caption(" ".join(notes))

    counts = pd.DataFrame({
        "group": [f"verify: {SENT}", f"verify: {QUEUED}", "soft: teen-safe defaults", "none: no action"],
        "accounts": [int(row["n_verify"]), n_queued, int(row["n_soft"]), int(row["n_none"])],
        "order": [0, 1, 2, 3],
    })
    st.altair_chart(alt.Chart(counts, height=90).mark_bar().encode(
        x=alt.X("sum(accounts):Q", title="accounts", stack="zero"),
        color=alt.Color("group:N", sort=list(counts["group"]),
                        scale=alt.Scale(domain=list(counts["group"]),
                                        range=[FT_HUE, BAND_BG["verify"], BAND_BG["soft"], "#cccccc"]),
                        legend=alt.Legend(title=None, orient="bottom")),
        order="order:Q", tooltip=["group", "accounts"]), width="stretch")

    view = st.radio("Show", VIEWS, horizontal=True, key="ranked_view")
    st.dataframe(_band_style(list_table(shown_rows(banded, view))), hide_index=True, width="stretch")
    st.caption("Signals are the account's three largest signed contributions (logit scale). Actions: "
               + "; ".join(f"{b} = {ACTIONS[b]}" for b in reversed(BANDS)) + ". A likelihood, not legal age.")

    st.markdown("**Account detail**")
    if files.contrib is None:
        st.info(f"{CONTRIB_FILE} is missing, so the detail card is off. Rerun python -m softsignal.explain.")
        return
    labels = [f"#{r} {i} (score {s:.3f})" for r, i, s in zip(banded["rank"], banded[ID_COL], banded["score"])]
    pick = st.selectbox("Account", range(len(banded)), format_func=lambda k: labels[k], key="detail_account")
    acc = banded.iloc[pick]
    detail = account_detail(files.contrib, acc[ID_COL])
    if detail.empty:
        st.warning(f"{acc[ID_COL]} has no rows in {CONTRIB_FILE}. Rerun python -m softsignal.explain.")
        return
    d1, d2, d3 = st.columns(3)
    d1.metric("Score p(teen)", f"{acc['score']:.3f}")
    d2.metric("Band", acc["band"])
    d3.metric("Action", acc["action"])
    st.caption(acc["reason"] + (f" Review: {acc['queue']}." if acc["queue"] else ""))
    st.altair_chart(detail_chart(detail), width="stretch")
    intercept, total = float(detail["intercept"].iloc[0]), float(detail["contrib"].sum())
    logit = intercept + total
    exact = math.isclose(1.0 / (1.0 + math.exp(-logit)), float(detail["score"].iloc[0]), abs_tol=1e-9)
    st.caption(f"logit(score) = intercept {intercept:+.2f} + sum of contributions {total:+.2f} = {logit:+.2f} "
               + ("(exact: the model is a logistic regression on these inputs)." if exact
                  else "(does NOT match the score: contrib.csv and ranked.csv are from different runs)."))
    st.dataframe(detail[["name", "signal", "raw", "z", "contrib"]].round(3), hide_index=True, width="stretch")


def render_results_tab() -> None:
    try:
        ladder, is_placeholder = load_ladder()
    except ValueError as e:
        st.error(f"Cannot show results: {e}")
        st.caption(FOOTER)
        return
    if is_placeholder:
        st.warning(BANNER)

    st.subheader("Results ladder")
    st.dataframe(ladder, hide_index=True, width="stretch")
    if not is_placeholder and (ladder["eval_set"] == "projected").any():
        st.warning("eval.csv contains rows marked projected. They are shown in the table but not in the tile or chart.")

    tag = " (projected)" if is_placeholder else ""
    shown = headline_rows(ladder, is_placeholder)
    pol, pol_error = read_policy()
    if pol_error is not None:
        st.warning(f"policy.yaml cannot be used ({pol_error}); the slider starts at {DEFAULT_CAP:.0%}.")
    try:
        files, files_error = load_policy_files(), None
    except ValueError as e:
        files, files_error = None, str(e)
    lo, hi = slider_bounds(files)
    default = min(max(round((pol or {}).get("cap_false_teen", DEFAULT_CAP) * 100), lo), hi)
    cap_pct = st.slider("False-teen cap", lo, hi, default, 1, format="%d%%",
                        key="cap_pct", help="The share of adults you accept sending to teen mode. Re-bands the list "
                        "below from policy_grid.csv; the ladder tile picks the best row within the cap.")
    cap = cap_pct / 100
    best = best_under_cap(shown, cap)
    c1, c2, c3 = st.columns(3)
    if best is None:
        c1.metric(f"Best recall under cap{tag}", "n/a")
    else:
        c1.metric(f"Best recall under cap{tag}", _pct(best["rec"]))
        c1.caption(str(best["stage"]))
        c2.metric(f"False-teen at that row{tag}", _pct(best["ft"]))
    c3.metric("Cap", _pct(cap))
    c3.caption(f"Band: {_pct(max(cap - FT_CI, 0.0))} to {_pct(cap + FT_CI)} (the plan's CI at 15%)")
    if best is not None:
        c2.caption("Point estimate. A value on the cap is inside its CI band.")

    points = shown.dropna(subset=["rec", "ft"])
    chart = (
        alt.Chart(points, title=f"Recall vs false-teen{tag}")
        .mark_circle(size=90)
        .encode(
            x=alt.X("ft:Q", title="false-teen rate"),
            y=alt.Y("rec:Q", title="recall"),
            tooltip=["stage", "eval_set", alt.Tooltip("rec:Q", format=".1%"), alt.Tooltip("ft:Q", format=".1%")],
        )
    )
    band = pd.DataFrame({"lo": [max(cap - FT_CI, 0.0)], "hi": [cap + FT_CI], "cap": [cap]})
    cap_band = alt.Chart(band).mark_rect(opacity=0.15).encode(x="lo:Q", x2="hi:Q")
    cap_line = alt.Chart(band).mark_rule(strokeDash=[4, 4]).encode(x="cap:Q")
    st.altair_chart(cap_band + chart + cap_line, width="stretch")

    st.caption(CI_NOTE)
    st.divider()
    render_tier3_panel()
    st.divider()
    render_policy_panel(cap, files, files_error, (pol or {}).get("review_budget"))
    st.divider()
    st.caption(FOOTER)
