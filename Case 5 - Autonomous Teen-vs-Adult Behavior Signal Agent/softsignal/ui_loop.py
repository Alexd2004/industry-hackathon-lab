"""Loop tab (step 12): header, tiles, round chart, agent cards, decision log, footer.

Reads the live files results/rounds.csv and decisions.jsonl (gitignored; appended by loop.py / crew.py runs)
together with the committed recorded run (rounds_recorded.csv, decisions_recorded.jsonl), so a rehearsal never
hides the recorded run (it is marked "(recorded)" in the picker); the placeholders only when neither exists. While a run started here is going, the tab re-reads the files every REFRESH_S seconds (a
fragment), so rounds show up as they land; otherwise it does not poll. "Run loop" starts crew.py in a
background thread; the tab itself only reads, never writes anything an agent reads, and never calls a model.
Replay is still a stub.
"""
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from softsignal import crew
from softsignal.agents.contracts import TEST_METRIC_KEYS
from softsignal.explain import FEATURE_NAMES
from softsignal.metrics import DEFAULT_CAP, ROUNDS_COLS
from softsignal.ui_results import FOOTER, FT_CI, FT_HUE, REC_HUE, headline_rows, load_ladder

RESULTS = Path(__file__).resolve().parents[1] / "results"
ROUNDS_FILE, DECISIONS_FILE = "rounds.csv", "decisions.jsonl"
ROUNDS_RECORDED, DECISIONS_RECORDED = "rounds_recorded.csv", "decisions_recorded.jsonl"
ROUNDS_PLACEHOLDER, DECISIONS_PLACEHOLDER = "rounds_placeholder.csv", "decisions_placeholder.jsonl"

BANNER = ("PLACEHOLDER, projected, not measured. Loop numbers are a hand-made path between the plan's "
          "R0 and R7 rows (eval_placeholder.csv), not results from this repo.")
AGENTS = ("a1", "a2", "a3", "a4", "a5")
AGENT_NAMES = {"a1": "A1 drift", "a2": "A2 policy", "a3": "A3 patterns", "a4": "A4 reviewer note", "a5": "A5 claim check"}
MODES = {"SHADOW", "ACTIVE"}
SOURCES = {"A2", "rule", "starter"}
STATUSES = {"LIVE", "FALLBACK", "REPLAY", None}
BADGE_COLOR = {"LIVE": "green", "FALLBACK": "orange", "REPLAY": "gray", "PLACEHOLDER": "gray",
               "RULE ONLY": "gray", "SHADOW": "gray", "ACTIVE": "blue"}
REFRESH_S = 1  # the tab re-reads the files this often (UI handover section 5)
NUMERIC = [c for c in ROUNDS_COLS if c not in ("run", "mode", "action", "applied_source")]
BLANK_OK = {"t_soft", "audit_ft", "psi", "refit_s"}
INTS = {"round", "diff_count", "n_flagged", "n_verify", "n_labels", "n_audit_adults"}
RATES = {"cap", "audit_ft", "prec", "rec", "ft", "mt", "auc"}
DIFF_ROWS = ("blend_w", "cutoff", "cap", "action")
LAST_ROUND = 7
INSUFFICIENT_TEXT = {"default": "insufficient_data: not enough labels yet (normal early on, not an error).",
                     "a1": "insufficient_data: no earlier batch to compare with yet (normal in rounds 0 and 1).",
                     "a4": "insufficient_data: no accounts were sent to verification in this batch."}


@dataclass
class LoopData:
    rounds: pd.DataFrame
    decisions: list  # one dict per (run, round), last line wins, sorted by (run, round)
    is_placeholder: bool
    n_skipped: int
    warnings: list = field(default_factory=list)
    source: str = "live"  # live / recorded / placeholder / none (live when live files exist, even with recorded)
    recorded: frozenset = frozenset()  # run ids that come from the committed recorded run


def read_complete_lines(path: Path) -> list[str]:
    """Lines of a file a writer may be appending to: a last line with no trailing newline is dropped."""
    try:
        text = path.read_text()
    except OSError as e:  # a folder where the file should be, no read permission
        raise ValueError(f"{path.name} cannot be read: {e.strerror or e}") from e
    lines = text.split("\n")
    return lines[:-1]  # the piece after the last \n is either "" or a half-written line


def empty_rounds() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object if c in ("run", "mode", "action", "applied_source") else float)
                         for c in ROUNDS_COLS})


def load_rounds(path: Path, warnings: list) -> pd.DataFrame:
    lines = [ln for ln in read_complete_lines(path) if ln.strip()]
    if not lines:
        raise ValueError(f"{path.name} is empty")
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines) + "\n"), dtype=str, keep_default_na=False)
    except pd.errors.ParserError as e:
        raise ValueError(f"{path.name} is not a readable CSV: {e}") from e
    missing = [c for c in ROUNDS_COLS if c not in df.columns]
    if missing:
        hint = ""
        if "diff_count" in missing:
            hint = " It is from an older schema: regenerate it (move it aside and run the loop again)."
        raise ValueError(f"{path.name} is missing columns {missing}.{hint}")
    df = df[ROUNDS_COLS].apply(lambda s: s.str.strip())
    for col in ("run", "mode", "action", "applied_source"):
        if (df[col] == "").any():
            raise ValueError(f"{path.name}: column {col} has blank values")
    for col in NUMERIC:
        blank = df[col] == ""
        num = pd.to_numeric(df[col].mask(blank), errors="coerce")
        bad = ~blank & num.isna()  # a non-number must fail loudly, not turn into a blank
        if bad.any():
            raise ValueError(f"{path.name}: column {col} has non-numeric values {df.loc[bad, col].tolist()[:3]}")
        if blank.any() and col not in BLANK_OK:
            raise ValueError(f"{path.name}: column {col} has blank values")
        if col in INTS and ((num < 0) | (num % 1 != 0)).any():
            raise ValueError(f"{path.name}: column {col} must be whole numbers >= 0")
        if col in RATES and ((num < 0) | (num > 1)).any():
            raise ValueError(f"{path.name}: column {col} has rates outside [0, 1]")
        df[col] = num
    df[list(INTS)] = df[list(INTS)].astype(int)
    if not df["mode"].isin(MODES).all():
        raise ValueError(f"{path.name}: column mode must be one of {sorted(MODES)}")
    if not df["applied_source"].isin(SOURCES).all():
        raise ValueError(f"{path.name}: column applied_source must be one of {sorted(SOURCES)}")
    dup = df.duplicated(["run", "round"], keep="last")
    if dup.any():
        warnings.append(f"{path.name}: {int(dup.sum())} repeated (run, round) rows; the last one is shown.")
        df = df[~dup]
    return df.sort_values(["run", "round"]).reset_index(drop=True)


def _valid_agent(block) -> bool:
    if block is None:
        return True  # agent still working
    return (isinstance(block, dict) and {"status", "output", "fallback_reason"} <= block.keys()
            and block["status"] in STATUSES)


def valid_decision(rec) -> bool:
    """Handover decisions.jsonl shape. No test metrics may appear in it."""
    if not isinstance(rec, dict) or TEST_METRIC_KEYS & rec.keys():
        return False
    if not (isinstance(rec.get("run"), str) and rec["run"]):
        return False
    rnd = rec.get("round")
    if not isinstance(rnd, int) or isinstance(rnd, bool) or rnd < 0:
        return False
    if any(not _valid_agent(rec.get(k)) for k in AGENTS):  # a missing agent (e.g. a 3-agent crew) = not run
        return False
    applied = rec.get("applied")
    if not isinstance(rec.get("rule_decision"), dict) or not isinstance(applied, dict):
        return False
    if not isinstance(applied.get("decision"), dict) or applied.get("source") not in SOURCES:
        return False
    if not isinstance(rec.get("diff", {}), dict):
        return False
    # decision blocks carry parameters, never test metrics (A5's output may quote them; A1-A4 never read it)
    return not any(TEST_METRIC_KEYS & block.keys()
                   for block in (rec["rule_decision"], applied["decision"], rec.get("diff", {})))


def load_decisions(path: Path) -> tuple[list, int]:
    """(records, n_skipped). Bad JSON or invalid records are skipped and counted; same (run, round): last wins."""
    by_key, skipped = {}, 0
    for line in read_complete_lines(path):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not valid_decision(rec):
            skipped += 1
            continue
        rec.setdefault("diff", {})  # a missing agent key stays missing: "not run", unlike null ("working")
        by_key[(rec["run"], rec["round"])] = rec
    return [by_key[k] for k in sorted(by_key)], skipped


def _load_pair(r_path: Path, d_path: Path, warnings: list) -> tuple[pd.DataFrame, list, int]:
    """(rounds, decisions, n_skipped) of one pair; a missing file counts as empty."""
    rounds = load_rounds(r_path, warnings) if r_path.exists() else empty_rounds()
    decisions, skipped = load_decisions(d_path) if d_path.exists() else ([], 0)
    return rounds, decisions, skipped


def load_loop(results_dir: Path | None = None) -> LoopData:
    """The live runs together with the committed recorded run, so a rehearsal never hides the recorded one
    (a run in both: the live copy wins); the placeholders only when neither pair has a file."""
    results_dir = RESULTS if results_dir is None else results_dir
    pairs = {"live": (ROUNDS_FILE, DECISIONS_FILE), "recorded": (ROUNDS_RECORDED, DECISIONS_RECORDED)}
    present = [n for n, (r, d) in pairs.items() if (results_dir / r).exists() or (results_dir / d).exists()]
    warnings: list = []
    if not present:
        r_path, d_path = results_dir / ROUNDS_PLACEHOLDER, results_dir / DECISIONS_PLACEHOLDER
        source = "placeholder" if r_path.exists() or d_path.exists() else "none"
        rounds, decisions, skipped = _load_pair(r_path, d_path, warnings)
        return LoopData(rounds, decisions, source == "placeholder", skipped, warnings, source)
    loaded = {n: _load_pair(results_dir / pairs[n][0], results_dir / pairs[n][1], warnings) for n in present}
    frames = [loaded[n][0] for n in present if len(loaded[n][0])]
    rounds = (pd.concat(frames, ignore_index=True).drop_duplicates(["run", "round"])  # live first: it wins
              .sort_values(["run", "round"]).reset_index(drop=True)) if frames else empty_rounds()
    by_key = {(d["run"], d["round"]): d for n in reversed(present) for d in loaded[n][1]}  # live last: it wins

    def run_ids(name: str) -> set:
        r, d, _ = loaded.get(name, (empty_rounds(), [], 0))
        return set(r["run"]) | {x["run"] for x in d}

    recorded = frozenset(run_ids("recorded") - run_ids("live"))
    return LoopData(rounds, [by_key[k] for k in sorted(by_key)], False, sum(loaded[n][2] for n in present),
                    warnings, present[0], recorded)


def runs(data: LoopData) -> list[str]:
    """Run ids from both files, newest first (run ids are UTC timestamps, so they sort by time)."""
    return sorted(set(data.rounds["run"]) | {d["run"] for d in data.decisions}, reverse=True)


def for_run(data: LoopData, run: str | None) -> tuple[pd.DataFrame, list]:
    return data.rounds[data.rounds["run"] == run], [d for d in data.decisions if d["run"] == run]


def statuses(decisions: list) -> list:
    return [d[a]["status"] for d in decisions for a in AGENTS if d.get(a) is not None]


def run_badge(decisions: list, is_placeholder: bool) -> str:
    """What ran: REPLAY if any agent block was replayed, LIVE if any agent answered live, FALLBACK if agents
    ran but all fell back, RULE ONLY if no agent ran (a plain loop.py run). Never LIVE without a live call."""
    if is_placeholder:
        return "PLACEHOLDER"
    seen = statuses(decisions)
    for badge in ("REPLAY", "LIVE", "FALLBACK"):
        if badge in seen:
            return badge
    return "RULE ONLY"


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def diff_table(decision: dict) -> pd.DataFrame:
    """A2 vs the rule: rows blend_w/cutoff/cap/action; columns rule/A2/applied/changed."""
    out = (decision.get("a2") or {}).get("output")
    a2 = out if isinstance(out, dict) else {}
    rule, applied = decision["rule_decision"], decision["applied"]["decision"]
    rows = []
    for k in DIFF_ROWS:
        changed = k in decision["diff"] or (k in a2 and k in rule and a2[k] != rule[k])
        rows.append({"field": k, "rule": _fmt(rule.get(k)), "A2": _fmt(a2.get(k)),
                     "applied": _fmt(applied.get(k)), "changed": "YES" if changed else ""})
    return pd.DataFrame(rows)


def _pct(v) -> str:
    return f"{v:.0%}" if isinstance(v, (int, float)) and not isinstance(v, bool) else _fmt(v)


def log_line(decision: dict, round_row: pd.Series | None, tag: str = "") -> str:
    """One plain-English line per round for the decision log. tag marks the test metrics, e.g. " (projected)"."""
    d, src = decision["applied"]["decision"], decision["applied"]["source"]
    parts = [f"R{decision['round']}: applied {d.get('action', '?')} from {src}"]
    if d.get("cutoff") is not None:
        parts[0] += f", cutoff {_fmt(d['cutoff'])}"
    if d.get("cap") is not None:
        parts[0] += f", cap {_pct(d['cap'])}"
    if decision["diff"]:
        changes = ", ".join(f"{k} {_fmt(v[0])} -> {_fmt(v[1])}" if isinstance(v, list) and len(v) == 2 else k
                            for k, v in decision["diff"].items())
        parts.append(f"A2 differs from the rule on {changes}")
    fallbacks = [f"{a.upper()} ({decision[a]['fallback_reason'] or 'no reason'})" for a in AGENTS
                 if decision.get(a) is not None and decision[a]["status"] == "FALLBACK"]
    if fallbacks:
        parts.append("Fallback: " + ", ".join(fallbacks))
    if decision.get("agent_error"):
        parts.append(f"Agents failed, the rule decided: {decision['agent_error']}")
    a1 = (decision.get("a1") or {}).get("output")
    if isinstance(a1, dict) and a1.get("drift") in ("real", "not_real"):
        parts.append(f"A1 drift {a1['drift']}")
    ev = decision.get("evidence")
    if isinstance(ev, dict) and isinstance(ev.get("pooled_adults"), int) and ev.get("pooled_ft") is not None:
        parts.append(f"promote test: pooled audit false-teen {_pct(ev['pooled_ft'])} on {ev['pooled_adults']} "
                     f"adults, streak {ev.get('streak')}")
    if round_row is not None:
        parts.append(f"{round_row['mode']}, recall {round_row['rec']:.0%}, false-teen {round_row['ft']:.0%}{tag}")
    return ". ".join(parts) + "."


def _is_insufficient(block: dict) -> bool:
    out = block["output"]
    return (out == "insufficient_data" or block["fallback_reason"] == "insufficient_data"
            or (isinstance(out, dict) and out.get("drift") == "insufficient_data"))


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def plain(text) -> str:
    """Model-written text shown as typed: Markdown, links, images and $math$ are escaped, never rendered."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def _dicts(items) -> list[dict]:
    """The dict entries of a list; anything else (a model's malformed output) gives []."""
    return [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []


def _card_body(key: str, block: dict, decision: dict) -> None:
    out = block["output"]
    if key == "a1" and isinstance(out, dict):
        st.markdown(f"**Drift:** {plain(out.get('drift', '?'))}")
        st.caption(plain(out.get("reason", "")))
        ev = [f"{e.get('field')} = {e.get('value')}" for e in _dicts(out.get("evidence"))]
        if ev:
            st.caption("Evidence: " + plain("; ".join(ev)))
    elif key == "a3" and isinstance(out, dict):
        pats = _dicts(out.get("patterns"))
        st.markdown(f"**Top pattern:** {plain(pats[0].get('description', '?'))}" if pats else "No pattern found.")
        for ch in _dicts(out.get("suggested_param_changes")):
            st.caption(plain(f"Suggests {ch.get('param')} {ch.get('direction')}: {ch.get('reason', '')}"))
    elif key == "a2":
        if isinstance(out, dict):
            st.markdown(f"**Action:** {plain(out.get('action', '?'))}")
            st.caption(plain(out.get("reason", "")))
        st.dataframe(diff_table(decision), hide_index=True, width="stretch")
    elif key == "a4" and isinstance(out, dict):
        st.caption(plain(out.get("batch_reason", "")))  # no labels next to A4 notes
        cites = [FEATURE_NAMES.get(f, f) for f in out.get("based_on", []) if isinstance(f, str)]
        if cites:
            st.caption("Based on: " + plain(", ".join(cites)))
    elif key == "a5" and isinstance(out, list):
        verdicts = pd.Series([str(c.get("verdict")) for c in _dicts(out)]).value_counts()
        st.caption(", ".join(f"{n} {v}" for v, n in verdicts.items()) or "No claims checked.")


def agent_card(key: str, decision: dict | None, placeholder: bool = False) -> None:
    with st.container(border=True):
        st.markdown(f"**{AGENT_NAMES[key]}**")
        if decision is None:
            st.caption("Waiting for a run.")
            return
        if key not in decision:
            st.caption("Not run (no such agent in this crew).")
            return
        block = decision[key]
        if block is None:
            st.caption("working...")
            return
        if block["status"]:
            label = f"{block['status']} (placeholder)" if placeholder else block["status"]  # no agent was called
            st.badge(label, color=BADGE_COLOR[block["status"]])
        if _is_insufficient(block):
            st.caption(INSUFFICIENT_TEXT.get(key, INSUFFICIENT_TEXT["default"]))
            return
        if block["status"] == "FALLBACK":
            st.caption(f"Fallback reason: {block['fallback_reason'] or 'not given'}")
        if block["output"] is None:
            if key == "a2" and decision["applied"]["source"] == "starter":
                st.caption("Starter rule (no A2 decision at round 0).")
                st.dataframe(diff_table(decision), hide_index=True, width="stretch")
            elif key == "a2" and block["status"] == "FALLBACK":  # the rule decided: show what was applied
                st.dataframe(diff_table(decision), hide_index=True, width="stretch")
            elif block["status"] != "FALLBACK":
                st.caption("Not run this round.")
            return
        _card_body(key, block, decision)


def agent_row(decision: dict | None, placeholder: bool = False) -> None:
    for col, key in zip(st.columns(3), ("a1", "a3", "a2")):
        with col:
            agent_card(key, decision, placeholder)
    for col, key in zip(st.columns(2), ("a4", "a5")):
        with col:
            agent_card(key, decision, placeholder)


def _num(v, fmt: str) -> str:
    return "n/a" if v is None or pd.isna(v) else fmt.format(v)


def baselines() -> tuple[pd.DataFrame, bool]:
    """(rows, is_projected): ladder rows 1 (keyword) and 2 (starter blend) with a false-teen value.

    Same row rule as the Results tab: a real eval.csv gives held-out test rows only. Empty if the ladder fails.
    """
    try:
        ladder, ladder_placeholder = load_ladder(RESULTS)
    except ValueError:
        return pd.DataFrame(), False
    rows = headline_rows(ladder, ladder_placeholder)
    # placeholder names start "1 " / "2 "; real eval.csv names are keyword_baseline / starter_blend_...
    base = rows[rows["stage"].astype(str).str.match(r"^(?:[12] |keyword_baseline|starter_blend)")].dropna(subset=["ft"])
    return base.assign(label=base["stage"] + (" (projected)" if ladder_placeholder else "")), ladder_placeholder


def loop_chart(rounds: pd.DataFrame, cap: float, tag: str) -> alt.LayerChart:
    last = max(LAST_ROUND, int(rounds["round"].max()) if len(rounds) else 0)
    caps = rounds.set_index("round")["cap"] if len(rounds) else pd.Series(dtype=float)
    band, current = [], DEFAULT_CAP if caps.empty else float(caps.iloc[0])
    for r in range(last + 1):
        current = float(caps[r]) if r in caps.index else current
        band.append({"round": r, "cap": current, "lo": max(current - FT_CI, 0.0), "hi": current + FT_CI})
    x = alt.X("round:Q", title="round", scale=alt.Scale(domain=[0, last]), axis=alt.Axis(tickMinStep=1))
    layers = [alt.Chart(pd.DataFrame(band)).mark_area(opacity=0.15, color=FT_HUE).encode(
        x=x, y=alt.Y("lo:Q", title="rate", scale=alt.Scale(domain=[0, 1])), y2="hi:Q",
        tooltip=[alt.Tooltip("cap:Q", format=".0%"), alt.Tooltip("lo:Q", format=".1%"), alt.Tooltip("hi:Q", format=".1%")])]
    base, _ = baselines()
    if len(base):
        layers.append(alt.Chart(base).mark_rule(strokeDash=[4, 4], color="gray").encode(
            y="ft:Q", tooltip=["label", alt.Tooltip("ft:Q", format=".1%")]))
    if len(rounds):
        long = rounds.melt(id_vars=["round"], value_vars=["rec", "ft"], var_name="metric", value_name="rate")
        layers.append(alt.Chart(long).mark_line(point=True).encode(
            x=x, y="rate:Q",
            color=alt.Color("metric:N", scale=alt.Scale(domain=["ft", "rec"], range=[FT_HUE, REC_HUE]),
                            legend=alt.Legend(title="metric (ft = false-teen)")),
            tooltip=["round", "metric", alt.Tooltip("rate:Q", format=".1%")]))
    return alt.layer(*layers, title=f"Recall and false-teen by round{tag}")


def _start_run() -> None:
    """Button callback: start crew.py in the background and make the run picker follow the new run."""
    run_id = crew.start_background(RESULTS)
    if run_id is None:
        st.toast("A loop run is already going; it shows here as it lands.")
        return
    st.session_state["loop_run"] = run_id
    st.session_state["loop_started_run"] = run_id  # this session's run: only it sees that run's error


def render_header(data: LoopData, run: str | None, rounds: pd.DataFrame, decisions: list, running: bool) -> None:
    latest = rounds.iloc[-1] if len(rounds) else None
    seen = [int(r) for r in rounds["round"]] + [d["round"] for d in decisions]
    rnd = max(seen) if seen else None
    mode = latest["mode"] if latest is not None else "SHADOW"
    cap = latest["cap"] if latest is not None else DEFAULT_CAP
    badge = run_badge(decisions, data.is_placeholder)
    c1, c2, c3, c4 = st.columns([2, 4, 2, 2])
    with c1:
        st.badge(badge, color=BADGE_COLOR[badge])
        st.badge(mode, color=BADGE_COLOR[mode])
    with c2:
        st.caption(f"Run {run or '-'}. Round {'-' if rnd is None else rnd} of {LAST_ROUND}, cap {cap:.0%}."
                   + (" Running..." if running else ""))
    with c3:
        st.button("Run loop", disabled=running, on_click=_start_run,
                  help="Runs loop.py R0-R7 with the agents built so far (A1) in the background. Each round is "
                       "appended as it ends and this tab follows it. Agents are live only with ANTHROPIC_API_KEY set.")
    with c4:
        st.toggle("Replay", disabled=True, help="Stub: replay is not wired yet.")


def render_loop_tab() -> None:
    """The tab as a fragment that polls every REFRESH_S seconds only while a run is going (no idle polling)."""
    running = crew.status()["running"]
    st.session_state["loop_polling"] = running
    st.fragment(run_every=REFRESH_S if running else None)(_loop_fragment)()


def _loop_fragment() -> None:
    """The whole tab; while polling it re-runs on its own, without re-running the Results tab."""
    status = crew.status()
    if status["running"] != st.session_state.get("loop_polling", False):
        st.rerun()  # a run started or ended: rebuild the fragment with (or without) polling
    if status["error"] and status["run"] == st.session_state.get("loop_started_run"):
        st.error(f"The loop run you started failed: {status['error']}")
    try:
        data = load_loop()
    except ValueError as e:
        st.error(f"Cannot show loop: {e}")
        st.caption(FOOTER)
        return
    if data.is_placeholder:
        st.warning(BANNER)
    elif data.recorded:
        st.caption(f"Runs marked (recorded) are the committed run ({ROUNDS_RECORDED}, {DECISIONS_RECORDED}). "
                   f"Run loop adds new runs to {ROUNDS_FILE} / {DECISIONS_FILE}; the recorded run stays in the picker.")
    for w in data.warnings:
        st.warning(w)
    if data.n_skipped:
        st.caption(f"{data.n_skipped} lines skipped in decisions (bad JSON or invalid record).")

    all_runs = runs(data)
    live_run = status["run"] if status["running"] else None
    if live_run is not None and live_run not in all_runs:  # started, nothing written yet
        all_runs = [live_run, *all_runs]
    if st.session_state.get("loop_run") not in all_runs:  # e.g. the placeholder run once real files exist
        st.session_state.pop("loop_run", None)
    run = st.selectbox("Run (newest first)", all_runs, key="loop_run",
                       format_func=lambda r: f"{r} (running)" if r == live_run
                       else f"{r} (recorded)" if r in data.recorded else r) if all_runs else None
    rounds, decisions = for_run(data, run)
    tag = " (projected)" if data.is_placeholder else ""
    render_header(data, run, rounds, decisions, status["running"])
    if run is None:
        st.info("No loop run yet. Press Run loop (or run python -m softsignal.crew).")

    latest = rounds.iloc[-1] if len(rounds) else None
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Round", "-" if latest is None else str(int(latest["round"])))
    t2.metric(f"Labels learned{tag}", "-" if latest is None else f"{int(latest['n_labels']):,}")
    t3.metric(f"Cutoff (t_verify){tag}", "-" if latest is None else _num(latest["t_verify"], "{:.2f}"))
    t4.metric("Fallbacks this run", str(statuses(decisions).count("FALLBACK")))
    t5.metric(f"PSI{tag}", "n/a" if latest is None else _num(latest["psi"], "{:.2f}"))

    cap = DEFAULT_CAP if latest is None else float(latest["cap"])
    st.altair_chart(loop_chart(rounds, cap, tag), width="stretch")
    base, base_projected = baselines()
    dashed = ("Dashed lines: false-teen of the keyword baseline and the starter blend"
              + (" (projected, from eval_placeholder.csv). " if base_projected else ". ")) if len(base) else ""
    st.caption(f"Shaded band: cap +/- {FT_CI * 100:.1f} pts (the plan's 95% CI at 15% false-teen on 450 adults). "
               f"{dashed}Rates on the frozen test set.")

    st.subheader("Agents")
    by_round = sorted(decisions, key=lambda d: d["round"], reverse=True)
    if data.is_placeholder:
        st.caption("Hand-typed example outputs. No agent was called.")
    if not by_round:
        agent_row(None)
    else:
        st.caption(f"Round {by_round[0]['round']}")
        agent_row(by_round[0], data.is_placeholder)
        for d in by_round[1:]:
            with st.expander(f"Round {d['round']}"):
                agent_row(d, data.is_placeholder)

    st.subheader("Decision log")
    rows = rounds.set_index("round") if len(rounds) else None
    for d in by_round:
        row = rows.loc[d["round"]] if rows is not None and d["round"] in rows.index else None
        st.text(log_line(d, row, tag))
    if not by_round:
        st.caption("No decisions yet.")

    st.divider()
    st.caption(FOOTER)
