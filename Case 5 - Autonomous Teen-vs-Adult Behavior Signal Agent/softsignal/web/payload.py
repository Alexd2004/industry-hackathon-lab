"""JSON for the web console, read from the results files. Nothing here writes a file or calls a model.

Runs. Three sources, each a (rounds csv, decisions jsonl) pair: the committed crew recording
(rounds_recorded.csv + decisions_recorded.jsonl), the committed rule-only run (rounds_rule.csv, no agent
records) and the live files a "Run loop" appends to (rounds.csv + decisions.jsonl, gitignored). A torn last
line (a writer mid-append) is skipped; for decisions the last line of a (run, round) wins, as in ui_loop.

Test metrics in rounds.csv are the live rule's on the frozen 900: while the loop is in SHADOW the starter blend
is live, so recall and false-teen stay flat until a promote. The page shows that as it is.

Lists. Each run has its own likely-teen list, scored by its final live rule (run_list.py); the page opens on no run
and shows a run's list once the run has finished.

Ladder. Rows come from eval_tier1.csv, eval_tier3.csv and the static stack at each slider cap (ranked.csv scores
at policy_grid.csv's t_verify, scored against the test labels here, report only). The keyword baseline and the
tabular LR have no results file, so they are computed once per process (baselines.py, a few seconds).
"""
import io
import json
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.agents.base import outcome
from softsignal.data import ROOT
from softsignal.explain import ACTIONS, FEATURE_NAMES
from softsignal.features import ID_COL, TARGET
from softsignal.metrics import EVAL_COLS, ROUNDS_COLS, auc, eval_row
from softsignal.stack import TEXT_FEATURE
from softsignal.tier1 import STARTER_W

RESULTS = ROOT / "results"
DATA = ROOT / "data"
HOURLY = DATA / "blogger_hourly_sessions_sample.csv"
AGENT_KEYS = ("a1", "a2", "a3", "a4", "a5")
N_ROUNDS = 8  # R0 + 7 batches of 300 (oracle.BATCH_SIZE over 2,100 train accounts)
STATUS_ORDER = ("REPLAY", "LIVE", "FALLBACK")  # ui_loop.run_badge: REPLAY wins, never LIVE without a live call

SOURCES = {
    "recorded": ("rounds_recorded.csv", "decisions_recorded.jsonl", "Replay: recorded crew run"),
    "rule": ("rounds_rule.csv", None, "Replay: rule-only run"),
    "live": ("rounds.csv", "decisions.jsonl", "Live run"),
}

# The starter's rules (tier1.style_score / activity_score): weight, display name, group. Its score is
# (1 - w) * style + w * activity, so each rule's share of the top score is its weight times its side's weight.
STARTER_STYLE = (("avg_word_len", 0.35), ("first_person_rate", 0.25), ("exclaim_rate", 0.20),
                 ("slang_emoji_rate", 0.15), ("school_token_rate", 0.20))
STARTER_ACTIVITY = (("pct_active_school_hours", 0.25), ("pct_active_evening", 0.25),
                    ("share_short_video_views", 0.20), ("night_notification_open_rate", 0.15),
                    ("weekend_weekday_session_ratio", 0.15))
WRITING = {TEXT_FEATURE, "avg_word_len", "first_person_rate", "school_token_rate", "birthday_token_rate",
           "exclaim_rate", "slang_emoji_rate", "keyword_teen_flag"}
N_WEIGHTS = 7


def clean(v):
    """A JSON-safe value: NaN and inf to None, numpy scalars to Python."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def _complete_text(path: Path) -> str:
    """The file's text up to its last newline (a line still being appended is left out)."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return text if text.endswith("\n") else text[: text.rfind("\n") + 1]


def read_rounds(path: Path) -> pd.DataFrame:
    text = _complete_text(path)
    if not text.strip():
        return pd.DataFrame(columns=ROUNDS_COLS)
    df = pd.read_csv(io.StringIO(text), dtype={"run": str})
    return df.drop_duplicates(["run", "round"], keep="last").sort_values(["run", "round"])


def read_decisions(path: Path | None) -> dict:
    """{(run, round): record}, the last line of each (run, round) winning."""
    out: dict = {}
    if path is None:
        return out
    for line in _complete_text(path).splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "run" in rec and "round" in rec:
            out[(str(rec["run"]), int(rec["round"]))] = rec
    return out


def run_badge(decisions: list) -> str:
    seen = {(d.get(k) or {}).get("status") for d in decisions if d for k in AGENT_KEYS}
    return next((s for s in STATUS_ORDER if s in seen), "RULE ONLY")


def _agent_view(block: dict | None) -> dict | None:
    """The fields the page shows for one agent block (no hashes, no raw prompts)."""
    if not block:
        return None
    view = {k: block.get(k) for k in ("status", "output", "fallback_reason", "ms", "tokens_in", "tokens_out",
                                      "errors", "rejected")}
    view["outcome"] = outcome(block)  # LIVE, REPLAY, NOT_RUN or a fallback's kind (SCRIPTED ... FAILED)
    return view


def read_shadow(path: Path) -> dict:
    """{(run, round): shadow row} from a shadow csv (loop.write_run: the model in training on the test set)."""
    df = read_rounds(path) if path.exists() else None  # same reader: torn tail skipped, last (run, round) wins
    if df is None or df.empty:
        return {}
    return {(str(r["run"]), int(r["round"])): r for r in records(df)}


def _round_view(row: dict, rec: dict | None, shadow: dict | None = None) -> dict:
    out = dict(row)
    out["shadow"] = None if shadow is None else {k: shadow.get(k) for k in
                                                 ("candidate_round", "t_verify", "prec", "rec", "ft", "mt", "auc")}
    rec = rec or {}
    out["agents"] = {k: _agent_view(rec.get(k)) for k in AGENT_KEYS}
    out["rule_decision"] = rec.get("rule_decision")
    out["applied"] = rec.get("applied")
    out["diff"] = rec.get("diff") or {}
    out["evidence"] = rec.get("evidence")
    return out


def load_runs(results_dir: Path = RESULTS) -> list[dict]:
    """Every run in the three sources, newest live runs first, then the committed ones."""
    runs = []
    for source, (r_name, d_name, label) in SOURCES.items():
        rounds = read_rounds(results_dir / r_name)
        decs = read_decisions(results_dir / d_name if d_name else None)
        shadows = read_shadow(results_dir / r_name.replace("rounds", "shadow", 1))
        ids = sorted(set(rounds["run"]) | {k[0] for k in decs}, reverse=True)
        for run in ids:
            rows = records(rounds[rounds["run"] == run])
            by_round = {r: d for (rid, r), d in decs.items() if rid == run}
            views = [_round_view(row, by_round.get(int(row["round"])), shadows.get((run, int(row["round"]))))
                     for row in rows]
            crew = any(v.get("applied_source") == "A2" for v in views) or any(by_round.values())
            runs.append({
                "id": f"{source}:{run}", "run": run, "source": source,
                "label": label if source != "live" else f"{label} {run[:15]}",
                "kind": "crew" if crew else "rule",
                "badge": run_badge(list(by_round.values())),
                "rounds": views, "complete": len(views) >= N_ROUNDS,
            })
    order = {"live": 0, "recorded": 1, "rule": 2}
    return sorted(runs, key=lambda r: (order[r["source"]], [-ord(c) for c in r["run"]]))


# ---- signal weights ----
def starter_weights() -> list[dict]:
    rows = [(f, w * (1 - STARTER_W)) for f, w in STARTER_STYLE] + [(f, w * STARTER_W) for f, w in STARTER_ACTIVITY]
    total = sum(w for _, w in rows)
    return _weight_rows({f: w / total for f, w in rows})


def stack_weights(contrib_path: Path = RESULTS / "contrib.csv") -> list[dict]:
    """Each level-2 feature's share of the mean |contribution| over the held-out accounts (explain.py's output)."""
    if not contrib_path.exists():
        return []
    c = pd.read_csv(contrib_path, usecols=["feature", "contrib"])
    mean_abs = c.assign(a=c["contrib"].abs()).groupby("feature")["a"].mean()
    return _weight_rows((mean_abs / mean_abs.sum()).to_dict())


def _weight_rows(shares: dict) -> list[dict]:
    top = sorted(shares.items(), key=lambda kv: -kv[1])[:N_WEIGHTS]
    return [{"feature": f, "label": FEATURE_NAMES.get(f, f), "share": float(s),
             "group": "writing" if f in WRITING else "activity"} for f, s in top]


# ---- ranked list and cap grid ----
def ranked_payload(results_dir: Path = RESULTS) -> dict:
    grid_path = results_dir / "policy_grid.csv"
    if not grid_path.exists():
        return {"grid": [], "actions": ACTIONS}
    grid = pd.read_csv(grid_path)
    gcols = ["cap", "t_verify", "t_soft", "t_budget", "n_flagged", "n_verify", "n_soft", "n_none",
             "rec_flagged", "ft_flagged", "prec_sent", "rec_sent", "ft_sent"]
    return {"grid": records(grid[gcols]), "actions": ACTIONS}  # the slider's readout; lists come per run


def run_list(run_id: str, results_dir: Path = RESULTS) -> dict:
    """A run's own list (web/run_list.py): committed runs have one file per source, live runs one per run id."""
    source, _, run = run_id.partition(":")
    names = {"recorded": "ranked_recorded.csv", "rule": "ranked_rule.csv"}
    path = results_dir / "run_lists" / f"{run}.csv" if source == "live" else results_dir / names.get(source, "-")
    if not run or not path.exists():
        return {"run": run_id, "rows": None}
    df = pd.read_csv(path, dtype={ID_COL: str})
    keep = ["rank", ID_COL, "score", "band", "c1", "c2", "c3", "words"]
    model = "stack" if (df["f1"].fillna("") == TEXT_FEATURE).any() else "starter"
    return {"run": run_id, "model": model, "file": str(path.relative_to(results_dir)), "rows": records(df[keep])}


# ---- login heatmap ----
def heatmap(path: Path = HOURLY) -> dict:
    """Mean sessions per (day of week, hour) for teens and for adults (the 33-user hourly sample, synthetic)."""
    if not path.exists():
        return {}
    h = pd.read_csv(path, usecols=["blogger_id", "is_teen", "hour", "dow", "sessions"])
    out = {"n_users": {}}
    for teen, name in ((True, "teen"), (False, "adult")):
        part = h[h["is_teen"] == teen]
        grid = part.groupby(["dow", "hour"])["sessions"].mean().unstack("hour").reindex(index=range(7),
                                                                                       columns=range(24))
        out[name] = [[clean(v) for v in row] for row in grid.to_numpy()]
        out["n_users"][name] = int(part["blogger_id"].nunique())
    return out


# ---- held-out labels (report only) and the ladder ----
_cache: dict = {}
_cache_lock = threading.Lock()


def _test_frames():
    with _cache_lock:
        if "split" not in _cache:
            from softsignal.data import load_data

            _cache["split"] = load_data()
        return _cache["split"]


def test_counts() -> dict:
    _, test = _test_frames()
    y = test[TARGET]
    return {"n": int(len(y)), "teens": int(y.sum()), "adults": int((y == 0).sum())}


def _computed_rows() -> list[dict]:
    """Keyword baseline and tabular LR on the test set, computed once per process (no results file holds them)."""
    with _cache_lock:
        cached = _cache.get("baselines")
    if cached is not None:
        return cached
    from softsignal.baselines import keyword_baseline, tabular_lr

    train, test = _test_frames()
    rows = [keyword_baseline(test)] + [r for r in tabular_lr(train, test).rows if r["eval_set"] == "test"]
    with _cache_lock:
        _cache["baselines"] = rows
    return rows


def stack_rows(results_dir: Path = RESULTS, caps=(0.15, 0.10)) -> list[dict]:
    """The static stack at each cap: ranked.csv scores at that cap's t_verify, against the test labels."""
    ranked_path, grid_path = results_dir / "ranked.csv", results_dir / "policy_grid.csv"
    if not ranked_path.exists() or not grid_path.exists():
        return []
    _, test = _test_frames()
    ranked = pd.read_csv(ranked_path, dtype={ID_COL: str})
    y = test.set_index(ID_COL)[TARGET].reindex(ranked[ID_COL]).to_numpy()
    if np.isnan(y.astype(float)).any():
        return []  # ranked.csv is from another split
    grid = pd.read_csv(grid_path)
    out = []
    for cap in caps:
        hit = grid[(grid["cap"] - cap).abs() < 1e-9]
        if hit.empty:
            continue
        t = float(hit.iloc[0]["t_verify"])
        s = ranked["score"].to_numpy()
        out.append(eval_row(f"softsignal_stack_cap{round(cap * 100)}", "test", y.astype(int), (s >= t).astype(int), s))
    return out


LADDER_LABELS = {
    "keyword_baseline": "Keyword baseline",
    "starter_blend": "Starter blend (w=0.45, cut=0.50)",
    "alt_blend": "Starter blend, changed weight (case step 5)",
    "tune_cap_best": "tune() under the 15% cap",
    "tabular_lr": "Tabular LR, 16 columns",
    "softsignal_stack_cap15": "SoftSignal stack @ 15% cap",
    "softsignal_stack_cap10": "SoftSignal stack @ 10% cap",
    "softsignal_stack_cap5": "SoftSignal stack @ 5% cap",
    "loop_rule": "Loop R7, rule-based",
    "loop_crew_live": "Loop R7, five-agent crew (live)",
    "loop_crew_replay": "Loop R7, crew (replay)",
}


def _label(stage: str) -> str:
    return next((v for k, v in LADDER_LABELS.items() if stage.startswith(k)), stage)


def ladder(results_dir: Path = RESULTS) -> list[dict]:
    rows: list[dict] = []
    try:
        rows += [{**r, "source": "baselines.py"} for r in _computed_rows()]
    except Exception as e:  # noqa: BLE001 - the ladder still shows the file rows
        rows.append({"stage": "keyword_baseline", "eval_set": "error", "source": f"{type(e).__name__}: {e}"[:200]})
    for name in ("eval_tier1.csv",):
        p = results_dir / name
        if p.exists():
            rows += [{**r, "source": name} for r in records(pd.read_csv(p)[EVAL_COLS])]
    rows += [{**r, "source": "ranked.csv + policy_grid.csv"} for r in stack_rows(results_dir)]
    p = results_dir / "eval_tier3.csv"
    if p.exists():
        rows += [{**r, "source": "eval_tier3.csv"} for r in records(pd.read_csv(p)[EVAL_COLS])]
    order = list(LADDER_LABELS)
    rows.sort(key=lambda r: next((i for i, k in enumerate(order) if r["stage"].startswith(k)), len(order)))
    return [{**{k: clean(v) for k, v in r.items()}, "label": _label(r["stage"])} for r in rows]


def latency(results_dir: Path = RESULTS) -> list[dict]:
    p = results_dir / "tier3_latency.csv"
    return records(pd.read_csv(p)) if p.exists() else []


def static_payload(results_dir: Path = RESULTS) -> dict:
    """Everything that does not change during a loop run."""
    from softsignal.policy import load_policy

    return {
        "policy": load_policy(),
        "ranked": ranked_payload(results_dir),
        "heatmap": heatmap(),
        "weights": {"starter": starter_weights(), "stack": stack_weights(results_dir / "contrib.csv")},
        "test": test_counts(),
        "latency": latency(results_dir),
    }
