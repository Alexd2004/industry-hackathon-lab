"""A loop run's own likely-teen list: the 900 held-out accounts scored by the run's final live rule.

A run that promoted ends on a stack (explain.ranked: top 3 signed contributions and teen-leaning words). A run that
never promoted ends on the starter blend, which has no contributions: its "why" chips are the starter rules that fired
for the account, each with its weight in the score, and it has no words and no soft band (t_soft == t_verify).
Either way the bands are the run's own thresholds at its cap. Test labels are never read here.

Files: live runs (started from the console) write results/run_lists/<run>.csv (gitignored, like rounds.csv). The two
committed runs have committed lists, made by this module's CLI:

    python -m softsignal.web.run_list   # ranked_recorded.csv (the crew recording) and ranked_rule.csv (the rule run)

The crew recording never promoted, so its list is the starter's (no rerun needed). The rule run's final stack was not
saved, so it is rebuilt by rerunning the rule loop offline (seeded, seconds) and written only if the rerun's rounds
match rounds_rule.csv. Library versions can move a seeded refit by an account or two; then nothing is written and the
console says the run has no list.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.explain import PHRASES, apply_bands, rank_order, ranked
from softsignal.features import ID_COL
from softsignal.loop import ACTIVE, State, starter_score
from softsignal.metrics import RANKED_COLS
from softsignal.tier1 import STARTER_W

RESULTS = Path(__file__).resolve().parents[2] / "results"
LIVE_DIR = RESULTS / "run_lists"
COMMITTED = {"recorded": RESULTS / "ranked_recorded.csv", "rule": RESULTS / "ranked_rule.csv"}

# The starter's rules exactly as tier1.style_score / activity_score: (feature, fires when, weight, side).
# side "style" weighs (1 - STARTER_W), "activity" weighs STARTER_W in the blend.
STARTER_RULES = (
    ("avg_word_len", lambda v: v < 4.4, 0.35, "style"),
    ("first_person_rate", lambda v: v > 0.06, 0.25, "style"),
    ("exclaim_rate", lambda v: v > 0.008, 0.20, "style"),
    ("slang_emoji_rate", lambda v: v > 0.002, 0.15, "style"),
    ("school_token_rate", lambda v: v > 0, 0.20, "style"),
    ("pct_active_school_hours", lambda v: v < 0.25, 0.25, "activity"),
    ("pct_active_evening", lambda v: v > 0.35, 0.25, "activity"),
    ("share_short_video_views", lambda v: v > 0.40, 0.20, "activity"),
    ("night_notification_open_rate", lambda v: v > 0.22, 0.15, "activity"),
    ("weekend_weekday_session_ratio", lambda v: v > 1.2, 0.15, "activity"),
)
BELOW = {"avg_word_len", "pct_active_school_hours"}  # rules that fire on a low value read with the "below" phrase


def rule_weight(weight: float, side: str) -> float:
    return weight * (STARTER_W if side == "activity" else 1 - STARTER_W)


def starter_list(test: pd.DataFrame, t_verify: float) -> pd.DataFrame:
    """The starter blend's list: score, bands at t_verify (no soft band) and the rules that fired as chips."""
    fired = pd.DataFrame({f: test[f].map(fires).astype(float) * rule_weight(w, side)
                          for f, fires, w, side in STARTER_RULES})
    score = starter_score(test)
    frame = pd.DataFrame({ID_COL: test[ID_COL].astype(str).to_numpy(), "score": score, "words": ""})
    names, vals = list(fired.columns), fired.to_numpy()
    for i in range(3):
        cs, fs, vs = [], [], []
        for row in vals:
            top = [j for j in np.argsort(-row, kind="stable")[:3] if row[j] > 0]
            if i < len(top):
                f = names[top[i]]
                cs.append(f"{PHRASES[f][1 if f in BELOW else 0]} +{row[top[i]]:.2f}")
                fs.append(f)
                vs.append(float(row[top[i]]))
            else:
                cs.append("")
                fs.append("")
                vs.append(np.nan)
        frame[f"c{i + 1}"], frame[f"f{i + 1}"], frame[f"v{i + 1}"] = cs, fs, vs
    out = rank_order(apply_bands(frame, t_verify, t_verify))
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[RANKED_COLS]


def final_list(state: State, test: pd.DataFrame) -> pd.DataFrame:
    """The run's list from its final state: the stack once ACTIVE, else the starter."""
    th = state.live.th
    if state.mode == ACTIVE and state.live.model is not None:
        return ranked(state.live.model, test, th.t_soft, th.t_verify)
    return starter_list(test, th.t_verify)


def live_path(run: str) -> Path:
    return LIVE_DIR / f"{run}.csv"


def write_list(df: pd.DataFrame, path: Path) -> None:
    if list(df.columns) != RANKED_COLS:
        raise ValueError(f"a run list must have columns {RANKED_COLS}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, float_format="%.12g", lineterminator="\n")
    tmp.replace(path)


def main() -> None:
    from softsignal.agent_timer import AgentTimer
    from softsignal.data import load_data
    from softsignal.loop import make_env, new_state, run_loop
    from softsignal.web.payload import read_rounds

    train, test = load_data(on_param_mismatch="error")

    crew = read_rounds(RESULTS / "rounds_recorded.csv")
    last = crew.iloc[-1]
    if last["mode"] == ACTIVE:
        raise SystemExit("the crew recording promoted: its final stack was not saved, rerun it with crew.py --record")
    write_list(starter_list(test, float(last["t_verify"])), COMMITTED["recorded"])
    print(f"wrote {COMMITTED['recorded'].name} (starter, t_verify {last['t_verify']})")

    want = read_rounds(RESULTS / "rounds_rule.csv")
    with tempfile.TemporaryDirectory() as tmp:
        env = make_env(train, test, timer=AgentTimer(Path(tmp) / "calls.jsonl"))
        state = new_state(env.policy)
        got, _ = run_loop(env, state=state)
    cols = ["round", "action", "mode", "rec", "ft", "auc"]
    a, b = got[cols].reset_index(drop=True), want[cols].reset_index(drop=True)
    same = a[["round", "action", "mode"]].equals(b[["round", "action", "mode"]]) and np.allclose(
        a[["rec", "ft", "auc"]].to_numpy(float), b[["rec", "ft", "auc"]].to_numpy(float), atol=1e-9)
    if not same:
        # seen with other scikit-learn / pandas versions than the run was recorded with: a stack from this rerun is
        # not that run's model, so the rule run gets no list (a Rule only run from the console makes its own)
        print("rule rerun does not match rounds_rule.csv (library versions?), ranked_rule.csv not written:\n"
              f"rerun R7: rec {a['rec'].iloc[-1]:.4f} ft {a['ft'].iloc[-1]:.4f}; "
              f"committed R7: rec {b['rec'].iloc[-1]:.4f} ft {b['ft'].iloc[-1]:.4f}")
        return
    write_list(final_list(state, test), COMMITTED["rule"])
    print(f"wrote {COMMITTED['rule'].name} ({state.mode}, the rerun matches rounds_rule.csv)")


if __name__ == "__main__":
    main()
