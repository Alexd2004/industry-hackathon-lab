"""Tier 3 report reader (step 18, sub-step 1): load one rule run and one crew run and check they can be compared.

Reads results/rounds.csv style files (one row per (run, round)) and the per-call log, then builds the Tier 3
quality rows (sub-step 2) in the frozen eval.csv schema. Every number is copied from a run's final round, the
frozen 900-account test set; only F1 is derived, with metrics.f1. The latency table is sub-step 3.

Files: results/rounds_recorded.csv (the committed live crew run), results/rounds_rule.csv (a rule-only run on
the same seed, made by `python -m softsignal.tier3_report --make-rule`), results/eval_tier3.csv (the rows;
a later eval.py can merge them into eval.csv).

Comparable means: both runs have the same rounds, the same R0 row (the starter rule on the frozen test set,
before any agent or label, so a different R0 means a different seed, split or code), and the rule run has no
A2 decision in it. The seed and the batch order are constants in code (features.SEED, oracle), so R0 is the
check the files themselves can show.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.data import ROOT
from softsignal.loop import ROUNDS_CSV
from softsignal.metrics import EVAL_COLS, ROUNDS_COLS, f1

METRIC_COLS = ["prec", "rec", "ft", "mt", "auc"]
A2_SOURCE = "A2"
RESULTS = ROOT / "results"
ROUNDS_RECORDED = RESULTS / "rounds_recorded.csv"
ROUNDS_RULE = RESULTS / "rounds_rule.csv"
EVAL_TIER3 = RESULTS / "eval_tier3.csv"


def load_rounds(path: Path | str = ROUNDS_CSV) -> pd.DataFrame:
    """The rounds file with its frozen columns; a different header is an error, not a guess."""
    df = pd.read_csv(path, index_col=False, dtype={"run": str})
    if list(df.columns) != ROUNDS_COLS:
        raise ValueError(f"{Path(path).name}: columns {list(df.columns)} are not the frozen ROUNDS_COLS")
    return df


def run_ids(rounds: pd.DataFrame) -> list[str]:
    """Run ids in file order, each once."""
    return list(dict.fromkeys(rounds["run"]))


def get_run(rounds: pd.DataFrame, run: str) -> pd.DataFrame:
    """One run's rows sorted by round. It must exist and hold rounds 0..n with no gap or repeat."""
    rows = rounds[rounds["run"] == run].sort_values("round").reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"run {run!r} is not in the rounds file (runs: {run_ids(rounds)})")
    if list(rows["round"]) != list(range(len(rows))):
        raise ValueError(f"run {run!r} is incomplete: rounds {list(rows['round'])}, expected 0..{len(rows) - 1}")
    return rows


def final_row(run_rows: pd.DataFrame) -> pd.Series:
    """The last round of a run, the row the ladder quotes."""
    return run_rows.iloc[-1]


def sources(run_rows: pd.DataFrame) -> list[str]:
    """Who applied each round's decision (starter / rule / A2)."""
    return list(run_rows["applied_source"])


def comparison_errors(rule_rows: pd.DataFrame, crew_rows: pd.DataFrame) -> list[str]:
    """Why these two runs cannot be compared, as plain sentences. Empty means they can."""
    errors = []
    if list(rule_rows["round"]) != list(crew_rows["round"]):
        errors.append(f"rounds differ: rule {list(rule_rows['round'])}, crew {list(crew_rows['round'])}")
    if A2_SOURCE in sources(rule_rows):
        errors.append("the rule run has an A2 decision in it, so it is not a rule-only run")
    if not errors:
        r0_rule, r0_crew = rule_rows.iloc[0], crew_rows.iloc[0]
        a, b = r0_rule[METRIC_COLS].astype(float).to_numpy(), r0_crew[METRIC_COLS].astype(float).to_numpy()
        if not np.allclose(a, b, rtol=0, atol=1e-9):
            errors.append(f"R0 differs ({dict(zip(METRIC_COLS, a))} against {dict(zip(METRIC_COLS, b))}), "
                          "so the seed, the split or the starter rule is not the same")
    return errors


def check_comparable(rule_rows: pd.DataFrame, crew_rows: pd.DataFrame) -> None:
    """Raise ValueError listing every reason the two runs cannot be compared."""
    errors = comparison_errors(rule_rows, crew_rows)
    if errors:
        raise ValueError("cannot compare the runs: " + "; ".join(errors))


def calls_for_run(records: list[dict], run: str) -> list[dict]:
    """The per-call log lines of one run (agent_timer.load_records gives the list)."""
    return [r for r in records if r.get("run") == run]


def quality_row(stage: str, run_rows: pd.DataFrame) -> dict:
    """One eval.csv row from a run's final round (frozen test set). f1 is derived from prec and rec."""
    last = final_row(run_rows)
    prec, rec = float(last["prec"]), float(last["rec"])
    return {"stage": stage, "eval_set": "test", "prec": prec, "rec": rec, "ft": float(last["ft"]),
            "mt": float(last["mt"]), "f1": f1(prec, rec), "auc": float(last["auc"])}


def quality_rows(rule_rows: pd.DataFrame, crew_live_rows: pd.DataFrame,
                 crew_replay_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ladder rows 11 and 12 next to the rule baseline, all from the same seed and batch order.

    loop_rule_R<n>: the rule-only loop. loop_crew_live_R<n>: the recorded live crew run (row 11).
    loop_crew_replay_R<n>: the same crew replayed offline (row 12), only when given.
    Each crew run must be comparable with the rule run (check_comparable raises otherwise).
    """
    runs = [("loop_rule", rule_rows), ("loop_crew_live", crew_live_rows)]
    if crew_replay_rows is not None:
        runs.append(("loop_crew_replay", crew_replay_rows))
    for _, rows in runs[1:]:
        check_comparable(rule_rows, rows)
    return pd.DataFrame([quality_row(f"{name}_R{int(final_row(rows)['round'])}", rows) for name, rows in runs],
                        columns=EVAL_COLS)


def write_eval_tier3(rows: pd.DataFrame, path: Path | str = EVAL_TIER3) -> None:
    """Write the rows with the frozen eval.csv header, replacing the file."""
    if list(rows.columns) != EVAL_COLS:
        raise ValueError(f"columns {list(rows.columns)} are not the frozen EVAL_COLS")
    rows.to_csv(path, index=False)


def _fresh_env(timer_log: Path):
    """An env on the committed split with its own timer log, so a report run never touches agent_calls.jsonl."""
    from softsignal.agent_timer import AgentTimer
    from softsignal.data import load_data
    from softsignal.loop import make_env

    train, test = load_data(on_param_mismatch="error")
    return make_env(train, test, timer=AgentTimer(timer_log))


def make_rule_run(path: Path = ROUNDS_RULE) -> pd.DataFrame:
    """Run the rule-only loop (no agents, no network) and write its rounds, replacing the file."""
    import tempfile

    from softsignal.loop import run_loop

    with tempfile.TemporaryDirectory() as tmp:
        rounds, _ = run_loop(_fresh_env(Path(tmp) / "calls.jsonl"))
    rounds.to_csv(path, index=False)
    return rounds


def replay_crew_run() -> pd.DataFrame:
    """Replay the committed crew recording offline (no client, no network) and return its rounds."""
    import tempfile

    from softsignal.crew import DECISIONS_RECORDED, run_crew
    from softsignal.replay import Replayer

    with tempfile.TemporaryDirectory() as tmp:
        rounds, _ = run_crew(_fresh_env(Path(tmp) / "calls.jsonl"), None,
                             replayer=Replayer.from_file(DECISIONS_RECORDED))
    return rounds


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Tier 3 quality rows (rule run, live crew run, replayed crew run)")
    ap.add_argument("--make-rule", action="store_true", help=f"run the rule-only loop and write {ROUNDS_RULE.name}")
    args = ap.parse_args()
    if args.make_rule:
        make_rule_run()
        print(f"wrote {ROUNDS_RULE}")
        return
    rule = load_rounds(ROUNDS_RULE)
    recorded = load_rounds(ROUNDS_RECORDED)
    rule_rows, live_rows = get_run(rule, run_ids(rule)[-1]), get_run(recorded, run_ids(recorded)[-1])
    replay = replay_crew_run()
    replay_rows = get_run(replay, run_ids(replay)[-1])
    if sources(replay_rows) != sources(live_rows):
        print(f"note: replay applied {sources(replay_rows)}, the recording applied {sources(live_rows)}")
    rows = quality_rows(rule_rows, live_rows, replay_rows)
    write_eval_tier3(rows)
    pd.set_option("display.width", 200)
    print(rows.round(3).to_string(index=False))
    print(f"wrote {EVAL_TIER3}")


if __name__ == "__main__":
    main()
