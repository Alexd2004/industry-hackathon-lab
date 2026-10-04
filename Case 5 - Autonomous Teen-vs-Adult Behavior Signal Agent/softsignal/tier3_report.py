"""Tier 3 report reader (step 18, sub-step 1): load one rule run and one crew run and check they can be compared.

Reads results/rounds.csv style files (one row per (run, round)) and the per-call log. No numbers are made here;
later sub-steps build the quality rows and the latency table from what this module hands back.

Comparable means: both runs have the same rounds, the same R0 row (the starter rule on the frozen test set,
before any agent or label, so a different R0 means a different seed, split or code), and the rule run has no
A2 decision in it. The seed and the batch order are constants in code (features.SEED, oracle), so R0 is the
check the files themselves can show.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.loop import ROUNDS_CSV
from softsignal.metrics import ROUNDS_COLS

METRIC_COLS = ["prec", "rec", "ft", "mt", "auc"]
A2_SOURCE = "A2"


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
