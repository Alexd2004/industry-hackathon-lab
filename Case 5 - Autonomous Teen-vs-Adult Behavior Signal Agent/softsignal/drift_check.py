"""Drift check (option 3, step 2): what a covariate shift does to the rule loop, and whether A1 flags it.

For each seed it runs the rule loop (the rule decides every round) twice on the same batches: clean, and with a
Drift from --start-round that moves the strongest teen-leaning feature columns up by --shift train standard
deviations, so the population looks teenier. The drifted run is scored on drifted test rows (loop.make_env), i.e.
the world the live rule has to serve now. A1 runs offline (its fixed PSI threshold) on both, so the table shows
whether drift was flagged and when. One row per (seed, run, round). No agent is called; A2 is not applied.

Run: python -m softsignal.drift_check [--seeds 5] [--rounds N] [--shift 1.0] [--start-round 4] [--no-write]
"""
import argparse
import tempfile
from pathlib import Path

import pandas as pd

from softsignal.agent_timer import AgentTimer
from softsignal.crew import run_crew
from softsignal.data import ROOT, load_data
from softsignal.features import FEATURE_COLS, SEED, TARGET
from softsignal.loop import make_env
from softsignal.oracle import Drift

DRIFT_CSV = ROOT / "results" / "drift_check.csv"
N_COLUMNS = 3  # how many teen-leaning columns move
KEEP = ["seed", "run", "round", "mode", "action", "psi", "a1_drift", "prec", "rec", "ft", "auc"]


def teen_columns(train: pd.DataFrame, n: int = N_COLUMNS) -> tuple[str, ...]:
    """The n feature columns most positively correlated with the teen label on train (train only, never test)."""
    corr = train[FEATURE_COLS + [TARGET]].corr()[TARGET].drop(TARGET)
    return tuple(corr.sort_values(ascending=False, kind="stable").index[:n])


def one_run(train, test, seed: int, drift: Drift | None, n_rounds: int | None, scratch: Path) -> pd.DataFrame:
    env = make_env(train, test, seed=seed, drift=drift, timer=AgentTimer(scratch / "calls.jsonl"))
    rounds, records = run_crew(env, None, n_rounds, apply_a2=False)
    flag = [((r.get("a1") or {}).get("output") or {}).get("drift") for r in records]
    return rounds.assign(a1_drift=flag)


def drift_check(seeds, shift: float, start_round: int, n_rounds: int | None = None) -> pd.DataFrame:
    train, test = load_data(on_param_mismatch="error")
    drift = Drift(start_round, shift, teen_columns(train))
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            for name, d in (("clean", None), ("drift", drift)):
                r = one_run(train, test, seed, d, n_rounds, Path(tmp))
                out.append(r.assign(seed=seed, run=name)[KEEP])
    return pd.concat(out, ignore_index=True)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per run, the final round's test metrics and the rounds A1 called drift real, averaged over seeds."""
    last = table[table["round"] == table["round"].max()]
    flagged = table.assign(real=table["a1_drift"] == "real").groupby(["seed", "run"])["real"].sum()
    out = last.set_index(["seed", "run"]).join(flagged.rename("rounds_flagged"))
    return out.groupby("run")[["rec", "prec", "ft", "auc", "rounds_flagged"]].mean().round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--shift", type=float, default=1.0, help="train standard deviations")
    ap.add_argument("--start-round", type=int, default=4)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    table = drift_check(range(SEED, SEED + args.seeds), args.shift, args.start_round, args.rounds)
    pd.set_option("display.width", 140)
    print(summary(table).to_string())
    if not args.no_write:
        DRIFT_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(DRIFT_CSV, index=False)
        print(f"wrote {len(table)} rows to {DRIFT_CSV}")


if __name__ == "__main__":
    main()
