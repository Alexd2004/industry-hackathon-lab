"""Drift check (option 3, step 2): what a covariate shift does to the rule loop, and whether A1 flags it.

For each seed it runs the rule loop (the rule decides every round) twice on the same batches: clean, and with a
Drift from --start-round that moves the strongest teen-leaning feature columns up by --shift train standard
deviations, so the population looks teenier. The drifted run is scored on drifted test rows (loop.make_env), i.e.
the world the live rule has to serve now. A1 runs offline (its fixed PSI threshold) on both, so the table shows
whether drift was flagged and when. One row per (seed, run, round). No agent is called; A2 is not applied.

With --window W a third run, "window", repeats the drifted run with a scripted stand-in for A2 (step 4): it takes the
rule's action and cap, and asks for refit_window = W from the round A1 first calls drift real. This measures the
recency lever alone; no model is called.

Run: python -m softsignal.drift_check [--seeds 5] [--rounds N] [--shift 1.0] [--start-round 4] [--window W] [--no-write]
"""
import argparse
import tempfile
from pathlib import Path

import pandas as pd

from softsignal.agent_timer import AgentTimer
from softsignal.agents.base import LIVE
from softsignal.crew import run_crew
from softsignal.data import ROOT, load_data
from softsignal.features import FEATURE_COLS, SEED, TARGET
from softsignal.loop import SHADOW, DecisionContext, make_env
from softsignal.oracle import Drift

DRIFT_CSV = ROOT / "results" / "drift_check.csv"
N_COLUMNS = 3  # how many teen-leaning columns move
KEEP = ["seed", "run", "round", "mode", "action", "psi", "a1_drift", "prec", "rec", "ft", "auc"]


def teen_columns(train: pd.DataFrame, n: int = N_COLUMNS) -> tuple[str, ...]:
    """The n feature columns most positively correlated with the teen label on train (train only, never test)."""
    corr = train[FEATURE_COLS + [TARGET]].corr()[TARGET].drop(TARGET)
    return tuple(corr.sort_values(ascending=False, kind="stable").index[:n])


def window_decider(window: int):
    """A scripted A2: the rule's action and cap, plus refit_window = window once A1 has called drift real (and after)."""
    seen = {"real": False}

    def decide(ctx: DecisionContext, blocks: dict) -> dict:
        a1 = (blocks.get("a1") or {}).get("output") or {}
        seen["real"] = seen["real"] or a1.get("drift") == "real"
        out = {"action": ctx.rule["action"], "cap": ctx.rule["cap"]}
        if seen["real"]:
            out["refit_window"] = window
        return {"a2": {"status": LIVE, "output": out}}

    return decide


def one_run(train, test, seed: int, drift: Drift | None, n_rounds: int | None, scratch: Path,
            window: int | None = None) -> pd.DataFrame:
    env = make_env(train, test, seed=seed, drift=drift, timer=AgentTimer(scratch / "calls.jsonl"))
    if window is None:
        rounds, records = run_crew(env, None, n_rounds, apply_a2=False)
    else:
        rounds, records = run_crew(env, None, n_rounds, apply_a2=True, decider=window_decider(window))
    flag = [((r.get("a1") or {}).get("output") or {}).get("drift") for r in records]
    return rounds.assign(a1_drift=flag)


def drift_check(seeds, shift: float, start_round: int, n_rounds: int | None = None,
                window: int | None = None) -> pd.DataFrame:
    train, test = load_data(on_param_mismatch="error")
    drift = Drift(start_round, shift, teen_columns(train))
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            runs = [("clean", None, None), ("drift", drift, None)]
            if window is not None:
                runs.append(("window", drift, window))
            for name, d, w in runs:
                r = one_run(train, test, seed, d, n_rounds, Path(tmp), w)
                out.append(r.assign(seed=seed, run=name)[KEEP])
    return pd.concat(out, ignore_index=True)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per run, the share promoted, the final round's test metrics and the rounds A1 called drift real, averaged over seeds."""
    last = table[table["round"] == table["round"].max()]
    flagged = table.assign(real=table["a1_drift"] == "real").groupby(["seed", "run"])["real"].sum()
    out = last.set_index(["seed", "run"]).join(flagged.rename("rounds_flagged"))
    out["promoted"] = out["mode"] != SHADOW  # share of seeds whose candidate went live by the last round
    return out.groupby("run")[["promoted", "rec", "prec", "ft", "auc", "rounds_flagged"]].mean().round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--shift", type=float, default=1.0, help="train standard deviations")
    ap.add_argument("--start-round", type=int, default=4)
    ap.add_argument("--window", type=int, default=None, help="add a run whose scripted A2 asks for this refit_window")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    table = drift_check(range(SEED, SEED + args.seeds), args.shift, args.start_round, args.rounds, args.window)
    pd.set_option("display.width", 140)
    print(summary(table).to_string())
    if not args.no_write:
        DRIFT_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(DRIFT_CSV, index=False)
        print(f"wrote {len(table)} rows to {DRIFT_CSV}")


if __name__ == "__main__":
    main()
