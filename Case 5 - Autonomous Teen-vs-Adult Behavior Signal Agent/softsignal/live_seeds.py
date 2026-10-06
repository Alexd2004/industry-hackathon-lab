"""Live crew vs rule runs over several seeds (Step 19, item 2): does A2's own decision move recall / false-teen?

For each seed it runs the loop twice on the same batches: "rule" (the rule decides every round, A2 logged only) and
"crew" (A2's own action, cap and refit_window are applied). The crew run calls the live agents, so it needs a credential
(ANTHROPIC_API_KEY in the environment or the case folder's gitignored .env) and costs calls; offline it falls back and
the two runs would be the same, so this refuses to run without a client. Nothing is written to the recorded run or
to rounds.csv / decisions.jsonl. One row per (seed, run, round); a2_* columns say what A2 asked for.

Run: python -m softsignal.live_seeds [--seeds 5] [--rounds N] [--drift] [--shift 1.0] [--start-round 4] [--no-write]
"""
import argparse
import tempfile
from pathlib import Path

import pandas as pd

from softsignal.agent_timer import AgentTimer
from softsignal.agents.base import make_client
from softsignal.crew import run_crew
from softsignal.data import ROOT, load_data
from softsignal.drift_check import teen_columns
from softsignal.features import SEED
from softsignal.loop import SHADOW, make_env
from softsignal.oracle import Drift

LIVE_CSV = ROOT / "results" / "live_seeds.csv"
KEEP = ["seed", "run", "round", "mode", "action", "prec", "rec", "ft", "auc",
        "a2_status", "a2_action", "a2_cap", "a2_window"]


def a2_columns(records: list[dict]) -> pd.DataFrame:
    """What A2 said each round (status, and its action, cap and refit_window when it gave an output)."""
    rows = []
    for r in records:
        a2 = r.get("a2") or {}
        out = a2.get("output") or {}
        rows.append({"a2_status": a2.get("status"), "a2_action": out.get("action"), "a2_cap": out.get("cap"),
                     "a2_window": out.get("refit_window")})
    return pd.DataFrame(rows)


def one_run(train, test, seed: int, mode: str, client, drift: Drift | None, n_rounds: int | None,
            scratch: Path) -> pd.DataFrame:
    env = make_env(train, test, seed=seed, drift=drift, timer=AgentTimer(scratch / "calls.jsonl"))
    rounds, records = run_crew(env, client, n_rounds, apply_a2=mode == "crew")
    return pd.concat([rounds, a2_columns(records)], axis=1).assign(seed=seed, run=mode)[KEEP]


def live_seeds(seeds, client, drift: Drift | None = None, n_rounds: int | None = None) -> pd.DataFrame:
    train, test = load_data(on_param_mismatch="error")
    if drift is not None:
        drift = Drift(drift.start_round, drift.shift, teen_columns(train))
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            for mode in ("rule", "crew"):
                out.append(one_run(train, test, seed, mode, client, drift, n_rounds, Path(tmp)))
                print(f"seed {seed} {mode}: done", flush=True)
    return pd.concat(out, ignore_index=True)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per run, the share promoted and the last round's test metrics, averaged over seeds."""
    last = table[table["round"] == table["round"].max()].set_index(["seed", "run"])
    last = last.assign(promoted=last["mode"] != SHADOW)
    return last.groupby("run")[["promoted", "rec", "prec", "ft", "auc"]].mean().round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--drift", action="store_true", help="inject the drift_check drift into both runs")
    ap.add_argument("--shift", type=float, default=1.0, help="train standard deviations")
    ap.add_argument("--start-round", type=int, default=4)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    client = make_client()
    if client is None:
        raise SystemExit("no live client (no credential, or SOFTSIGNAL_OFFLINE is set): nothing to compare")
    drift = Drift(args.start_round, args.shift, ()) if args.drift else None
    table = live_seeds(range(SEED, SEED + args.seeds), client, drift, args.rounds)
    pd.set_option("display.width", 140)
    print(summary(table).to_string())
    if not args.no_write:
        LIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(LIVE_CSV, index=False)
        print(f"wrote {len(table)} rows to {LIVE_CSV}")


if __name__ == "__main__":
    main()
