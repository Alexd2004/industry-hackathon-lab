"""Crew run (Tier 3): loop.py with the agents built so far, each round appended as it lands.

Today that is loop.py's rule plus A1 (drift watcher) and A2 (controller), through loop.run_loop's callbacks:

    before_decision(state, batch, prior, psi)  -> A1 on (prior = the earlier batches, the batch rows, the score
                                                  PSI while the live rule is the starter, audit counts, earlier
                                                  rounds' PSI); its block goes into the round's record
    decide(context, blocks)                    -> A2 on (the live thresholds, audit counts, bounds, guards, the rule's
                                                  decision and A1's output; A3 is insufficient_data until step 16),
                                                  after the streak update and before the refit; its block and the diff
                                                  against the rule go in the record, and its action and cap are applied
                                                  unless it fell back or apply_a2 is False
    on_round(result)                           -> loop.write_run(this row, this record), when writing

A1 runs after the reveal and the PSI, before the decision, where A2 reads it. Offline (no key) A2 falls
back to the rule's own decision every round, so an offline run's decisions are exactly loop.run_loop's. Score PSI goes to A1
only while the live rule is the starter: once the stack is live every refit changes the model, and
score PSI would measure that change, not drift. A4 is not run here: while the loop is in SHADOW the
starter blend picks the verify band and has no explanations; wire it once the live rule is the stack
(agents/a4_triage.py). Every run has its own run id (its own AgentTimer).

Files: run_crew writes nothing unless asked (write=True). The CLI and the Loop tab's "Run loop" append to
the live files results/rounds.csv and decisions.jsonl, which are gitignored. --record writes one run to
results/rounds_recorded.csv and decisions_recorded.jsonl instead (swapped in only once the full run has
finished, so a failed run leaves the old one; --rounds is refused): the committed run a
fresh clone shows.

Run: python -m softsignal.crew [--mode crew|rule] [--rounds N] [--no-write | --record]   (agents offline unless
ANTHROPIC_API_KEY)
  --mode crew (default): A2's decision is applied. --mode rule: A1 and A2 still run and are logged next to the rule's
  decision, but the rule is applied every round (the counterfactual run; --record needs crew).
"""
import argparse
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from softsignal.agent_timer import DEFAULT_LOG, AgentTimer, round_agent_summary
from softsignal.agents.a1_drift import run_a1
from softsignal.agents.base import make_client, merge_block
from softsignal.agents.a2_controller import run_a2
from softsignal.agents.contracts import INSUFFICIENT_INPUT, a1_history, a1_input, a2_input
from softsignal.data import load_data
from softsignal.loop import (DECISIONS_JSONL, ROUNDS_CSV, SHADOW, DecisionContext, Env, State, make_env, run_loop,
                             write_run)
from softsignal.metrics import ROUNDS_COLS

ROUNDS_RECORDED = ROUNDS_CSV.with_name("rounds_recorded.csv")
DECISIONS_RECORDED = DECISIONS_JSONL.with_name("decisions_recorded.jsonl")


def run_crew(env: Env, client=None, n_rounds: int | None = None, write: bool = False,
             rounds_path: Path = ROUNDS_CSV, decisions_path: Path = DECISIONS_JSONL,
             state: State | None = None, apply_a2: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """R0 then each oracle batch (at most n_rounds) with A1 on every round. Returns (rounds, records) like
    loop.run_loop. write=True appends each round to rounds_path / decisions_path when it ends.
    client: base.make_client() (None = offline, the agents use their fallbacks). A2 runs every round from R1 and
    its decision is logged next to the rule's; apply_a2 (crew mode) applies A2's own decision, False logs it only."""
    psi_drift = env.policy.get("psi_drift")
    history: list[dict] = []

    def before_decision(state: State, batch, prior: pd.DataFrame, psi_val) -> dict:
        rnd = 0 if batch is None else batch.round
        rows = prior.iloc[0:0] if batch is None else batch.rows
        score_psi = psi_val if state.mode == SHADOW else None  # the live model changes every refit once ACTIVE
        payload = a1_input(prior, rows, score_psi, env.oracle.audit_counts(), history, rnd, psi_drift)
        a1 = run_a1(payload, client, env.timer, rnd)
        if batch is not None:
            history.append(a1_history(payload))
        return {"a1": merge_block(a1, round_agent_summary(env.timer.records, rnd, env.timer.run).get(a1.agent))}

    def decide(ctx: DecisionContext, blocks: dict) -> dict:
        a1 = (blocks.get("a1") or {}).get("output")  # whole, as A1 returned it; A3 is not built (step 16)
        payload = a2_input(ctx.round, ctx.thresholds, ctx.audit, ctx.bounds, ctx.guards, ctx.rule,
                           a1=a1 if isinstance(a1, dict) else INSUFFICIENT_INPUT, a3=INSUFFICIENT_INPUT,
                           policy=env.policy)
        a2 = run_a2(payload, client, env.timer, ctx.round)
        return {"a2": merge_block(a2, round_agent_summary(env.timer.records, ctx.round, env.timer.run).get(a2.agent))}

    def on_round(result) -> None:
        if write:
            write_run(pd.DataFrame([result.row], columns=ROUNDS_COLS), [result.record], rounds_path, decisions_path)

    return run_loop(env, n_rounds, state, before_decision=before_decision, on_round=on_round, decide=decide,
                    apply_a2=apply_a2)


# ---- background run for the Loop tab ----
@dataclass
class BackgroundRun:
    """The run this process started last: its thread, run id and error text (None while fine)."""

    thread: threading.Thread | None = None
    run: str | None = None
    error: str | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


_lock = threading.Lock()
_current = BackgroundRun()


def status() -> dict:
    """{"running": bool, "run": the latest run id started here or None, "error": its error text or None}."""
    return {"running": _current.running, "run": _current.run, "error": _current.error}


def _background(run: BackgroundRun, timer: AgentTimer, results_dir: Path, n_rounds: int | None) -> None:
    try:
        train, test = load_data(on_param_mismatch="error")
        env = make_env(train, test, timer=timer)
        run_crew(env, make_client(), n_rounds, True, results_dir / ROUNDS_CSV.name, results_dir / DECISIONS_JSONL.name)
    except Exception as e:  # noqa: BLE001 - shown in the Loop tab instead of dying silently in a thread
        run.error = f"{type(e).__name__}: {e}"[:500]


def start_background(results_dir: Path = ROUNDS_CSV.parent, n_rounds: int | None = None) -> str | None:
    """Start a run in a daemon thread and return its run id, or None while another run is still going.

    The thread appends to results_dir's rounds.csv, decisions.jsonl and agent_calls.jsonl; the app only reads.
    """
    global _current
    with _lock:
        if _current.running:
            return None
        results_dir = Path(results_dir)
        timer = AgentTimer(results_dir / DEFAULT_LOG.name)  # its own run id, its log next to its results
        run = BackgroundRun(run=timer.run)
        run.thread = threading.Thread(target=_background, args=(run, timer, results_dir, n_rounds),
                                      daemon=True, name="softsignal-crew")
        _current = run
        run.thread.start()
        return timer.run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("crew", "rule"), default="crew", help="crew: apply A2; rule: apply the rule")
    ap.add_argument("--rounds", type=int, default=None)
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--no-write", action="store_true", help="print only")
    out.add_argument("--record", action="store_true", help="replace the committed recorded run with this one")
    args = ap.parse_args()
    if args.record and args.rounds is not None:
        ap.error("--record writes the full run that a fresh clone shows; drop --rounds")
    if args.record and args.mode != "crew":
        ap.error("--record writes the crew run that a fresh clone shows; drop --mode rule")
    train, test = load_data(on_param_mismatch="error")
    env = make_env(train, test)
    client = make_client()
    recorded = (ROUNDS_RECORDED, DECISIONS_RECORDED)
    # --record builds the new run next to the committed one and swaps it in only once the run has finished
    paths = tuple(p.with_name(p.name + ".new") for p in recorded) if args.record else (ROUNDS_CSV, DECISIONS_JSONL)
    for p in paths if args.record else ():
        p.unlink(missing_ok=True)
    try:
        rounds, records = run_crew(env, client, args.rounds, not args.no_write, *paths, apply_a2=args.mode == "crew")
    except BaseException:
        for p in paths if args.record else ():
            p.unlink(missing_ok=True)
        raise
    if args.record:
        for new, old in zip(paths, recorded):
            os.replace(new, old)
        paths = recorded
    pd.set_option("display.width", 220)
    print(rounds.drop(columns=["run"]).round(3).to_string(index=False))
    print(f"\nA1 ({'live' if client else 'offline: policy.yaml threshold'}):")
    for r in records:
        a1 = r["a1"]
        o = a1["output"]
        print(f"  R{r['round']}: {a1['status']}" + (f" ({a1['fallback_reason']})" if a1["fallback_reason"] else "")
              + f" -> {o['drift']}: {o['reason']}")
    if not args.no_write:
        print(f"appended run {env.timer.run} to {paths[0].name} and {paths[1].name}")


if __name__ == "__main__":
    main()
