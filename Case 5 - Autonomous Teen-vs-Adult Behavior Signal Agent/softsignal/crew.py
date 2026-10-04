"""Crew run (Tier 3): loop.py with the agents built so far, each round appended as it lands.

Today that is loop.py's rule plus A1 (drift watcher) and A5 (honesty auditor), through loop.run_loop's two
callbacks:

    before_decision(state, batch, prior, psi)  -> A1 on (prior = the earlier batches, the batch rows, the score
                                                  PSI while the live rule is the starter, audit counts, earlier
                                                  rounds' PSI); its block goes into the round's record
    on_round(result)                           -> when writing, the round is appended at once with A5 "working"
                                                  (a5 null), so the screen never waits on A5; then A5 checks the
                                                  finished round (its headline numbers and A2's reason, against
                                                  this run's rows so far), after the decision, so nothing A5
                                                  writes can reach A1-A4; then the record is appended again with
                                                  A5 filled in (decisions.jsonl: the last line of a round wins)

Offline (no client), each agent first asks replay.Replayer for a recorded LIVE output made from the same input
(marked REPLAY); otherwise it uses its fallback. --record never replays: a recording holds live or fallback
output only.

A1 runs after the reveal and the PSI, before the decision, where A2 will read it. With no A2 yet its
verdict informs no decision, so the rule's decisions are exactly loop.run_loop's. Score PSI goes to A1
only while the live rule is the starter: once the stack is live every refit changes the model, and
score PSI would measure that change, not drift. A4 is not run here: while the loop is in SHADOW the
starter blend picks the verify band and has no explanations; wire it once the live rule is the stack
(agents/a4_triage.py). Every run has its own run id (its own AgentTimer).

Files: run_crew writes nothing unless asked (write=True). The CLI and the Loop tab's "Run loop" append to
the live files results/rounds.csv and decisions.jsonl, which are gitignored. --record writes one run to
results/rounds_recorded.csv and decisions_recorded.jsonl instead (swapped in only once the full run has
finished, so a failed run leaves the old one; --rounds is refused): the committed run a
fresh clone shows.

Run: python -m softsignal.crew [--rounds N] [--no-write | --record]   (agents offline unless ANTHROPIC_API_KEY)
"""
import argparse
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from softsignal.agent_timer import DEFAULT_LOG, AgentTimer, round_agent_summary
from softsignal.agents import a1_drift, a5_audit
from softsignal.agents.base import FALLBACK, AgentResult, make_client, merge_block
from softsignal.agents.contracts import (
    a1_history, a1_input, a5_input, a5_sources, evidence_source, load_checklist, round_claims,
)
from softsignal.data import load_data
from softsignal.loop import DECISIONS_JSONL, ROUNDS_CSV, SHADOW, Env, State, make_env, run_loop, write_run
from softsignal.metrics import ROUNDS_COLS
from softsignal.replay import Replayer, read_records, serve

A5_ERROR = "agent_error"  # A5 raised: the round keeps going with an empty, FALLBACK A5 block

ROUNDS_RECORDED = ROUNDS_CSV.with_name("rounds_recorded.csv")
DECISIONS_RECORDED = DECISIONS_JSONL.with_name("decisions_recorded.jsonl")


def run_crew(env: Env, client=None, n_rounds: int | None = None, write: bool = False,
             rounds_path: Path = ROUNDS_CSV, decisions_path: Path = DECISIONS_JSONL,
             state: State | None = None, replayer: Replayer | None = None,
             checklist: list[dict] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """R0 then each oracle batch (at most n_rounds) with A1 and A5 on every round. Returns (rounds, records)
    like loop.run_loop. write=True appends each round to rounds_path / decisions_path when it ends.
    client: base.make_client() (None = offline). replayer: recorded outputs to serve while offline (None = none).
    checklist: A5's risk list (default claims/risks.yaml; [] if that file is missing)."""
    psi_drift = env.policy.get("psi_drift")
    history: list[dict] = []
    rows_so_far: list[dict] = []
    offline = replayer if client is None else None  # live when online, recorded replay when offline
    if checklist is None:
        try:
            checklist = load_checklist()
        except FileNotFoundError:
            checklist = []

    def block(result, rnd: int) -> dict:
        return merge_block(result, round_agent_summary(env.timer.records, rnd, env.timer.run).get(result.agent))

    def before_decision(state: State, batch, prior: pd.DataFrame, psi_val) -> dict:
        rnd = 0 if batch is None else batch.round
        rows = prior.iloc[0:0] if batch is None else batch.rows
        score_psi = psi_val if state.mode == SHADOW else None  # the live model changes every refit once ACTIVE
        payload = a1_input(prior, rows, score_psi, env.oracle.audit_counts(), history, rnd, psi_drift)
        a1 = (serve(offline, "a1", rnd, payload, a1_drift.validate_output, env.timer)
              or a1_drift.run_a1(payload, client, env.timer, rnd))
        if batch is not None:
            history.append(a1_history(payload))
        return {"a1": block(a1, rnd)}

    def _a5_round(result, rnd: int) -> AgentResult:
        """A5 on the finished round. Never raises: the round is already on disk, so an A5 error must not end it."""
        try:
            # a fixed logical name: the run id says which file the rows are in, and a recorded run
            # (rounds_recorded.csv) must hash like a normal one (rounds.csv) or offline replay of A5 never matches
            sources = a5_sources(rounds_path.parent, pd.DataFrame(rows_so_far, columns=ROUNDS_COLS), ROUNDS_CSV.name,
                                 files=("rounds",))
            sources.append(evidence_source(result.record, env.policy, rnd, result.row))  # what A2's reason may quote
            claims = round_claims(result.row, result.record)
            payload = a5_input(claims, sources, checklist, "round", rnd)
            # the round's own headline only (no A2 reason yet): the script checks it, no model call needed
            script_only = len(claims) == 1
            return (serve(offline, "a5", rnd, payload, a5_audit.validate_output, env.timer)
                    or a5_audit.run_a5(payload, None if script_only else client, env.timer, rnd,
                                       script_only=script_only))
        except Exception as e:  # noqa: BLE001 - the agents' contract: never break the round
            return AgentResult("A5", FALLBACK, [], A5_ERROR, "", [f"{type(e).__name__}: {e}"[:500]])

    def on_round(result) -> None:
        rnd = int(result.row["round"])
        rows_so_far.append(result.row)
        if write:  # the round shows now, A5's card reads "working..." (a5 null) until its line lands
            write_run(pd.DataFrame([result.row], columns=ROUNDS_COLS), [{**result.record, "a5": None}],
                      rounds_path, decisions_path)
        result.record["a5"] = block(_a5_round(result, rnd), rnd)
        if write:  # the same round again with A5 in it: the last line of a (run, round) wins
            write_run(pd.DataFrame(columns=ROUNDS_COLS), [result.record], rounds_path, decisions_path)

    return run_loop(env, n_rounds, state, before_decision=before_decision, on_round=on_round)


def compact_decisions(path: Path) -> None:
    """Keep only the last line of each (run, round), in round order: a live run writes A5's "working" line first,
    which only the live screen needs."""
    by_key = {(r["run"], r["round"]): r for r in read_records(path)}
    path.write_text("".join(json.dumps(by_key[k]) + "\n" for k in sorted(by_key)), encoding="utf-8")


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
        run_crew(env, make_client(), n_rounds, True, results_dir / ROUNDS_CSV.name, results_dir / DECISIONS_JSONL.name,
                 replayer=Replayer.from_file(results_dir / DECISIONS_RECORDED.name))
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
    ap.add_argument("--rounds", type=int, default=None)
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--no-write", action="store_true", help="print only")
    out.add_argument("--record", action="store_true", help="replace the committed recorded run with this one")
    ap.add_argument("--no-replay", action="store_true", help="offline, use the fallbacks, never the recording")
    args = ap.parse_args()
    if args.record and args.rounds is not None:
        ap.error("--record writes the full run that a fresh clone shows; drop --rounds")
    train, test = load_data(on_param_mismatch="error")
    env = make_env(train, test)
    client = make_client()
    recorded = (ROUNDS_RECORDED, DECISIONS_RECORDED)
    # --record builds the new run next to the committed one and swaps it in only once the run has finished
    paths = tuple(p.with_name(p.name + ".new") for p in recorded) if args.record else (ROUNDS_CSV, DECISIONS_JSONL)
    for p in paths if args.record else ():
        p.unlink(missing_ok=True)
    try:
        replayer = None if args.record or args.no_replay else Replayer.from_file(DECISIONS_RECORDED)
        rounds, records = run_crew(env, client, args.rounds, not args.no_write, *paths, replayer=replayer)
    except BaseException:
        for p in paths if args.record else ():
            p.unlink(missing_ok=True)
        raise
    if args.record:
        compact_decisions(paths[1])  # the committed file: one record per round (Crew Plan section 8)
        for new, old in zip(paths, recorded):
            os.replace(new, old)
        paths = recorded
    pd.set_option("display.width", 220)
    print(rounds.drop(columns=["run"]).round(3).to_string(index=False))
    print(f"\nA1 and A5 ({'live' if client else 'offline'}):")
    for r in records:
        a1, a5 = r["a1"], r["a5"]
        o = a1["output"]
        counts = pd.Series([v["verdict"] for v in a5["output"] or []]).value_counts().to_dict()
        print(f"  R{r['round']}: A1 {a1['status']}" + (f" ({a1['fallback_reason']})" if a1["fallback_reason"] else "")
              + f" -> {o['drift']}; A5 {a5['status']} -> {counts}")
    if not args.no_write:
        print(f"appended run {env.timer.run} to {paths[0].name} and {paths[1].name}")


if __name__ == "__main__":
    main()
