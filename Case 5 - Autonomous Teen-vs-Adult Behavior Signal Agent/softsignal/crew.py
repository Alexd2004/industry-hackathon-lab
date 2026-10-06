"""Crew run (Tier 3): loop.py with the agents built so far, each round appended as it lands.

Today that is loop.py's rule plus A1 (drift watcher), A3 (error analyst), A2 (controller) and A5 (honesty auditor), through
loop.run_loop's three callbacks:

    before_decision(state, batch, prior, psi)  -> A1 on (prior = the earlier batches, the batch rows, the score
                                                  PSI while the live rule is the starter, audit counts, earlier
                                                  rounds' PSI), then A3 on (the audit-slice errors of earlier
                                                  rounds, see a3_payload); their blocks go into the round's record
    decide(context, blocks)                    -> A2 on (the live thresholds, audit counts, bounds, guards, the rule's
                                                  decision, A1's output and A3's when it analysed something),
                                                  after the streak update and before the refit; its block and the diff
                                                  against the rule go in the record, and its action and cap are applied
                                                  unless it fell back or apply_a2 is False
    on_round(result)                           -> when writing, the round is appended at once with A5 "working"
                                                  (a5 null), so the screen never waits on A5; then A5 checks the
                                                  finished round (its headline numbers and A2's reason, against
                                                  this run's rows so far), after the decision, so nothing A5
                                                  writes can reach A1-A4; then the record is appended again with
                                                  A5 filled in (decisions.jsonl: the last line of a round wins)

Offline (no client), each agent first asks replay.Replayer for a recorded LIVE output made from the same input
(marked REPLAY); otherwise it uses its fallback. --record never replays: a recording holds live or fallback
output only. A2 is replayed too (its recorded decision is applied, marked REPLAY, only when its input hash matches
and it still passes validate_output). A recording made from a different input is not served and the agent block's
errors say replay_hash_mismatch.

A1 runs after the reveal and the PSI, before the decision, where A2 reads it. Offline with nothing recorded A2
falls back to the rule's own decision every round, so such a run's decisions are exactly loop.run_loop's. Score PSI goes
to A1 only while the live rule is the starter: once the stack is live every refit changes the model, and
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
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.agent_timer import DEFAULT_LOG, AgentTimer, round_agent_summary
from softsignal.agents import a1_drift, a2_controller, a3_errors, a5_audit
from softsignal.agents.a2_controller import run_a2
from softsignal.agents.base import (FALLBACK, OFFLINE, REPLAY_MISMATCH, AgentResult, input_hash, is_real_fallback,
                                    make_client, merge_block)
from softsignal.agents.contracts import (
    A3_COLS, INSUFFICIENT_INPUT, a1_history, a1_input, a2_input, a3_input, a5_input, a5_sources, evidence_source,
    load_checklist, round_claims,
)
from softsignal.data import load_data
from softsignal.explain import explain_frame
from softsignal.features import ID_COL, TARGET
from softsignal.loop import (DECISIONS_JSONL, ROUNDS_CSV, RUN_MODES, SHADOW, Decide, DecisionContext, Env, State,
                             check_rounds_header, make_env, run_loop, write_run)
from softsignal.metrics import ROUNDS_COLS
from softsignal.replay import Replayer, read_records, serve

A5_ERROR = "agent_error"  # A5 raised: the round keeps going with an empty, FALLBACK A5 block
A3_ERROR = "agent_error"  # A3's input or run raised: the round keeps going with a FALLBACK, no-analysis A3 block


def a3_payload(env: Env, state: State, batch, prior: pd.DataFrame) -> dict:
    """A3's input for one round (call it from before_decision, where state.seen and state.live_scores already
    include this round's batch and prior does not).

    Errors: the audit-slice accounts revealed in EARLIER rounds, each scored as the live rule scored its batch
    when it arrived (state.live_scores, never re-scored, so never in-sample), against the live t_verify.
    After a promote live_scores restarts, so only the rounds scored since then (on the live stack's scale) count.
    Signals and words: explain.py's contributions from the live stack, else the candidate (the latest refit), else
    none. The candidate was fit on these accounts' labels, so the signals describe the accounts and are not an
    out-of-sample explanation; with no stack yet there are no signals and A3 returns insufficient_data.
    """
    rnd = 0 if batch is None else batch.round
    n_cur = 0 if batch is None else len(batch.rows)
    scores = np.asarray(state.live_scores, dtype=float)[: len(state.live_scores) - n_cur]
    rows = prior.iloc[len(prior) - len(scores):]  # live_scores and seen grow together, so these line up
    audit = env.oracle.revealed("audit", before_round=rnd)
    labels = dict(zip(audit[ID_COL].astype(str), audit[TARGET].astype(int)))
    ids = rows[ID_COL].astype(str).to_numpy()
    keep = np.array([i in labels for i in ids], dtype=bool)
    rows, scores, ids = rows[keep], scores[keep], ids[keep]
    model = state.live.model or (None if state.candidate is None else state.candidate.model)
    if model is not None and len(rows):
        frame = explain_frame(model, rows)
        frame["score"] = scores  # the live rule's score at arrival, not the explaining model's
    else:
        frame = pd.DataFrame({ID_COL: ids, "score": scores})
        for col in A3_COLS[2:]:
            frame[col] = np.nan if col.startswith("v") else ""
    th = state.live.th
    return a3_input(frame[A3_COLS], {i: labels[i] for i in ids}, float(th.t_verify), rnd,
                    env.policy.get("min_a3_errors"), test_ids=env.test[ID_COL])

ROUNDS_RECORDED = ROUNDS_CSV.with_name("rounds_recorded.csv")
DECISIONS_RECORDED = DECISIONS_JSONL.with_name("decisions_recorded.jsonl")


def run_crew(env: Env, client=None, n_rounds: int | None = None, write: bool = False,
             rounds_path: Path = ROUNDS_CSV, decisions_path: Path = DECISIONS_JSONL,
             state: State | None = None, replayer: Replayer | None = None,
             checklist: list[dict] | None = None, apply_a2: bool = True,
             decider: Decide | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """R0 then each oracle batch (at most n_rounds) with A1 and A5 on every round and A2 from R1. Returns
    (rounds, records) like loop.run_loop. write=True appends each round to rounds_path / decisions_path when it ends.
    client: base.make_client() (None = offline). replayer: recorded outputs to serve while offline (None = none;
    A1 and A5 only). checklist: A5's risk list (default claims/risks.yaml; [] if that file is missing).
    A2's decision is logged next to the rule's; apply_a2 (crew mode) applies A2's own decision, False logs it only.
    decider: a scripted stand-in for A2, (context, blocks) -> {"a2": block}; replaces the agent call (drift_check)."""
    psi_drift = env.policy.get("psi_drift")
    history: list[dict] = []
    rows_so_far: list[dict] = []
    offline = replayer if client is None else None  # live when online, recorded replay when offline
    if checklist is None:
        try:
            checklist = load_checklist()
        except FileNotFoundError:
            checklist = []

    def replay_or_run(key: str, rnd: int, payload: dict, validate, run) -> AgentResult:
        """The recorded output when offline and the input matches (REPLAY), else run() as usual. If this round and
        agent were recorded from a different input, the result's errors say so, and an offline fallback is reported
        as replay_hash_mismatch: the recording exists but no longer fits (an input contract changed)."""
        got = serve(offline, key, rnd, payload, validate, env.timer)
        if got is not None:
            return got
        result = run()
        note = offline.mismatch_note(rnd, key, input_hash(payload)) if offline is not None else None
        if note:
            result.errors.append(note)
            if result.fallback_reason == OFFLINE:
                result.fallback_reason = REPLAY_MISMATCH
        return result

    def block(result, rnd: int) -> dict:
        return merge_block(result, round_agent_summary(env.timer.records, rnd, env.timer.run).get(result.agent))

    def before_decision(state: State, batch, prior: pd.DataFrame, psi_val) -> dict:
        rnd = 0 if batch is None else batch.round
        rows = prior.iloc[0:0] if batch is None else batch.rows
        score_psi = psi_val if state.mode == SHADOW else None  # the live model changes every refit once ACTIVE
        payload = a1_input(prior, rows, score_psi, env.oracle.audit_counts(), history, rnd, psi_drift)
        a1 = replay_or_run("a1", rnd, payload, a1_drift.validate_output,
                           lambda: a1_drift.run_a1(payload, client, env.timer, rnd))
        if batch is not None:
            history.append(a1_history(payload))
        return {"a1": block(a1, rnd), "a3": block(_a3_round(state, batch, prior, rnd), rnd)}

    def _a3_round(state: State, batch, prior: pd.DataFrame, rnd: int) -> AgentResult:
        """A3 on this round's earlier-round errors. Never raises: an A3 error must not take A1 or the round down."""
        try:
            payload = a3_payload(env, state, batch, prior)
            return replay_or_run("a3", rnd, payload, a3_errors.validate_output,
                                 lambda: a3_errors.run_a3(payload, client, env.timer, rnd))
        except Exception as e:  # noqa: BLE001 - the agents' contract: never break the round
            return AgentResult("A3", FALLBACK, a3_errors.fallback_output(), A3_ERROR, "", [f"{type(e).__name__}: {e}"[:500]])

    def _a5_round(result, rnd: int) -> AgentResult:
        """A5 on the finished round. Never raises: the round is already on disk, so an A5 error must not end it."""
        try:
            # a fixed logical name: the run id says which file the rows are in, and a recorded run
            # (rounds_recorded.csv) must hash like a normal one (rounds.csv) or offline replay of A5 never matches
            sources = a5_sources(rounds_path.parent, pd.DataFrame(rows_so_far, columns=ROUNDS_COLS), ROUNDS_CSV.name,
                                 files=("rounds",))
            sources.append(evidence_source(result.record, env.policy, rnd, result.row,  # what A2's reason may quote
                                           audit=env.oracle.audit_counts()))
            claims = round_claims(result.row, result.record)
            payload = a5_input(claims, sources, checklist, "round", rnd)
            # the round's own headline only (no A2 reason yet): the script checks it, no model call needed
            script_only = len(claims) == 1
            return replay_or_run("a5", rnd, payload, a5_audit.validate_output,
                                 lambda: a5_audit.run_a5(payload, None if script_only else client, env.timer, rnd,
                                                         script_only=script_only))
        except Exception as e:  # noqa: BLE001 - the agents' contract: never break the round
            return AgentResult("A5", FALLBACK, [], A5_ERROR, "", [f"{type(e).__name__}: {e}"[:500]])

    def decide(ctx: DecisionContext, blocks: dict) -> dict:
        a1 = (blocks.get("a1") or {}).get("output")  # whole, as A1 returned it
        a3 = (blocks.get("a3") or {}).get("output")  # whole when it analysed something, else insufficient_data
        a3_ok = isinstance(a3, dict) and a3.get("status") == "ok"
        payload = a2_input(ctx.round, ctx.thresholds, ctx.audit, ctx.bounds, ctx.guards, ctx.rule,
                           a1=a1 if isinstance(a1, dict) else INSUFFICIENT_INPUT,
                           a3=a3 if a3_ok else INSUFFICIENT_INPUT, policy=env.policy)
        a2 = replay_or_run("a2", ctx.round, payload, a2_controller.validate_output,
                           lambda: run_a2(payload, client, env.timer, ctx.round))
        return {"a2": merge_block(a2, round_agent_summary(env.timer.records, ctx.round, env.timer.run).get(a2.agent))}

    def on_round(result) -> None:
        rnd = int(result.row["round"])
        rows_so_far.append(result.row)
        if write:  # the round shows now, A5's card reads "working..." (a5 null) until its line lands
            write_run(pd.DataFrame([result.row], columns=ROUNDS_COLS), [{**result.record, "a5": None}],
                      rounds_path, decisions_path)
        result.record["a5"] = block(_a5_round(result, rnd), rnd)
        if write:  # the same round again with A5 in it: the last line of a (run, round) wins
            write_run(pd.DataFrame(columns=ROUNDS_COLS), [result.record], rounds_path, decisions_path)

    return run_loop(env, n_rounds, state, before_decision=before_decision, on_round=on_round,
                    decide=decider or decide,
                    apply_a2=apply_a2)


def compact_decisions(path: Path) -> None:
    """Keep only the last line of each (run, round), in round order: a live run writes A5's "working" line first,
    which only the live screen needs."""
    by_key = {(r["run"], r["round"]): r for r in read_records(path)}
    path.write_text("".join(json.dumps(by_key[k]) + "\n" for k in sorted(by_key)), encoding="utf-8")


AGENT_KEYS = ("a1", "a2", "a3", "a4", "a5")


def run_fallbacks(records: list[dict]) -> int:
    """Real fallbacks in one run (base.is_real_fallback: a reply refused by a check, or no usable reply). The
    scripted ones (insufficient_data, script_only) needed no model call and offline ones had none to make."""
    return sum(1 for r in records for k in AGENT_KEYS if is_real_fallback(r.get(k)))


def pick_canonical(records: list[dict], n_rounds: int) -> str | None:
    """The run id to commit as the recording, or None if no run has all n_rounds rounds.

    Rule, fixed before any run is looked at: among complete runs, the fewest real fallbacks (run_fallbacks), ties to
    the earliest run id. It never looks at recall, false-teen or any metric, so picking the recording cannot
    be picking the best-looking numbers.
    """
    by_run: dict[str, list[dict]] = {}
    for r in records:
        by_run.setdefault(r["run"], []).append(r)
    complete = {run: rs for run, rs in by_run.items() if len({r["round"] for r in rs}) == n_rounds}
    return min(complete, key=lambda run: (run_fallbacks(complete[run]), run), default=None)


def keep_run(paths: tuple[Path, Path], run: str) -> None:
    """Rewrite the rounds csv and decisions jsonl at paths so they hold only this run."""
    rounds = pd.read_csv(paths[0], dtype={"run": str})
    rounds[rounds["run"] == run].to_csv(paths[0], index=False, lineterminator="\n")
    kept = [r for r in read_records(paths[1]) if r["run"] == run]
    paths[1].write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")


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
    ap.add_argument("--mode", choices=RUN_MODES, default="crew",
                    help="crew: apply A2; rule: apply the rule")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat the full run N times, each with its own run id (timings for p50 / p95); "
                         "with --record the canonical run (pick_canonical) is the one committed")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--no-write", action="store_true", help="print only")
    out.add_argument("--record", action="store_true", help="replace the committed recorded run with this one")
    ap.add_argument("--no-replay", action="store_true", help="offline, use the fallbacks, never the recording")
    args = ap.parse_args()
    if args.record and args.rounds is not None:
        ap.error("--record writes the full run that a fresh clone shows; drop --rounds")
    if args.record and args.mode != "crew":
        ap.error("--record writes the crew run that a fresh clone shows; drop --mode rule")
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    if not (args.no_write or args.record):
        check_rounds_header(ROUNDS_CSV, ROUNDS_COLS)  # fail now, not after every refit has run
    train, test = load_data(on_param_mismatch="error")
    client = make_client()
    recorded = (ROUNDS_RECORDED, DECISIONS_RECORDED)
    # --record builds the new run next to the committed one and swaps it in only once the run has finished
    paths = tuple(p.with_name(p.name + ".new") for p in recorded) if args.record else (ROUNDS_CSV, DECISIONS_JSONL)
    for p in paths if args.record else ():
        p.unlink(missing_ok=True)
    try:
        replayer = None if args.record or args.no_replay else Replayer.from_file(DECISIONS_RECORDED)
        all_rounds, all_records = [], []
        for _ in range(args.runs):
            # a fresh timer per run: its own run id, its calls logged next to the other runs' calls
            env = make_env(train, test, timer=AgentTimer(DEFAULT_LOG) if args.runs > 1 else None)
            rounds, records = run_crew(env, client, args.rounds, not args.no_write, *paths, replayer=replayer,
                                       apply_a2=args.mode == "crew")
            all_rounds.append(rounds)
            all_records.append(records)
            if args.runs > 1:
                print(f"run {env.timer.run}: {run_fallbacks(records)} real fallbacks")
    except BaseException:
        for p in paths if args.record else ():
            p.unlink(missing_ok=True)
        raise
    if args.record:
        compact_decisions(paths[1])  # the committed file: one record per round (Crew Plan section 8)
        if args.runs > 1:
            canon = pick_canonical(read_records(paths[1]), len(all_rounds[0]))
            if canon is None:
                raise SystemExit("no run has every round; nothing recorded")
            keep_run(paths, canon)
            rounds = next(r for r in all_rounds if (r["run"] == canon).all())
            records = [r for rs in all_records for r in rs if r["run"] == canon]
            print(f"canonical run: {canon} (fewest real fallbacks, ties to the earliest; metrics not used)")
        from softsignal.recorded_check import scan_recorded  # lazy: recorded_check imports this module

        problems = scan_recorded(*paths)  # the repo is public: nothing is swapped in if the new run is not clean
        if problems:
            for p in paths:
                p.unlink(missing_ok=True)
            raise SystemExit("recording not saved, safety scan found:\n  " + "\n  ".join(problems))
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
