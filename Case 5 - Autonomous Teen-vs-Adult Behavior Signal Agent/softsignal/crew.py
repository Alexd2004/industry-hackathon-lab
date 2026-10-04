"""Crew run (Tier 3): loop.py round by round with the agents built so far, each round appended as it lands.

Today that is loop.py's rule plus A1 (drift watcher). Each round:

    prior = state.seen                                  # the rows of every earlier batch
    result = loop.run_round(state, batch, env)          # the rule scores, verifies, reveals, refits, decides
    A1 on (prior, batch rows, the round's score PSI, audit counts, earlier rounds' PSI)  -> record["a1"]
    loop.write_run(this row, this record)               # appended now: the Loop tab shows it within a second

A1 runs after the rule decided. With no A2 yet its verdict informs no decision, and its input is what it
would be before the decision (the reveal, the PSI and the audit counts are final by then). When A2 lands, A1
moves before the decision with the same input. A4 is not run here: while the loop is in SHADOW the starter
blend picks the verify band and has no explanations, so A4 has nothing to explain; wire it once the live
rule is the stack (agents/a4_triage.py). Every run has its own run id (its own AgentTimer), so rehearsal
runs never mix in the Loop tab.

Run: python -m softsignal.crew [--rounds N] [--no-write]   (agents are offline unless ANTHROPIC_API_KEY is set)
The Loop tab's "Run loop" button starts the same run in a background thread (start_background).
"""
import argparse
import threading
from pathlib import Path

import pandas as pd

from softsignal.agent_timer import AgentTimer, load_records, round_agent_summary
from softsignal.agents.a1_drift import run_a1
from softsignal.agents.base import make_client, merge_block
from softsignal.agents.contracts import a1_history, a1_input
from softsignal.data import load_data
from softsignal.features import FEATURE_COLS
from softsignal.loop import DECISIONS_JSONL, ROUNDS_CSV, Env, State, make_env, new_state, round0, run_round, write_run
from softsignal.metrics import ROUNDS_COLS


def _rollup(timer: AgentTimer, round_id: int, agent: str) -> dict | None:
    """The agent's timed calls this round (ms, tokens, call errors), from agent_calls.jsonl."""
    return round_agent_summary(load_records(timer.path), round_id, timer.run).get(agent)


def run_crew(env: Env, client=None, n_rounds: int | None = None, write: bool = True,
             rounds_path: Path = ROUNDS_CSV, decisions_path: Path = DECISIONS_JSONL,
             state: State | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """R0 then each oracle batch (at most n_rounds), A1 on every round, each round appended when it ends.

    Returns the rounds table and the records, like loop.run_loop. client: base.make_client() (None =
    offline, the agents use their fallbacks).
    """
    state = new_state(env.policy) if state is None else state
    psi_drift = env.policy.get("psi_drift")
    rows, records, history = [], [], []

    def finish(result, a1) -> None:
        result.record["a1"] = merge_block(a1, _rollup(env.timer, result.row["round"], a1.agent))
        rows.append(result.row)
        records.append(result.record)
        if write:
            write_run(pd.DataFrame([result.row], columns=ROUNDS_COLS), [result.record], rounds_path, decisions_path)

    none = pd.DataFrame(columns=FEATURE_COLS)
    result = round0(env, state)  # no batch yet: A1 returns insufficient_data without a model call
    finish(result, run_a1(a1_input(none, none, None, env.oracle.audit_counts(), [], 0, psi_drift),
                          client, env.timer, 0))
    for batch in env.oracle:
        if n_rounds is not None and batch.round > n_rounds:
            break
        prior = state.seen  # reassigned (not mutated) by the round, so this stays the earlier batches
        result = run_round(state, batch, env)
        payload = a1_input(prior, batch.rows, result.row["psi"], env.oracle.audit_counts(), history,
                           batch.round, psi_drift)
        finish(result, run_a1(payload, client, env.timer, batch.round))
        history.append(a1_history(payload))
    return pd.DataFrame(rows, columns=ROUNDS_COLS), records


# ---- background run for the Loop tab ----
_lock = threading.Lock()
_current = {"thread": None, "run": None, "error": None}


def status() -> dict:
    """{"running": bool, "run": the latest run id started here or None, "error": its error text or None}."""
    t = _current["thread"]
    return {"running": t is not None and t.is_alive(), "run": _current["run"], "error": _current["error"]}


def _background(timer: AgentTimer, results_dir: Path, n_rounds: int | None) -> None:
    try:
        train, test = load_data(on_param_mismatch="error")
        env = make_env(train, test, timer=timer)
        run_crew(env, make_client(), n_rounds, True, results_dir / ROUNDS_CSV.name, results_dir / DECISIONS_JSONL.name)
    except Exception as e:  # noqa: BLE001 - shown in the Loop tab instead of dying silently in a thread
        _current["error"] = f"{type(e).__name__}: {e}"[:500]


def start_background(results_dir: Path = ROUNDS_CSV.parent, n_rounds: int | None = None) -> str | None:
    """Start a run in a daemon thread and return its run id, or None while another run is still going.

    The thread only appends to rounds.csv and decisions.jsonl (and agent_calls.jsonl); the app only reads them.
    """
    with _lock:
        if status()["running"]:
            return None
        timer = AgentTimer()  # its own run id: never shared with an earlier run of this process
        _current.update(run=timer.run, error=None)
        _current["thread"] = threading.Thread(target=_background, args=(timer, Path(results_dir), n_rounds),
                                              daemon=True, name="softsignal-crew")
        _current["thread"].start()
        return timer.run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--no-write", action="store_true", help="print only, leave rounds.csv and decisions.jsonl alone")
    args = ap.parse_args()
    train, test = load_data(on_param_mismatch="error")
    env = make_env(train, test)
    client = make_client()
    rounds, records = run_crew(env, client, args.rounds, write=not args.no_write)
    pd.set_option("display.width", 220)
    print(rounds.drop(columns=["run"]).round(3).to_string(index=False))
    print(f"\nA1 ({'live' if client else 'offline: policy.yaml threshold'}):")
    for r in records:
        a1 = r["a1"]
        out = a1["output"]
        print(f"  R{r['round']}: {a1['status']}" + (f" ({a1['fallback_reason']})" if a1["fallback_reason"] else "")
              + f" -> {out['drift']}: {out['reason']}")
    if not args.no_write:
        print(f"appended run {env.timer.run} to {ROUNDS_CSV.name} and {DECISIONS_JSONL.name}")


if __name__ == "__main__":
    main()
