"""Recorded replay (Tier 3, step 17, the replay.py half): the Wi-Fi-off demo, honestly labelled.

The recorded run is results/rounds_recorded.csv + decisions_recorded.jsonl (the plan's
agent_decisions_recorded.jsonl), written by python -m softsignal.crew --record. Two uses:

1. Agents, offline (Combined Plan: "live call if online, recorded replay if offline"). Replayer serves a recorded
   LIVE agent output when the same round and agent come up again with the same input hash, so a decision is
   never replayed onto a different input. The output is re-checked by the agent's own validator, then marked
   REPLAY (timed as a retrieval, so agent_timer.summarize leaves it out of the latency figures). No recorded
   output, a different input, or a recorded FALLBACK means the agent runs as usual (its fallback offline):
   a fallback is deterministic, so replaying one would add nothing. crew.py uses it when there is no client.
2. The Loop tab's Replay mode (UI handover section 5): the same screen fed from the recorded run, one round
   every REPLAY_STEP_S seconds, every agent badge REPLAY (as_replay keeps the recorded status, so the cards and
   the log still say what happened when it was recorded).

Reads files only; never calls a model.
"""
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from softsignal.agent_timer import AgentTimer
from softsignal.agents.base import LIVE, REPLAY, AgentResult, input_hash
from softsignal.data import ROOT

RESULTS = ROOT / "results"
ROUNDS_RECORDED, DECISIONS_RECORDED = "rounds_recorded.csv", "decisions_recorded.jsonl"
AGENT_KEYS = ("a1", "a2", "a3", "a4", "a5")
REPLAY_STEP_S = 3.0  # UI handover section 5: one round every 3 seconds


def read_records(path: Path) -> list[dict]:
    """The complete JSON lines of a decisions file (a half-written last line or bad JSON is skipped)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").split("\n")[:-1]:  # the piece after the last newline is partial
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("run"), str) and isinstance(rec.get("round"), int):
            out.append(rec)
    return out


@dataclass(frozen=True)
class Replayer:
    """Recorded LIVE agent outputs of one run: (round, agent key) -> (input hash, output)."""

    run: str | None = None
    blocks: dict = field(default_factory=dict)

    @classmethod
    def from_records(cls, records: list[dict], run: str | None = None) -> "Replayer":
        """The given run, else the latest one (run ids are UTC timestamps). Only LIVE blocks with an input hash."""
        runs = sorted({r["run"] for r in records})
        run = run if run is not None else (runs[-1] if runs else None)
        blocks = {}
        for rec in records:
            if rec["run"] != run:
                continue
            for key in AGENT_KEYS:
                b = rec.get(key)
                live = isinstance(b, dict) and b.get("status") == LIVE and b.get("input_hash")
                if live and b.get("output") is not None:
                    blocks[(rec["round"], key)] = (b["input_hash"], b["output"])
        return cls(run, blocks)

    @classmethod
    def from_file(cls, path: Path = RESULTS / DECISIONS_RECORDED) -> "Replayer":
        return cls.from_records(read_records(Path(path)))

    def __len__(self) -> int:
        return len(self.blocks)

    def lookup(self, round_id: int, key: str, payload_hash: str):
        """The recorded output for this round and agent if it was made from the same input, else None."""
        hit = self.blocks.get((int(round_id), key))
        return hit[1] if hit is not None and hit[0] == payload_hash else None

    def mismatch_note(self, round_id: int, key: str, payload_hash: str) -> str | None:
        """A note when this round and agent were recorded LIVE from a different input (so the recording was not
        served), else None. Not the same as "nothing recorded": the caller puts it in the agent block's errors."""
        hit = self.blocks.get((int(round_id), key))
        if hit is None or hit[0] == payload_hash:
            return None
        return f"replay_hash_mismatch: recorded input {hit[0]}, now {payload_hash}; recording not served"


def serve(replayer: "Replayer | None", key: str, round_id: int, payload: dict,
          validate: Callable[[object, dict], tuple], timer: AgentTimer | None = None) -> AgentResult | None:
    """A REPLAY result from the recording, or None (then the caller runs the agent as usual).

    validate is the agent's own output check (e.g. a1_drift.validate_output): a recording that would not pass
    it today is not served.
    """
    if replayer is None or not len(replayer):
        return None
    h = input_hash(payload)
    output = replayer.lookup(round_id, key, h)
    if output is None:
        return None
    try:  # a validator that changed since the recording must mean "no replay", never a crash
        if validate(output, payload)[0] is not None:
            return None
    except Exception:  # noqa: BLE001
        return None
    if timer is not None:
        with timer.call(key.upper(), "replay", "retrieval", status=REPLAY, round_id=round_id):
            pass
    return AgentResult(key.upper(), REPLAY, output, None, h)


# ---- the Loop tab's Replay mode ----
def as_replay(record: dict) -> dict:
    """A copy of a recorded round with every agent block that ran marked REPLAY; recorded_status keeps what it
    was (LIVE / FALLBACK), so the screen can still say what happened when it was recorded."""
    out = dict(record)
    for key in AGENT_KEYS:
        b = record.get(key)
        if isinstance(b, dict) and b.get("status") is not None:
            out[key] = {**b, "status": REPLAY, "recorded_status": b.get("recorded_status") or b["status"]}
    return out


def load_recorded(results_dir: Path = RESULTS) -> tuple[pd.DataFrame, list[dict]]:
    """The latest recorded run: (its rounds.csv rows, its records marked REPLAY), both in round order."""
    records = read_records(Path(results_dir) / DECISIONS_RECORDED)
    path = Path(results_dir) / ROUNDS_RECORDED
    rounds = pd.read_csv(path, dtype={"run": str}) if path.exists() else pd.DataFrame(columns=["run", "round"])
    runs = sorted(set(rounds["run"]) | {r["run"] for r in records})
    if not runs:
        return rounds.iloc[0:0], []
    run = runs[-1]
    rounds = rounds[rounds["run"] == run].sort_values("round").reset_index(drop=True)
    by_round = {r["round"]: r for r in records if r["run"] == run}  # last line wins, as in the Loop tab
    return rounds, [as_replay(by_round[k]) for k in sorted(by_round)]


def revealed_rounds(started: float, now: float | None = None, total: int = 8, step_s: float = REPLAY_STEP_S) -> int:
    """How many rounds a replay started at `started` shows by `now`: R0 at once, then one more every step_s."""
    now = time.time() if now is None else now
    return max(1, min(total, 1 + int(max(0.0, now - started) // step_s)))
