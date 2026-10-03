"""Timer wrapper for agent model calls, tool calls and local reads (build step 13).

Every timed call appends one JSON line to a per-call log:

    {run, round, agent, step, kind, status, start, end, ms,
     tokens_in, tokens_out, error}

- kind: "model" (LLM call), "tool" (tune(), run_round(), PSI calc...) or
  "retrieval" (local file / DataFrame reads). Lets us split time three ways.
- status: "LIVE", "FALLBACK", "REPLAY" or null when not applicable (tools).
- start / end: wall-clock ISO 8601 (UTC). ms: time.perf_counter() duration.
- tokens_in / tokens_out: null when unknown or not an LLM call (never 0).
- error: null, or "ExcType: message" if the block raised (it is re-raised).

Where the lines go (resolving a conflict between the two plans):
the Combined Plan says per-call timing goes to decisions.jsonl, but the Crew
Plan section 8 makes decisions.jsonl ONE record per round, with nested
per-agent fields (input hash, output, status, fallback reason, ms, tokens).
Writing per-call lines into it would break that schema and the readers
(app.py, replay.py). So per-call lines go to results/agent_calls.jsonl, and
round_agent_summary() rolls them up into the per-agent {status, ms, tokens}
block that the round record in decisions.jsonl embeds. Both plans are then
satisfied: every call is logged, and decisions.jsonl stays one line per round.

Percentiles use the nearest-rank method: the p-th percentile of n sorted
values is the value at rank ceil(p / 100 * n) (1-based). No interpolation,
so every reported number is a real observed call.

Standard library only; no network. No latency has been measured yet.
"""
from __future__ import annotations

import functools
import json
import math
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "results" / "agent_calls.jsonl"
KINDS = ("model", "tool", "retrieval")
STATUSES = ("LIVE", "FALLBACK", "REPLAY")
RUN_ROUND_STEP = "run_round"


@dataclass
class Call:
    """Handle yielded inside a timed block; set tokens / status before it ends."""

    tokens_in: int | None = None
    tokens_out: int | None = None
    status: str | None = None

    def usage(self, response: Any) -> Any:
        """Read tokens from any object with .usage.input_tokens/.output_tokens."""
        u = getattr(response, "usage", None)
        if u is not None:
            self.tokens_in = getattr(u, "input_tokens", self.tokens_in)
            self.tokens_out = getattr(u, "output_tokens", self.tokens_out)
        return response


def _check_status(status: str | None) -> None:
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


class AgentTimer:
    """Appends one JSONL record per timed call. Safe to share across threads."""

    def __init__(self, path: Path | str = DEFAULT_LOG, run: str | None = None) -> None:
        self.path = Path(path)
        # The log is append-only across rehearsal runs, so every run needs its own id
        # or per-round sums would mix runs. Default: UTC start time of this timer.
        self.run = run or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.round: int | None = None  # caller sets this at the start of each round
        self._lock = threading.Lock()

    @contextmanager
    def call(
        self, agent: str, step: str, kind: str = "model", status: str | None = None
    ) -> Iterator[Call]:
        """Time a block. Logs even if the block raises, then re-raises."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        _check_status(status)
        c = Call(status=status)
        error = None
        start_wall, t0 = time.time(), time.perf_counter()
        try:
            yield c
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self._write({
                "run": self.run, "round": self.round, "agent": agent, "step": step,
                "kind": kind, "status": c.status,
                "start": _iso(start_wall), "end": _iso(start_wall + ms / 1000.0),
                "ms": round(ms, 3), "tokens_in": c.tokens_in, "tokens_out": c.tokens_out,
                "error": error,
            })
        _check_status(c.status)  # only reached when the block did not raise

    def timed(self, agent: str, step: str, kind: str = "tool") -> Callable:
        """Decorator form. For kind="model", tokens are read from the return value."""

        def deco(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.call(agent, step, kind) as c:
                    out = fn(*args, **kwargs)
                    if kind == "model":
                        c.usage(out)
                    return out

            return wrapper

        return deco

    def _write(self, record: dict) -> None:
        line = json.dumps(record) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)


def load_records(path: Path | str = DEFAULT_LOG) -> list[dict]:
    """Read the per-call log; missing file means no records.

    A line without its trailing newline is skipped: the UI re-reads this file
    while the agent thread is appending, so the last line may be half-written.
    """
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.endswith("\n") and line.strip()]


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: sorted value at 1-based rank ceil(p/100 * n)."""
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p!r}")
    if not values:
        return None
    xs = sorted(values)
    # p * n / 100, not p / 100 * n: the latter gives rank 8 for p=7, n=100 (float error)
    rank = max(1, math.ceil(p * len(xs) / 100))
    return xs[rank - 1]


def _sum_or_none(values: list[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _stats(records: list[dict]) -> dict:
    ms = [r["ms"] for r in records]
    return {
        "n": len(records),
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
        "tokens_in": _sum_or_none([r["tokens_in"] for r in records]),
        "tokens_out": _sum_or_none([r["tokens_out"] for r in records]),
        "errors": sum(1 for r in records if r["error"]),
    }


def summarize(
    records: list[dict], exclude_status: tuple[str, ...] = ("REPLAY",)
) -> dict[str, dict[str, dict]]:
    """Per-call p50/p95 ms and token totals, grouped by agent, kind and agent/kind.

    REPLAY calls are excluded by default: their ms is a file read, not model
    latency, and must not enter the measured p50/p95. Pass () to keep them.
    """
    groups: dict[str, dict[str, list[dict]]] = {
        "by_agent": defaultdict(list), "by_kind": defaultdict(list),
        "by_agent_kind": defaultdict(list),
    }
    for r in records:
        if r["status"] in exclude_status:
            continue
        groups["by_agent"][r["agent"]].append(r)
        groups["by_kind"][r["kind"]].append(r)
        groups["by_agent_kind"][f"{r['agent']}/{r['kind']}"].append(r)
    return {g: {k: _stats(v) for k, v in sorted(rs.items())} for g, rs in groups.items()}


def _round_records(records: list[dict], round_id: int, run: str | None) -> list[dict]:
    rs = [r for r in records if r["round"] == round_id and (run is None or r["run"] == run)]
    if run is None and len({r["run"] for r in rs}) > 1:
        raise ValueError(f"round {round_id} appears in several runs; pass run=")
    return rs


def round_agent_summary(records: list[dict], round_id: int, run: str | None = None) -> dict[str, dict]:
    """Per-agent roll-up for one round, to embed in that round's decisions.jsonl record.

    ms is the sum of the agent's calls (excluding run_round, which round_time
    counts separately); status is the last non-null status the agent logged.
    run may be omitted only if the log holds a single run for this round.
    """
    out: dict[str, dict] = {}
    for r in _round_records(records, round_id, run):
        a = out.setdefault(r["agent"], {
            "status": None, "ms": 0.0, "model_ms": 0.0, "tool_ms": 0.0,
            "retrieval_ms": 0.0, "tokens_in": None, "tokens_out": None, "errors": [],
        })
        if r["step"] != RUN_ROUND_STEP:
            a["ms"] += r["ms"]
            a[f"{r['kind']}_ms"] += r["ms"]
        for k in ("tokens_in", "tokens_out"):
            if r[k] is not None:
                a[k] = (a[k] or 0) + r[k]
        if r["status"] is not None:
            a["status"] = r["status"]
        if r["error"]:
            a["errors"].append(f"{r['step']}: {r['error']}")
    return out


def round_time_ms(records: list[dict], round_id: int, run: str | None = None) -> float:
    """max(A1, A3) + A2 + tool_time(run_round) + max(A4, A5); missing agents count as 0."""
    agents = round_agent_summary(records, round_id, run)
    ms = {name: agents.get(name, {}).get("ms", 0.0) for name in ("A1", "A2", "A3", "A4", "A5")}
    run_round = sum(
        r["ms"] for r in _round_records(records, round_id, run) if r["step"] == RUN_ROUND_STEP
    )
    return max(ms["A1"], ms["A3"]) + ms["A2"] + run_round + max(ms["A4"], ms["A5"])
