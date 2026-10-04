"""Timer wrapper for agent model calls, tool calls and local reads (build step 13).

Every timed call appends one JSON line to a per-call log:

    {run, round, agent, step, kind, status, start, end, ms,
     tokens_in, tokens_out, error}

- kind: "model" (LLM call), "tool" (tune(), run_round(), PSI calc...) or
  "retrieval" (local file / DataFrame reads). Lets us split time three ways.
- status: "LIVE", "FALLBACK", "REPLAY" or null when not applicable (tools).
- round: the round when the call STARTED (a slow call that ends after the
  orchestrator moved on still belongs to its own round).
- start / end: wall-clock ISO 8601 (UTC). ms: time.perf_counter() duration.
- tokens_in / tokens_out: null when unknown or not an LLM call (never 0).
  tokens_in includes prompt-cache reads and writes.
- error: null, or "ExcType: message" (cut to 500 chars) if the block raised
  (it is re-raised).

One timer per process: every agent must use get_timer() (or one timer the
orchestrator creates and hands to each agent). Separate AgentTimer objects get
separate run ids and separate locks, so one round would be split across runs.

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
so every reported number is a real observed call. With n < 20, p95 is the
max, so always report n next to it.

Standard library only; no network. No latency has been measured yet.
"""
from __future__ import annotations

import functools
import json
import math
import numbers
import sys
import threading
import time
import uuid
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
RECORD_KEYS = frozenset({"run", "round", "agent", "step", "kind", "status", "start", "end",
                         "ms", "tokens_in", "tokens_out", "error"})
MAX_ERROR_CHARS = 500
_UNSET = object()


@dataclass
class Call:
    """Handle yielded inside a timed block; set tokens / status before it ends."""

    tokens_in: int | None = None
    tokens_out: int | None = None
    status: str | None = None

    def usage(self, response: Any) -> Any:
        """Read tokens from any object with .usage.input_tokens/.output_tokens.

        Prompt-cache reads and writes are input that input_tokens leaves out, so
        they are added to tokens_in. tokens_in is tokens processed, not a cost
        basis: cache reads and writes are priced differently from plain input.
        """
        u = getattr(response, "usage", None)
        if u is not None:
            parts = [_as_int(getattr(u, k, None)) for k in
                     ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")]
            known = [v for v in parts if v is not None]
            if known:
                self.tokens_in = sum(known)
            out = _as_int(getattr(u, "output_tokens", None))
            if out is not None:
                self.tokens_out = out
        return response


def _as_int(v: Any) -> int | None:
    # numbers.Integral also covers numpy ints; bool and mocks are not token counts
    return int(v) if isinstance(v, numbers.Integral) and not isinstance(v, bool) else None


def _json_default(o: Any) -> Any:
    # numpy scalars and other odd values must not make a log write raise
    if isinstance(o, numbers.Integral):
        return int(o)
    if isinstance(o, numbers.Real):
        return float(o)
    return str(o)


def _check_status(status: str | None) -> None:
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


def _new_run_id() -> str:
    # timestamp keeps ids sortable; the suffix keeps them unique on coarse clocks
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:6]


class AgentTimer:
    """Appends one JSONL record per timed call. Safe to share across threads."""

    def __init__(self, path: Path | str = DEFAULT_LOG, run: str | None = None) -> None:
        self.path = Path(path)
        # The log is append-only across rehearsal runs, so every run needs its own id
        # or per-round sums would mix runs. Default: UTC start time of this timer.
        self.run = run or _new_run_id()
        self.round: int | None = None  # caller sets this at the start of each round
        self.write_errors = 0  # log writes that failed; timing never breaks the agent
        self.records: list[dict] = []  # this timer's own records, so a round roll-up never re-reads the log
        self._lock = threading.Lock()

    @contextmanager
    def call(
        self, agent: str, step: str, kind: str = "model", status: str | None = None,
        round_id: Any = _UNSET,
    ) -> Iterator[Call]:
        """Time a block. Logs even if the block raises, then re-raises.

        The round is read when the block starts; pass round_id to set it explicitly.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        _check_status(status)
        rnd = self.round if round_id is _UNSET else round_id
        c = Call(status=status)
        error = None
        bad_status = False
        start_wall, t0 = time.time(), time.perf_counter()
        try:
            yield c
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
            raise
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            status_out = c.status
            if status_out is not None and status_out not in STATUSES:
                # never write a value outside the frozen format; record why instead
                bad_status = True
                error = error or f"ValueError: invalid status {status_out!r}"[:MAX_ERROR_CHARS]
                status_out = None
            self._write({
                "run": self.run, "round": rnd, "agent": agent, "step": step,
                "kind": kind, "status": status_out,
                "start": _iso(start_wall), "end": _iso(start_wall + ms / 1000.0),
                "ms": round(ms, 3), "tokens_in": c.tokens_in, "tokens_out": c.tokens_out,
                "error": error,
            })
        if bad_status:  # only reached when the block did not raise
            _check_status(c.status)

    def timed(
        self, agent: str, step: str, kind: str = "tool", status: str | None = None
    ) -> Callable:
        """Decorator form. For kind="model", tokens are read from the return value.

        status is fixed for every call; a path that can fall back must use call().
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        _check_status(status)

        def deco(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.call(agent, step, kind, status=status) as c:
                    out = fn(*args, **kwargs)
                    if kind == "model":
                        c.usage(out)
                    return out

            return wrapper

        return deco

    def _write(self, record: dict) -> None:
        # runs inside finally: raising here would replace the block's own
        # exception (e.g. the timeout a fallback is waiting to catch)
        try:
            line = json.dumps(record, default=_json_default) + "\n"
        except Exception as exc:  # circular or deep values, a __str__ that raises
            self._write_failed(f"could not encode record: {exc}")
            return
        with self._lock:
            self.records.append(record)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line)
            except OSError as exc:
                self._write_failed(f"could not write {self.path}: {exc}")

    def _write_failed(self, msg: str) -> None:
        # warn once; callers show write_errors next to the timing summary
        self.write_errors += 1
        if self.write_errors == 1:
            try:
                print(f"agent_timer: {msg} (further failures only counted in write_errors)",
                      file=sys.stderr)
            except Exception:  # closed or broken stderr: the warning is best effort
                pass


_default_timer: AgentTimer | None = None
_default_lock = threading.Lock()


def get_timer() -> AgentTimer:
    """The one shared timer for this process (one run id, one lock)."""
    global _default_timer
    with _default_lock:
        if _default_timer is None:
            _default_timer = AgentTimer()
        return _default_timer


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _valid_record(obj: Any) -> bool:
    """True if obj has every field the readers rely on, with a usable type."""
    if not (isinstance(obj, dict) and RECORD_KEYS <= obj.keys()):
        return False
    if not (_is_num(obj["ms"]) and obj["kind"] in KINDS
            and (obj["status"] is None or obj["status"] in STATUSES)
            and isinstance(obj["agent"], str) and isinstance(obj["step"], str)):
        return False
    if not all(v is None or (isinstance(v, int) and not isinstance(v, bool))
               for v in (obj["tokens_in"], obj["tokens_out"])):
        return False
    try:
        _parse_ts(obj["start"])
        _parse_ts(obj["end"])
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def load_records(path: Path | str = DEFAULT_LOG) -> list[dict]:
    """Read the per-call log; missing file means no records.

    A line without its trailing newline is skipped: the UI re-reads this file
    while the agent thread is appending, so the last line may be half-written.
    A line that is not a full record (a killed write that the next append ran
    into, or stray JSON) is skipped too, so one bad line cannot break every reader.
    """
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n") or not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _valid_record(obj):
                out.append(obj)
    return out


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
    """Per-call p50/p95 ms and token totals, grouped by agent, kind, agent/kind and status.

    REPLAY calls are excluded by default: their ms is a file read, not model
    latency, and must not enter the measured p50/p95. Pass () to keep them.
    by_status splits LIVE from FALLBACK (timed-out calls) for model latency.
    """
    groups: dict[str, dict[str, list[dict]]] = {
        "by_agent": defaultdict(list), "by_kind": defaultdict(list),
        "by_agent_kind": defaultdict(list), "by_status": defaultdict(list),
    }
    for r in records:
        if r["status"] in exclude_status:
            continue
        groups["by_agent"][r["agent"]].append(r)
        groups["by_kind"][r["kind"]].append(r)
        groups["by_agent_kind"][f"{r['agent']}/{r['kind']}"].append(r)
        groups["by_status"][f"{r['kind']}/{r['status'] or 'none'}"].append(r)
    return {g: {k: _stats(v) for k, v in sorted(rs.items())} for g, rs in groups.items()}


def _round_records(records: list[dict], round_id: int, run: str | None) -> list[dict]:
    rs = [r for r in records if r["round"] == round_id and (run is None or r["run"] == run)]
    if run is None and len({r["run"] for r in rs}) > 1:
        raise ValueError(f"round {round_id} appears in several runs; pass run=")
    return rs


def _agent_rollup(rs: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rs:
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


def round_agent_summary(records: list[dict], round_id: int, run: str | None = None) -> dict[str, dict]:
    """Per-agent roll-up for one round, to embed in that round's decisions.jsonl record.

    ms is the sum of the agent's calls (excluding run_round, which round_time
    counts separately); status is the last non-null status the agent logged.
    run may be omitted only if the log holds a single run for this round.
    """
    return _agent_rollup(_round_records(records, round_id, run))


def round_time_ms(records: list[dict], round_id: int, run: str | None = None) -> float:
    """max(A1, A3) + A2 + tool_time(run_round) + max(A4, A5); missing agents count as 0.

    This models the planned critical path from timed blocks only. It assumes
    A1/A3 and A4/A5 really run in parallel and ignores untimed work between
    calls; compare it with round_span_ms(), the measured wall time.
    """
    rs = _round_records(records, round_id, run)
    agents = _agent_rollup(rs)
    ms = {name: agents.get(name, {}).get("ms", 0.0) for name in ("A1", "A2", "A3", "A4", "A5")}
    run_round = sum(r["ms"] for r in rs if r["step"] == RUN_ROUND_STEP)
    return max(ms["A1"], ms["A3"]) + ms["A2"] + run_round + max(ms["A4"], ms["A5"])


def _parse_ts(s: str) -> datetime:
    # "Z" only parses from Python 3.11. A naive time is taken as UTC, which is
    # what _iso() writes; a hand-made record in local time would shift the span.
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def round_span_ms(records: list[dict], round_id: int, run: str | None = None) -> float | None:
    """Measured wall time of a round: last call end minus first call start.

    Covers real concurrency and gaps between calls, but not work before the
    first or after the last timed call. None when the round has no calls.
    """
    rs = _round_records(records, round_id, run)
    if not rs:
        return None
    start = min(_parse_ts(r["start"]) for r in rs)
    end = max(_parse_ts(r["end"]) for r in rs)
    return (end - start).total_seconds() * 1000.0
