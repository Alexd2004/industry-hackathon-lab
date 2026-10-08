"""Shared pieces for the five agents (Tier 3): model settings, the client, results, checks, one model call.

Every agent follows the same contract (Combined Plan section 7a): a fresh prompt built only from its own
input, a schema-validated output, every number in the output present in the input, and a deterministic
fallback when anything fails. call_model() never raises: each failure comes back as a fallback reason,
so an agent never breaks the round:

    offline          no client (no anthropic package, no credentials, or SOFTSIGNAL_OFFLINE=1)
    timeout          the call took longer than the agent's timeout (no retries: max_retries=0)
    connection       network error (Wi-Fi off)
    api_error        any other API or client error (auth, rate limit, server error, unresolved credentials)
    refusal          stop_reason "refusal" (checked before the text is read, so it is never mislabelled)
    invalid_output   cut off at max_tokens, no text, or text that does not match the schema
    number_not_in_input / cites_unknown_field / age_claim / unsupported_verdict / guardrail   the agents' checks
    insufficient_data   the input lacks what the agent needs; no model call is made

Each reason has a kind (FALLBACK_KINDS, outcome()): SCRIPTED (no call needed), UNAVAILABLE (offline, replay
mismatch), REJECTED (a check refused the reply) or FAILED (no usable reply). Only REJECTED and FAILED count as
fallbacks on screen and in crew.run_fallbacks; the stored status stays FALLBACK for all of them.

Model: claude-haiku-4-5-20251001 for all five agents (cheap and fast; Crew Plan section 9: one setup to measure).
No effort setting is sent by default (not verified for Haiku 4.5); set SOFTSIGNAL_AGENT_EFFORT to send one.
Timeouts: 4 s for live calls (Crew Plan sections 6 and 9); A5's per-round check gets 8 s (it runs after the
round is written, and its p95 was 3.8 s); A2 gets 6 s (its
measured p95 was 3.4 s and its slowest call 3.6 s, too close to 4 s); A4, off the decision path, gets longer
(TIMEOUTS; not measured yet, set after measuring); A3 gets 12 s (every call timed out at 4 s; to be set from its
measured p95); A5's one-off slide pass gets 60 s. Overrides:
SOFTSIGNAL_AGENT_MODEL / SOFTSIGNAL_AGENT_EFFORT / SOFTSIGNAL_AGENT_TIMEOUT_S / SOFTSIGNAL_A2_TIMEOUT_S /
SOFTSIGNAL_A3_TIMEOUT_S /
SOFTSIGNAL_A4_TIMEOUT_S /
SOFTSIGNAL_A5_TIMEOUT_S (A5's per-round check; its slide pass uses a5_audit.SLIDE_TIMEOUT_S). SOFTSIGNAL_OFFLINE=1
forces offline (no client: recorded replay, else the fallbacks), e.g. for a Wi-Fi-off demo with a key set.
The API key lives in the environment only (ANTHROPIC_API_KEY); the repo is public. For local use it can sit in a
`.env` file in the case folder (gitignored), read by load_env_file below; a variable already set wins. Refusals go to the
deterministic fallback, as the plan says, not to a server-side model fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from softsignal.agent_timer import AgentTimer

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_env_file(path: Path = ENV_FILE) -> list[str]:
    """Set KEY=value lines of a .env file into os.environ and return the names set. Variables already set are
    kept, blank lines and # comments are skipped, one pair of quotes around a value is removed. A missing or
    unreadable file sets nothing."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []
    names = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            names.append(key)
    return names


load_env_file()  # before the settings below, so SOFTSIGNAL_* in .env applies too

MODEL = os.environ.get("SOFTSIGNAL_AGENT_MODEL", "claude-haiku-4-5-20251001")
EFFORT = os.environ.get("SOFTSIGNAL_AGENT_EFFORT", "")  # empty: no effort sent
TIMEOUT_S = float(os.environ.get("SOFTSIGNAL_AGENT_TIMEOUT_S", "4.0"))  # decision-path agents (A1-A3)
TIMEOUTS = {"A2": float(os.environ.get("SOFTSIGNAL_A2_TIMEOUT_S", "6.0")),  # measured p95 3.4 s, max 3.6 s: 4 s was tight
            # A3 timed out at 4 s on every call that reached the model (its answer is the largest); not measured
            # yet at this size: read its p95 from tier3_latency.csv after the next recorded run and set it from that
            "A3": float(os.environ.get("SOFTSIGNAL_A3_TIMEOUT_S", "12.0")),
            "A4": float(os.environ.get("SOFTSIGNAL_A4_TIMEOUT_S", "15.0")),  # off the decision path
            # A5 per round runs after the round is written, so 4 s bought nothing (its p95 was 3.8 s); its one-off
            # slide pass uses 60 s
            "A5": float(os.environ.get("SOFTSIGNAL_A5_TIMEOUT_S", "8.0"))}
MAX_RETRIES = 0  # the SDK retries twice by default, which turns a 4 s timeout into about 12 s
MAX_TOKENS = 4096  # room for the short JSON answer (and any thinking, if an effort is set)
MAX_REJECTED_CHARS = 2000  # a rejected reply is kept (truncated) for A5 and prompt tuning

LIVE, FALLBACK, REPLAY = "LIVE", "FALLBACK", "REPLAY"
INSUFFICIENT = "insufficient_data"
OFFLINE, TIMEOUT, CONNECTION, API_ERROR = "offline", "timeout", "connection", "api_error"
REFUSAL, INVALID = "refusal", "invalid_output"
NUMBER_NOT_IN_INPUT, UNKNOWN_FIELD, AGE_CLAIM = "number_not_in_input", "cites_unknown_field", "age_claim"
UNSUPPORTED = "unsupported_verdict"  # e.g. A1 says drift is real without citing any PSI
FORBIDDEN_COLUMN = "forbidden_column"  # A3: the text names a column the model never uses (age, job, ...)
GUARDRAIL = "guardrail"  # A2: an action the hold rule or the promote guard forbids
SCRIPT_ONLY = "script_only"  # A5: only the round's own headline to check, so the script checks it; no model call
REPLAY_MISMATCH = "replay_hash_mismatch"  # offline, and the recording of this round was made from another input
AGENT_ERROR = "agent_error"  # the agent's input or run raised; the round went on without it

# What a FALLBACK came to. The files keep the status LIVE / FALLBACK / REPLAY; the kind is read from the
# fallback_reason. Only REJECTED (a check refused the reply) and FAILED (no usable reply) are the agent failing:
# SCRIPTED needed no model call by design, UNAVAILABLE had no model to call. NOT_RUN: no block or no status.
SCRIPTED, UNAVAILABLE, REJECTED, FAILED, NOT_RUN = "SCRIPTED", "UNAVAILABLE", "REJECTED", "FAILED", "NOT_RUN"
FALLBACK_KINDS = {
    INSUFFICIENT: SCRIPTED, SCRIPT_ONLY: SCRIPTED,
    OFFLINE: UNAVAILABLE, REPLAY_MISMATCH: UNAVAILABLE,
    TIMEOUT: FAILED, CONNECTION: FAILED, API_ERROR: FAILED, REFUSAL: FAILED, AGENT_ERROR: FAILED,
    INVALID: REJECTED, NUMBER_NOT_IN_INPUT: REJECTED, UNKNOWN_FIELD: REJECTED, AGE_CLAIM: REJECTED,
    UNSUPPORTED: REJECTED, FORBIDDEN_COLUMN: REJECTED, GUARDRAIL: REJECTED,
}
REAL_FALLBACKS = (REJECTED, FAILED)
CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID")

# A number: optional sign, digits with an optional decimal part (or a bare decimal), optional %. It must not
# follow a letter, digit, "_" or "." (so "c1", "R1" and "B4155846" are not numbers) nor run into one.
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.]*\d)%?")
# An age stated as a number: "14 years old", "13-17 year-olds", "15 y/o", "aged 15", "age 16". The output is
# a likelihood, never an age (Combined Plan section 3), so no agent may write one.
_AGE = re.compile(r"\b\d{1,2}(?:\s*(?:-|to|or)\s*\d{1,2})?[\s-]*(?:years?|yrs?)[\s-]*olds?\b"
                  r"|\b\d{1,2}\s*y/?o\b|\baged?\s+\d{1,2}\b", re.IGNORECASE)


@dataclass
class AgentResult:
    """One agent's result for one round. block() is its entry in decisions.jsonl (see merge_block).

    rejected: the model's reply when it was not used (failed a check, bad schema, refusal text), truncated;
    kept for A5 and prompt tuning, never shown as the agent's output.
    """

    agent: str
    status: str  # LIVE or FALLBACK (REPLAY is set by replay.py)
    output: Any
    fallback_reason: str | None
    input_hash: str
    errors: list[str] = field(default_factory=list)
    rejected: str | None = None

    def block(self) -> dict:
        return {"status": self.status, "output": self.output, "fallback_reason": self.fallback_reason,
                "input_hash": self.input_hash, "errors": list(self.errors), "rejected": self.rejected}


def trim_text(text: str, limit: int) -> str:
    """text cut to at most limit characters at the last sentence end that fits, keeping at least a third of the
    limit. Only whole sentences are kept, so a qualifier or a negation later in a sentence is never cut off and a
    number is never cut in half. With no such sentence end the text comes back unchanged (too long: the schema then
    rejects it, as before). The result is a prefix of text, so it says nothing the full text did not, and the
    agent's checks still run on it."""
    if len(text) <= limit:
        return text
    # sentence ends judged on the full text, so "3.5" cut after "3." is never taken for one
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text) if m.end() <= limit]
    if ends and ends[-1] >= limit // 3:
        return text[:ends[-1]].rstrip()
    return text


def outcome(block: dict | None) -> str:
    """What an agent block came to: LIVE, REPLAY, NOT_RUN (no block or no status), or for a FALLBACK its kind from
    FALLBACK_KINDS. An unknown fallback reason counts as FAILED, so a new failure is never hidden."""
    if not isinstance(block, dict) or not block.get("status"):
        return NOT_RUN
    if block["status"] != FALLBACK:
        return block["status"]
    return FALLBACK_KINDS.get(block.get("fallback_reason"), FAILED)


def is_real_fallback(block: dict | None) -> bool:
    """The agent itself failed: its reply was refused by a check, or no usable reply came back."""
    return outcome(block) in REAL_FALLBACKS


def merge_block(result: AgentResult, rollup: dict | None = None) -> dict:
    """The decisions.jsonl agent block: agent_timer.round_agent_summary()'s per-agent rollup (ms, tokens,
    call errors) plus the result. The result's status wins; the error lists are joined."""
    rollup = dict(rollup or {})
    out = {**rollup, **result.block()}
    out["errors"] = list(rollup.get("errors", [])) + result.errors
    return out


def input_hash(payload: Any) -> str:
    """Short, stable hash of an agent input, logged so a replay can show it saw the same input."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _strip_thousands(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def number_spans(text: str) -> list[tuple[str, int, int]]:
    """(number as written, start, end) for every number in text (no thousands commas inside a token: "1,000" is
    two tokens here; claims write counts without them)."""
    return [(m.group(), m.start(), m.end()) for m in _NUMBER.finditer(text)]


def number_tokens(text: str) -> list[str]:
    """Every number written in text as written ("92%", "0.955", "1000"), thousands commas removed: the
    precision a claim states matters when it is compared with a file (A5)."""
    return [m.group() for m in _NUMBER.finditer(_strip_thousands(text))]


def numbers_in(text: str) -> list[float]:
    """Every number written in text, as floats (percent signs dropped, thousands commas removed)."""
    return [float(m.group().rstrip("%")) for m in _NUMBER.finditer(_strip_thousands(text))]


def _percent_of(token: str, inputs: list[float]) -> bool:
    """A percent written with at least one decimal ("35.56%", "35.6%") that is an input fraction (strictly between
    0 and 1) x 100 rounded to those decimals. A whole percent ("36%") is not accepted this way: with many fractions in
    an input, almost any whole percent lies within half a point of one of them, so it would let a computed share
    through; 0 and 1 are left out for the same reason (any zero or any count of 1)."""
    if not token.endswith("%"):
        return False
    text = token.rstrip("%").lstrip("+-")
    if "." not in text:
        return False
    decimals = len(text.split(".")[1])
    v = abs(float(text))
    return any(round(x * 100, decimals) == v for x in inputs if 0 < x < 1)


def numbers_not_in_input(output_text: str, allowed_from: Any) -> list[str]:
    """Numbers in an agent's output that do not appear in allowed_from (Combined Plan: checked in code).

    Pass the part of the input the agent may cite, not every number it saw. Compared by absolute value: a
    sign flip is not caught here (A5 and the logs exist for wrong reasoning with real numbers). Numbers are
    compared as written, so 3.2 is not accepted for an input 3.21. A percent with decimals may also be an input
    fraction x 100 at the precision it is written with ("35.56%" or "35.6%" for 0.355555...): a rounding, never new
    arithmetic (see _percent_of for why a whole percent is not).
    """
    inputs = [abs(x) for x in numbers_in(json.dumps(allowed_from, default=str))]
    allowed = set(inputs)
    return sorted({m.group() for m in _NUMBER.finditer(_strip_thousands(output_text))
                   if abs(float(m.group().rstrip("%"))) not in allowed and not _percent_of(m.group(), inputs)})


def age_claims(text: str) -> list[str]:
    """Numeric age statements in an agent's output ("14 years old", "aged 15"): never allowed."""
    return [m.group() for m in _AGE.finditer(text)]


def config_dir() -> Path:
    """Where `ant auth login` keeps profiles: $ANTHROPIC_CONFIG_DIR, else ~/.config/anthropic."""
    return Path(os.environ.get("ANTHROPIC_CONFIG_DIR") or Path.home() / ".config" / "anthropic")


def has_credentials() -> bool:
    """A credential variable is set, or a saved profile exists (credentials/<profile>.json). An empty config
    folder left behind by the CLI does not count: the SDK cannot resolve auth from it."""
    if any(os.environ.get(k) for k in CREDENTIAL_ENV):
        return True
    creds = config_dir() / "credentials"
    return creds.is_dir() and any(creds.glob("*.json"))


def make_client(timeout: float = TIMEOUT_S, max_retries: int = MAX_RETRIES):
    """An Anthropic client for live agent calls, or None (offline: the agents use their fallbacks).

    None when SOFTSIGNAL_OFFLINE is set, the anthropic package is missing, or no credential source is
    configured, so an offline demo never waits on a network call that cannot succeed.
    """
    if os.environ.get("SOFTSIGNAL_OFFLINE") or not has_credentials():
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(timeout=timeout, max_retries=max_retries)


def timeout_for(agent: str) -> float:
    return TIMEOUTS.get(agent, TIMEOUT_S)


@dataclass
class ModelReply:
    """What call_model() got: the output dict, or None with the fallback reason, errors and the raw text."""

    output: dict | None = None
    fallback_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    raw: str | None = None  # the reply text when it was not used
    notes: list[str] = field(default_factory=list)  # what repair() changed in an accepted reply ("TRUNCATED ...")


Check = Callable[[dict], "tuple[str | None, list[str]]"]  # output -> (fallback reason or None, errors)
# The parsed reply -> (reply, notes): applied before the schema. Only for limits the API does not enforce (it passes
# maxItems above 1 and maxLength to the model as hints only), and only by cutting: a repair never adds anything,
# and the schema and the agent's checks still run on what it returns.
Repair = Callable[[dict], "tuple[dict, list[str]]"]


def _error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:300]


def _read_reply(resp, schema: type, check: Check | None, repair: Repair | None = None) -> ModelReply:
    """stop_reason first (a refusal can carry partial text that is not valid JSON), then repair, the schema, check."""
    from pydantic import ValidationError

    text = "".join(getattr(b, "text", "") for b in (resp.content or []) if getattr(b, "type", None) == "text")
    raw = text[:MAX_REJECTED_CHARS] or None
    if resp.stop_reason == "refusal":
        return ModelReply(fallback_reason=REFUSAL, errors=[f"refusal: {getattr(resp, 'stop_details', None)}"], raw=raw)
    if resp.stop_reason != "end_turn":
        return ModelReply(fallback_reason=INVALID, errors=[f"stop_reason {resp.stop_reason}"], raw=raw)
    if not text:
        return ModelReply(fallback_reason=INVALID, errors=["no text in the reply"])
    notes: list[str] = []
    if repair is not None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None  # the schema below reports it
        if isinstance(data, dict):
            data, notes = repair(data)
            text = json.dumps(data)  # validated as JSON, exactly as the reply would have been
    try:
        output = schema.model_validate_json(text).model_dump()
    except ValidationError as e:
        return ModelReply(fallback_reason=INVALID, errors=[f"schema: {str(e)[:300]}"], raw=raw)
    if check is not None:
        reason, errors = check(output)
        if reason is not None:
            return ModelReply(fallback_reason=reason, errors=errors + notes, raw=raw)
    return ModelReply(output=output, notes=notes)


def output_config(schema: type) -> dict:
    """output_config for one call: the JSON schema format, plus effort only when EFFORT is set."""
    import anthropic

    cfg: dict = {"format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)}}
    return {"effort": EFFORT, **cfg} if EFFORT else cfg


def call_model(client, *, agent: str, step: str, system: str, user: str, schema: type, check: Check | None = None,
               timer: AgentTimer | None = None, round_id: Any = None, timeout: float | None = None,
               repair: Repair | None = None) -> ModelReply:
    """One structured-output call (messages.create with output_config.format = the schema), timed if a timer
    is given. Never raises. check (the agent's own rules, e.g. numbers in input) runs inside the timed block,
    so the call's record reads LIVE only when the output was accepted and FALLBACK otherwise.
    """
    def _call(c=None) -> ModelReply:
        try:
            api = client.with_options(timeout=timeout or timeout_for(agent), max_retries=MAX_RETRIES) \
                if hasattr(client, "with_options") else client
            resp = api.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}],
                output_config=output_config(schema))
        except Exception as e:  # noqa: BLE001 - the contract is "never raise"; classify what we can
            return ModelReply(fallback_reason=_classify(e), errors=[_error(e)])
        try:
            if c is not None:
                c.usage(resp)
            return _read_reply(resp, schema, check, repair)
        except Exception as e:  # noqa: BLE001 - a malformed reply object or a failing check: fall back
            return ModelReply(fallback_reason=INVALID, errors=[f"unexpected {_error(e)}"])

    if timer is None:
        return _call()
    rnd = {} if round_id is None else {"round_id": round_id}  # None: the timer's current round
    with timer.call(agent, step, "model", **rnd) as c:
        reply = _call(c)
        c.status = LIVE if reply.fallback_reason is None else FALLBACK
        c.reason = reply.fallback_reason
    return reply


def _classify(e: BaseException) -> str:
    """Fallback reason for an exception from the API call (timeout before connection: it is a subclass)."""
    try:
        import anthropic
    except ImportError:
        return API_ERROR
    if isinstance(e, anthropic.APITimeoutError):
        return TIMEOUT
    if isinstance(e, anthropic.APIConnectionError):
        return CONNECTION
    return API_ERROR  # status errors, unresolved credentials (TypeError in the SDK), anything else
