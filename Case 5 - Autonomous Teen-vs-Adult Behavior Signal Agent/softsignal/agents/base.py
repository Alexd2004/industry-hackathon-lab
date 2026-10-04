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
    number_not_in_input / cites_unknown_field / age_claim / guardrail   the agent's own checks (validate in each agent)
    insufficient_data   the input lacks what the agent needs; no model call is made

Model: claude-opus-5 for all five agents (Crew Plan section 9: one setup to measure), effort "low".
Timeouts: 4 s for agents on the decision path (Crew Plan section 6); A4 and A5 run after the round, off the
path, so they get longer (TIMEOUTS; not measured yet, set after measuring). Overrides:
SOFTSIGNAL_AGENT_MODEL / SOFTSIGNAL_AGENT_EFFORT / SOFTSIGNAL_AGENT_TIMEOUT_S / SOFTSIGNAL_A4_TIMEOUT_S.
The API key lives in the environment only (ANTHROPIC_API_KEY); the repo is public. Refusals go to the
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

MODEL = os.environ.get("SOFTSIGNAL_AGENT_MODEL", "claude-opus-5")
EFFORT = os.environ.get("SOFTSIGNAL_AGENT_EFFORT", "low")
TIMEOUT_S = float(os.environ.get("SOFTSIGNAL_AGENT_TIMEOUT_S", "4.0"))  # decision-path agents (A1-A3)
TIMEOUTS = {"A4": float(os.environ.get("SOFTSIGNAL_A4_TIMEOUT_S", "15.0"))}  # off the decision path
MAX_RETRIES = 0  # the SDK retries twice by default, which turns a 4 s timeout into about 12 s
MAX_TOKENS = 4096  # adaptive thinking at low effort shares this with the short JSON answer
MAX_REJECTED_CHARS = 2000  # a rejected reply is kept (truncated) for A5 and prompt tuning

LIVE, FALLBACK, REPLAY = "LIVE", "FALLBACK", "REPLAY"
INSUFFICIENT = "insufficient_data"
OFFLINE, TIMEOUT, CONNECTION, API_ERROR = "offline", "timeout", "connection", "api_error"
REFUSAL, INVALID = "refusal", "invalid_output"
NUMBER_NOT_IN_INPUT, UNKNOWN_FIELD, AGE_CLAIM = "number_not_in_input", "cites_unknown_field", "age_claim"
GUARDRAIL = "guardrail"  # A2: an action the hold rule or the promote guard forbids
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


def numbers_in(text: str) -> list[float]:
    """Every number written in text, as floats (percent signs dropped, thousands commas removed)."""
    return [float(m.group().rstrip("%")) for m in _NUMBER.finditer(_strip_thousands(text))]


def numbers_not_in_input(output_text: str, allowed_from: Any) -> list[str]:
    """Numbers in an agent's output that do not appear in allowed_from (Combined Plan: checked in code).

    Pass the part of the input the agent may cite, not every number it saw. Compared by absolute value: a
    sign flip is not caught here (A5 and the logs exist for wrong reasoning with real numbers). Numbers are
    compared as written, so 3.2 is not accepted for an input 3.21.
    """
    allowed = {abs(x) for x in numbers_in(json.dumps(allowed_from, default=str))}
    return sorted({m.group() for m in _NUMBER.finditer(_strip_thousands(output_text))
                   if abs(float(m.group().rstrip("%"))) not in allowed})


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


Check = Callable[[dict], "tuple[str | None, list[str]]"]  # output -> (fallback reason or None, errors)


def _error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:300]


def _read_reply(resp, schema: type, check: Check | None) -> ModelReply:
    """stop_reason first (a refusal can carry partial text that is not valid JSON), then the schema, then check."""
    from pydantic import ValidationError

    text = "".join(getattr(b, "text", "") for b in (resp.content or []) if getattr(b, "type", None) == "text")
    raw = text[:MAX_REJECTED_CHARS] or None
    if resp.stop_reason == "refusal":
        return ModelReply(fallback_reason=REFUSAL, errors=[f"refusal: {getattr(resp, 'stop_details', None)}"], raw=raw)
    if resp.stop_reason != "end_turn":
        return ModelReply(fallback_reason=INVALID, errors=[f"stop_reason {resp.stop_reason}"], raw=raw)
    if not text:
        return ModelReply(fallback_reason=INVALID, errors=["no text in the reply"])
    try:
        output = schema.model_validate_json(text).model_dump()
    except ValidationError as e:
        return ModelReply(fallback_reason=INVALID, errors=[f"schema: {str(e)[:300]}"], raw=raw)
    if check is not None:
        reason, errors = check(output)
        if reason is not None:
            return ModelReply(fallback_reason=reason, errors=errors, raw=raw)
    return ModelReply(output=output)


def call_model(client, *, agent: str, step: str, system: str, user: str, schema: type, check: Check | None = None,
               timer: AgentTimer | None = None, round_id: Any = None, timeout: float | None = None) -> ModelReply:
    """One structured-output call (messages.create with output_config.format = the schema), timed if a timer
    is given. Never raises. check (the agent's own rules, e.g. numbers in input) runs inside the timed block,
    so the call's record reads LIVE only when the output was accepted and FALLBACK otherwise.
    """
    def _call(c=None) -> ModelReply:
        try:
            import anthropic

            api = client.with_options(timeout=timeout or timeout_for(agent), max_retries=MAX_RETRIES) \
                if hasattr(client, "with_options") else client
            resp = api.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}],
                output_config={"effort": EFFORT, "format": {"type": "json_schema",
                                                             "schema": anthropic.transform_schema(schema)}})
        except Exception as e:  # noqa: BLE001 - the contract is "never raise"; classify what we can
            return ModelReply(fallback_reason=_classify(e), errors=[_error(e)])
        try:
            if c is not None:
                c.usage(resp)
            return _read_reply(resp, schema, check)
        except Exception as e:  # noqa: BLE001 - a malformed reply object or a failing check: fall back
            return ModelReply(fallback_reason=INVALID, errors=[f"unexpected {_error(e)}"])

    if timer is None:
        return _call()
    rnd = {} if round_id is None else {"round_id": round_id}  # None: the timer's current round
    with timer.call(agent, step, "model", **rnd) as c:
        reply = _call(c)
        c.status = LIVE if reply.fallback_reason is None else FALLBACK
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
