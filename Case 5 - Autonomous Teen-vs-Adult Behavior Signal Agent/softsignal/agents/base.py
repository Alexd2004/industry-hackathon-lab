"""Shared pieces for the five agents (Tier 3): model settings, the client, results, checks, one model call.

Every agent follows the same contract (Combined Plan section 7a): a fresh prompt built only from its own
input, a schema-validated output, every number in the output present in the input, and a deterministic
fallback when anything fails. call_model() turns each failure into a fallback reason instead of raising,
so an agent never breaks the round:

    offline          no client (no anthropic package, no credentials, or SOFTSIGNAL_OFFLINE=1)
    timeout          the call took longer than TIMEOUT_S (no retries: max_retries=0, Crew Plan section 6)
    connection       network error (Wi-Fi off)
    api_error        any other API error (auth, rate limit, server error, bad request)
    refusal          stop_reason "refusal" (checked before reading the output)
    invalid_output   cut off at max_tokens, or the output does not match the schema
    number_not_in_input / cites_unknown_field   the agent's own checks (see validate in each agent)
    insufficient_data   the input lacks what the agent needs; no model call is made

Model: claude-opus-5 for all five agents (Crew Plan section 9: one setup to measure), effort "low" to fit
the 4 s per-call budget. Override with SOFTSIGNAL_AGENT_MODEL / SOFTSIGNAL_AGENT_EFFORT /
SOFTSIGNAL_AGENT_TIMEOUT_S. The API key lives in the environment only (ANTHROPIC_API_KEY); the repo is public.
Refusals go to the deterministic fallback, as the plan says, not to a server-side model fallback: a
second model run would not fit the 4 s budget.
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
TIMEOUT_S = float(os.environ.get("SOFTSIGNAL_AGENT_TIMEOUT_S", "4.0"))
MAX_RETRIES = 0  # the SDK retries twice by default, which turns a 4 s timeout into about 12 s
MAX_TOKENS = 4096  # adaptive thinking at low effort shares this with the short JSON answer

LIVE, FALLBACK, REPLAY = "LIVE", "FALLBACK", "REPLAY"
INSUFFICIENT = "insufficient_data"
OFFLINE, TIMEOUT, CONNECTION, API_ERROR = "offline", "timeout", "connection", "api_error"
REFUSAL, INVALID = "refusal", "invalid_output"
NUMBER_NOT_IN_INPUT, UNKNOWN_FIELD = "number_not_in_input", "cites_unknown_field"
CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID")
PROFILE_DIR = Path.home() / ".config" / "anthropic"  # where `ant auth login` keeps a profile

# A number: optional sign, digits with an optional decimal part (or a bare decimal), optional %. It must not
# follow a letter, digit, "_" or "." (so "c1", "R1" and "B4155846" are not numbers) nor run into one.
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.]*\d)%?")


@dataclass
class AgentResult:
    """One agent's result for one round. block() is its entry in decisions.jsonl (see merge_block)."""

    agent: str
    status: str  # LIVE or FALLBACK (REPLAY is set by replay.py)
    output: Any
    fallback_reason: str | None
    input_hash: str
    errors: list[str] = field(default_factory=list)

    def block(self) -> dict:
        return {"status": self.status, "output": self.output, "fallback_reason": self.fallback_reason,
                "input_hash": self.input_hash, "errors": list(self.errors)}


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


def numbers_in(text: str) -> list[float]:
    """Every number written in text, as floats (percent signs dropped, thousands commas removed)."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return [float(m.group().rstrip("%")) for m in _NUMBER.finditer(text)]


def numbers_not_in_input(output_text: str, payload: Any) -> list[str]:
    """Numbers in an agent's output that do not appear in its input (Combined Plan: checked in code).

    Compared by absolute value: a sign flip is not caught here (A5 and the logs exist for wrong reasoning
    with real numbers). Numbers are compared as written, so 3.2 is not accepted for an input 3.21.
    """
    allowed = {abs(x) for x in numbers_in(json.dumps(payload, default=str))}
    return sorted({m.group() for m in _NUMBER.finditer(re.sub(r"(?<=\d),(?=\d{3}\b)", "", output_text))
                   if abs(float(m.group().rstrip("%"))) not in allowed})


def has_credentials() -> bool:
    return any(os.environ.get(k) for k in CREDENTIAL_ENV) or PROFILE_DIR.is_dir()


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


@dataclass
class ModelReply:
    """What call_model() got: the parsed output as a dict, or None with the fallback reason and errors."""

    output: dict | None = None
    fallback_reason: str | None = None
    errors: list[str] = field(default_factory=list)


Check = Callable[[dict], "tuple[str | None, list[str]]"]  # output -> (fallback reason or None, errors)


def call_model(client, *, agent: str, step: str, system: str, user: str, schema: type, check: Check | None = None,
               timer: AgentTimer | None = None, round_id: Any = None) -> ModelReply:
    """One structured-output call (client.messages.parse with a Pydantic schema), timed if a timer is given.

    Never raises for API, network, refusal or schema problems: they come back as a fallback reason. check
    (the agent's own rules, e.g. numbers in input) runs inside the timed block, so the call's record reads
    LIVE only when the output was accepted and FALLBACK otherwise.
    """
    import anthropic  # only reached with a live client
    from pydantic import ValidationError

    kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS, system=system,
                  messages=[{"role": "user", "content": user}], output_format=schema,
                  output_config={"effort": EFFORT})

    def _call(c=None) -> ModelReply:
        try:
            resp = client.messages.parse(**kwargs)
        except anthropic.APITimeoutError as e:  # a subclass of APIConnectionError: catch it first
            return ModelReply(fallback_reason=TIMEOUT, errors=[f"{type(e).__name__}: {e}"])
        except anthropic.APIConnectionError as e:
            return ModelReply(fallback_reason=CONNECTION, errors=[f"{type(e).__name__}: {e}"])
        except anthropic.APIStatusError as e:
            return ModelReply(fallback_reason=API_ERROR, errors=[f"{type(e).__name__} {e.status_code}: {e.message}"])
        except anthropic.AnthropicError as e:  # credentials and other client-side SDK errors
            return ModelReply(fallback_reason=API_ERROR, errors=[f"{type(e).__name__}: {e}"])
        except (ValidationError, ValueError) as e:  # the reply did not parse into the schema
            return ModelReply(fallback_reason=INVALID, errors=[f"{type(e).__name__}: {str(e)[:300]}"])
        if c is not None:
            c.usage(resp)
        if resp.stop_reason == "refusal":
            return ModelReply(fallback_reason=REFUSAL, errors=[f"refusal: {getattr(resp, 'stop_details', None)}"])
        if resp.stop_reason != "end_turn":
            return ModelReply(fallback_reason=INVALID, errors=[f"stop_reason {resp.stop_reason}"])
        parsed = getattr(resp, "parsed_output", None)
        try:  # re-validate: a reply object is not trusted to have been checked against the schema
            output = schema.model_validate(parsed.model_dump() if hasattr(parsed, "model_dump") else parsed).model_dump()
        except ValidationError as e:
            return ModelReply(fallback_reason=INVALID, errors=[f"schema: {str(e)[:300]}"])
        if check is not None:
            reason, errors = check(output)
            if reason is not None:
                return ModelReply(output=output, fallback_reason=reason, errors=errors)
        return ModelReply(output=output)

    if timer is None:
        return _call()
    rnd = {} if round_id is None else {"round_id": round_id}  # None: the timer's current round
    with timer.call(agent, step, "model", **rnd) as c:
        reply = _call(c)
        c.status = LIVE if reply.fallback_reason is None else FALLBACK
    return reply
