"""What a FALLBACK came to (agents/base.py outcome): only a refused reply or no usable reply counts as the agent
failing. No network, no model call."""
import json

import pytest

from softsignal import crew
from softsignal.agent_timer import AgentTimer, load_records
from softsignal.agents import a3_errors
from softsignal.agents.base import (
    FAILED, NOT_RUN, REJECTED, SCRIPTED, UNAVAILABLE, is_real_fallback, outcome,
)


def block(status, reason=None):
    return {"status": status, "fallback_reason": reason}


@pytest.mark.parametrize("reason, kind", [
    ("insufficient_data", SCRIPTED), ("script_only", SCRIPTED),
    ("offline", UNAVAILABLE), ("replay_hash_mismatch", UNAVAILABLE),
    ("timeout", FAILED), ("connection", FAILED), ("api_error", FAILED), ("refusal", FAILED), ("agent_error", FAILED),
    ("invalid_output", REJECTED), ("number_not_in_input", REJECTED), ("cites_unknown_field", REJECTED),
    ("age_claim", REJECTED), ("unsupported_verdict", REJECTED), ("forbidden_column", REJECTED),
    ("guardrail", REJECTED),
])
def test_every_fallback_reason_has_a_kind(reason, kind):
    assert outcome(block("FALLBACK", reason)) == kind
    assert is_real_fallback(block("FALLBACK", reason)) == (kind in (REJECTED, FAILED))


def test_live_replay_and_missing_blocks():
    assert outcome(block("LIVE")) == "LIVE" and outcome(block("REPLAY")) == "REPLAY"
    assert outcome(None) == NOT_RUN and outcome(block(None)) == NOT_RUN
    assert not is_real_fallback(block("LIVE")) and not is_real_fallback(None)


def test_an_unknown_reason_is_never_hidden():
    assert outcome(block("FALLBACK", "something_new")) == FAILED
    assert outcome(block("FALLBACK", None)) == FAILED


def test_run_fallbacks_counts_only_real_ones():
    recs = [{"run": "r", "round": 0, "a1": block("FALLBACK", "insufficient_data"), "a5": block("FALLBACK", "script_only")},
            {"run": "r", "round": 1, "a1": block("FALLBACK", "offline"), "a2": block("FALLBACK", "cites_unknown_field"),
             "a3": block("FALLBACK", "timeout"), "a4": block(None)}]
    assert crew.run_fallbacks(recs) == 2


def test_the_timer_logs_the_reason_only_when_set(tmp_path):
    t = AgentTimer(tmp_path / "calls.jsonl")
    with t.call("A1", "fallback", "tool", status="FALLBACK", reason="offline", round_id=1):
        pass
    with t.call("A1", "drift", "model", round_id=1) as c:
        c.status = "LIVE"
    first, second = load_records(tmp_path / "calls.jsonl")
    assert first["reason"] == "offline" and "reason" not in second


def test_a3_keeps_why_it_had_nothing_to_analyse():
    payload = {"round": 0, "audit": {"adults": 0, "teens": 0}}
    got = a3_errors.run_a3(payload)
    assert got.fallback_reason == "insufficient_data"
    assert got.errors == ["no audit labels revealed in earlier rounds"]
    json.dumps(got.block())
