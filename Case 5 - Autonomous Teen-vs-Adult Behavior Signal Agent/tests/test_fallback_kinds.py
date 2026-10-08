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


# --- the shared pieces behind fewer fallbacks ---------------------------------------------------------------

def test_a_percent_may_round_an_input_fraction_at_its_own_precision():
    from softsignal.agents.base import numbers_not_in_input

    inputs = {"ft": 0.35555555, "rec": 0.7977}
    assert numbers_not_in_input("false-teen 35.56% or 35.6%, recall 79.8%", inputs) == []
    assert numbers_not_in_input("false-teen 35.5%", inputs) == ["35.5%"]  # a wrong rounding is still caught
    assert numbers_not_in_input("about 36%", inputs) == ["36%"]  # a whole percent must be exact, as before
    assert numbers_not_in_input("100.0% of them", {"streak": 1}) == ["100.0%"]  # a count of 1 is not a rate
    assert numbers_not_in_input("0.36 of adults", inputs) == ["0.36"]  # only percents are rounded, as before
    assert numbers_not_in_input("420 adults", {"a": 213, "b": 207}) == ["420"]  # arithmetic is still refused


def test_trim_text_cuts_at_a_sentence_end_or_a_word():
    from softsignal.agents.base import trim_text

    assert trim_text("Short.", 10) == "Short."
    assert trim_text("First sentence here. Second one is long.", 30) == "First sentence here."
    assert trim_text("one two three four five", 12) == "one two three four five"  # no sentence end: unchanged
    assert trim_text("Rate is 3.5 now. More words follow here.", 20) == "Rate is 3.5 now."  # never "Rate is 3."


def test_a5_repair_cuts_long_notes_only():
    from softsignal.agents.a5_audit import repair
    from softsignal.agents.schemas import A5_MAX_NOTE_CHARS

    out, notes = repair({"verdicts": [{"claim_id": "c1", "note": "Fine. " * 60}, {"claim_id": "c2", "note": "Ok."}]})
    assert len(out["verdicts"][0]["note"]) <= A5_MAX_NOTE_CHARS and out["verdicts"][1]["note"] == "Ok."
    assert len(notes) == 1 and notes[0].startswith("TRUNCATED c1 note")


def test_the_evidence_row_carries_what_a2_saw():
    from softsignal.agents.contracts import evidence_source

    record = {"a1": {"output": {"drift": "not_real", "evidence": [{"field": "psi.score", "value": 0.07}]}}}
    src = evidence_source(record, {"min_audit_adults": 120, "cap_margin": 0.0}, 2, {}, audit={"adults": 27, "teens": 33})
    values = src["rows"][0]["values"]
    assert (values["audit_adults"], values["audit_teens"], values["audit_total"]) == (27, 33, 60)
    assert values["psi_score"] == 0.07 and values["min_audit_adults"] == 120
    assert {"margin_max", "window_min", "rule_cap_margin"} <= set(values)
