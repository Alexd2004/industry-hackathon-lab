"""Step 15: the A2 against rule diff and format_diff (pure functions, no model, no loop)."""
import pytest

from softsignal import side_by_side
from softsignal.agents.base import FALLBACK, LIVE, REPLAY
from softsignal.side_by_side import diff, diff_count, format_diff
from softsignal.ui_loop import DIFF_ROWS, diff_table, valid_decision

RULE = {"action": "hold", "cap": 0.15}


def block(status=LIVE, **out) -> dict:
    output = {"action": "hold", "cap": 0.15, "reason": "Only 57 audit adults.", "cites": ["audit"]} | out
    return {"status": status, "output": output, "fallback_reason": None}


def test_fields_match_the_loop_tab_rows():
    assert side_by_side.DIFF_FIELDS == DIFF_ROWS


def test_same_decision_is_an_empty_diff():
    assert diff(block(), RULE) == {} and diff_count(block(), RULE) == 0


def test_differing_fields_are_listed_as_rule_then_a2():
    assert diff(block(action="re-tune", cap=0.10), RULE) == {"action": ["hold", "re-tune"], "cap": [0.15, 0.10]}
    assert diff_count(block(action="re-tune", cap=0.10), RULE) == 2


def test_float_noise_is_not_a_difference():
    assert diff(block(cap=0.15000000000000002), RULE) == {}


@pytest.mark.parametrize("a2_block", [
    None, {}, {"status": None, "output": None, "fallback_reason": None},
    {"status": FALLBACK, "output": {"action": "re-tune", "cap": 0.3}, "fallback_reason": "offline"},
    {"status": LIVE, "output": None, "fallback_reason": None},
    {"status": LIVE, "output": "insufficient_data", "fallback_reason": None},
])
def test_no_decision_of_its_own_means_an_empty_diff(a2_block):
    assert diff(a2_block, RULE) == {}


def test_replay_is_compared_like_live():
    assert diff(block(status=REPLAY, action="promote"), RULE) == {"action": ["hold", "promote"]}


def test_a_field_one_side_does_not_state_is_skipped():
    rule = {"action": "hold", "cap": 0.15, "cutoff": 0.5}
    assert diff(block(blend_w=None), rule) == {}  # A2 has no cutoff or blend_w: nothing to compare
    assert diff(block(cutoff=0.55), rule) == {"cutoff": [0.5, 0.55]}


def test_bool_is_not_compared_as_a_number():
    assert diff(block(cap=True), {"action": "hold", "cap": 1}) == {"cap": [1, True]}


def test_diff_does_not_change_its_inputs():
    b, r = block(action="re-tune"), dict(RULE)
    before = (str(b), str(r))
    diff(b, r)
    assert (str(b), str(r)) == before


def record(a2, rule=None, source="A2") -> dict:
    rule = rule or RULE
    return {"run": "x", "round": 5, "a2": a2, "rule_decision": rule, "diff": diff(a2, rule),
            "applied": {"decision": rule, "source": source}}


def test_format_when_they_differ():
    text = format_diff(record(block(action="re-tune", cap=0.10, reason="Audit false-teen is high."), source="A2"))
    assert text == ("A2 decided action re-tune, cap 10% because Audit false-teen is high. "
                    "The rule would have decided action hold, cap 15% (differs on cap, action). Applied: A2.")


def test_format_when_they_agree():
    text = format_diff(record(block(), source="rule"))
    assert "A2 decided action hold, cap 15% because Only 57 audit adults." in text
    assert "The rule would have decided the same." in text and text.endswith("Applied: rule.")


def test_format_on_fallback_says_why_and_has_no_diff():
    fb = {"status": FALLBACK, "output": {"action": "hold", "cap": 0.15}, "fallback_reason": "timeout"}
    rec = record(fb, source="rule")
    assert rec["diff"] == {}
    assert format_diff(rec) == "A2 fell back (timeout), so the rule decided: action hold, cap 15%."


@pytest.mark.parametrize("a2", [None, {}, {"status": None, "output": None, "fallback_reason": None}])
def test_format_when_a2_did_not_run(a2):
    assert format_diff(record(a2)) == "A2 did not decide this round, the rule decided: action hold, cap 15%."


def test_format_never_raises_on_a_partial_record():
    for rec in ({}, {"a2": "x"}, {"rule_decision": None, "a2": block()}, {"a2": block(reason=None)}):
        assert isinstance(format_diff(rec), str)


def test_no_em_dash_in_the_sentence():
    assert "—" not in format_diff(record(block(action="promote")))


def test_a_record_with_the_diff_still_passes_the_loop_tab_checks():
    rec = record(block(action="re-tune", cap=0.10))
    rec.update({k: {"status": None, "output": None, "fallback_reason": None} for k in ("a1", "a3", "a4", "a5")})
    assert valid_decision(rec)
    table = diff_table(rec).set_index("field")
    assert table.loc["action", "changed"] == "YES" and table.loc["cap", "changed"] == "YES"
