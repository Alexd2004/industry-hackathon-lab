"""A2 loop controller (Tier 3, step 14): contract, information barrier, checks, clamp, fallbacks, live path.

No test calls the network. Live replies come from a fake client.
"""
import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from softsignal.agent_timer import AgentTimer, load_records, round_agent_summary
from softsignal.agents.a2_controller import (
    GUARDRAIL, SYSTEM, clamp_output, fallback_output, run_a2, user_message, validate_output,
)
from softsignal.agents.base import (
    AGE_CLAIM, API_ERROR, CONNECTION, FALLBACK, INVALID, LIVE, NUMBER_NOT_IN_INPUT, OFFLINE, REFUSAL, TIMEOUT,
    UNKNOWN_FIELD, merge_block,
)
from softsignal.agents.contracts import BarrierError, a2_input, dotted_paths
from softsignal.agents.schemas import A2Output

THRESHOLDS = {"t_verify": 0.81, "t_soft": 0.42, "cap": 0.15, "flags": []}
AUDIT = {"mode": "SHADOW", "streak": 1, "audit_adults": 150, "audit_teens": 140, "round_audit_adults": 30,
         "pooled_adults": 60, "pooled_false_teen": 0.12, "candidate_false_teen": 0.1}
BOUNDS = {"cap_min": 0.08, "cap_max": 0.3, "min_audit_adults": 120, "cap_default": 0.15}
GUARDS = {"hold_required": False, "promote_allowed": False}
RULE = {"action": "re-tune", "cap": 0.15}


def make_payload(**over):
    kw = dict(round_id=3, thresholds=THRESHOLDS, audit=AUDIT, bounds=BOUNDS, guards=GUARDS, rule=RULE)
    kw |= over
    return a2_input(**kw)


@pytest.fixture
def payload():
    return make_payload()


def good(**over) -> dict:
    out = {"action": "re-tune", "cap": 0.15, "reason": "150 audit adults, above the floor of 120: re-tune.",
           "cites": ["audit.audit_adults", "bounds.min_audit_adults"]}
    return out | over


class FakeClient:
    def __init__(self, reply=None, raises=None):
        self.calls, self.options, self.reply, self.raises = [], [], reply, raises
        self.messages = SimpleNamespace(create=self._create)

    def with_options(self, **kw):
        self.options.append(kw)
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        if self.raises is not None:
            raise self.raises
        return self.reply


def reply(output=None, stop_reason="end_turn", text=None):
    body = text if text is not None else (json.dumps(output) if output is not None else "")
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None,
                           content=[SimpleNamespace(type="text", text=body)] if body else [],
                           usage=SimpleNamespace(input_tokens=900, output_tokens=60, cache_creation_input_tokens=None,
                                                 cache_read_input_tokens=None))


# --- contract and information barrier -------------------------------------------------------------

def test_input_has_only_the_contract_fields(payload):
    assert set(payload) == {"agent", "round", "a1", "a3", "thresholds", "audit", "bounds", "guards", "rule"}
    assert payload["a1"] == payload["a3"] == "insufficient_data"  # absent agents are marked, never guessed


def test_extra_caller_fields_are_not_copied():
    audit = AUDIT | {"label_teen": 1, "recall": 0.9, "secret": "x"}
    p = make_payload(audit=audit, rule=RULE | {"extra": 1})
    text = json.dumps(p)
    for key in ("label_teen", "recall", "secret", "extra"):
        assert key not in text


def test_missing_field_is_an_error():
    with pytest.raises(ValueError, match="missing"):
        make_payload(audit={k: v for k, v in AUDIT.items() if k != "streak"})


def test_bad_rule_action_is_an_error():
    with pytest.raises(ValueError, match="rule action"):
        make_payload(rule={"action": "ban", "cap": 0.15})


@pytest.mark.parametrize("key", ["prec", "rec", "ft", "mt", "f1", "auc", "label_teen", "age", "job"])
def test_agent_output_carrying_a_forbidden_key_is_refused(key):
    with pytest.raises(BarrierError):
        make_payload(a1={"drift": "real", key: 0.5})
    with pytest.raises(BarrierError):
        make_payload(a3={"patterns": [{"description": "x", key: 1}]})


def test_non_json_input_is_refused():
    with pytest.raises(TypeError):
        make_payload(audit=AUDIT | {"streak": object()})


def test_a1_and_a3_output_passes_through():
    p = make_payload(a1={"drift": "not_real", "reason": "psi 0.04"}, a3="insufficient_data")
    assert p["a1"]["drift"] == "not_real" and p["a3"] == "insufficient_data"


def test_dotted_paths_name_nodes_and_leaves(payload):
    paths = dotted_paths(payload)
    assert {"audit", "audit.audit_adults", "bounds.cap_max", "rule.action", "guards.hold_required"} <= paths
    assert "audit.nope" not in paths


# --- schema ---------------------------------------------------------------------------------------

def test_schema_has_action_and_cap_only():
    assert set(A2Output.model_fields) == {"action", "cap", "reason", "cites"}  # no blend_w, no cutoff


@pytest.mark.parametrize("bad", [
    {"action": "ban"}, {"cap": "high"}, {"reason": ""}, {"cites": []}, {"reason": "x" * 401},
    {"cites": list("abcdef")}, {"blend_w": 0.5},
])
def test_schema_rejects(bad):
    with pytest.raises(ValidationError):
        A2Output(**(good() | bad))


def test_schema_accepts_an_out_of_range_cap():
    assert A2Output(**good(cap=0.9)).cap == 0.9  # clamped in code, not rejected


# --- checks ---------------------------------------------------------------------------------------

def test_valid_output_passes(payload):
    assert validate_output(good(), payload) == (None, [])


@pytest.mark.parametrize("change,reason", [
    ({"reason": "A 14 years old teen is likely."}, AGE_CLAIM),
    ({"cites": ["audit.nothing"]}, UNKNOWN_FIELD),
    ({"cites": ["audit.audit_adults", "audit.audit_adults"]}, UNKNOWN_FIELD),
    ({"reason": "237 audit adults, so re-tune."}, NUMBER_NOT_IN_INPUT),
    ({"reason": "Cap 0.5 would be better."}, NUMBER_NOT_IN_INPUT),
])
def test_bad_outputs_are_rejected(payload, change, reason):
    got, errors = validate_output(good(**change), payload)
    assert got == reason and errors


def test_hold_rule_is_enforced_in_code():
    p = make_payload(guards=GUARDS | {"hold_required": True}, audit=AUDIT | {"audit_adults": 90})
    assert validate_output(good(action="re-tune", reason="90 audit adults."), p)[0] == GUARDRAIL
    assert validate_output(good(action="promote", reason="90 audit adults."), p)[0] == GUARDRAIL
    assert validate_output(good(action="hold", reason="90 audit adults."), p) == (None, [])


def test_promote_needs_the_loops_pooled_test():
    assert validate_output(good(action="promote"), make_payload())[0] == GUARDRAIL
    p = make_payload(guards=GUARDS | {"promote_allowed": True})
    assert validate_output(good(action="promote"), p) == (None, [])


def test_hold_is_allowed_after_the_floor(payload):
    assert validate_output(good(action="hold"), payload) == (None, [])  # A2 may be more careful than the rule


@pytest.mark.parametrize("cap,want", [(0.15, 0.15), (0.5, 0.3), (0.01, 0.08), (0.3, 0.3), (0.08, 0.08)])
def test_cap_is_clamped_to_bounds(payload, cap, want):
    out, notes = clamp_output(good(cap=cap), payload)
    assert out["cap"] == want and bool(notes) == (cap != want)
    assert all(n.startswith("CLAMPED") for n in notes)


# --- fallback -------------------------------------------------------------------------------------

@pytest.mark.parametrize("action", ["hold", "re-tune", "promote"])
def test_fallback_is_the_rule_decision_and_passes_its_own_checks(action):
    p = make_payload(rule={"action": action, "cap": 0.2}, guards=GUARDS | {"promote_allowed": action == "promote"})
    out = fallback_output(p)
    assert (out["action"], out["cap"]) == (action, 0.2)
    A2Output(**out)
    assert validate_output(out, p) == (None, [])


# --- live path and failures -----------------------------------------------------------------------

def test_live_reply_is_used(payload, tmp_path):
    timer = AgentTimer(path=tmp_path / "calls.jsonl", run="t")
    r = run_a2(payload, client=FakeClient(reply(good())), timer=timer, round_id=3)
    assert r.status == LIVE and r.fallback_reason is None and r.errors == []
    assert r.output == good()
    assert round_agent_summary(load_records(timer.path), 3, "t")["A2"]["status"] == LIVE


def test_an_out_of_range_cap_is_clamped_and_logged_not_a_fallback(payload):
    r = run_a2(payload, client=FakeClient(reply(good(cap=0.5))))
    assert r.status == LIVE and r.output["cap"] == 0.3
    assert r.errors == ["CLAMPED cap 0.5 -> 0.3"]


def test_the_request_is_one_fresh_prompt(payload):
    c = FakeClient(reply(good()))
    run_a2(payload, client=c)
    (call,) = c.calls
    assert call["system"] == SYSTEM and len(call["messages"]) == 1
    assert call["messages"][0]["content"] == user_message(payload)
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert c.options and c.options[0]["max_retries"] == 0


def test_prompt_has_no_test_metric_or_label_names(payload):
    text = user_message(payload)
    for word in ("label_teen", "is_teen", "account_age_days", "friend_count"):
        assert word not in text


def test_offline_uses_the_rule_without_a_call(payload, tmp_path):
    timer = AgentTimer(path=tmp_path / "calls.jsonl", run="t")
    r = run_a2(payload, client=None, timer=timer, round_id=3)
    assert r.status == FALLBACK and r.fallback_reason == OFFLINE
    assert (r.output["action"], r.output["cap"]) == ("re-tune", 0.15)
    assert round_agent_summary(load_records(timer.path), 3, "t")["A2"]["status"] == FALLBACK


def refusal():
    return reply(text="no", stop_reason="refusal")


@pytest.mark.parametrize("make,reason", [
    (lambda: FakeClient(reply(text="not json")), INVALID),
    (lambda: FakeClient(reply(good(action="ban"))), INVALID),
    (lambda: FakeClient(reply(good(blend_w=0.5))), INVALID),
    (lambda: FakeClient(reply(good(), stop_reason="max_tokens")), INVALID),
    (lambda: FakeClient(refusal()), REFUSAL),
    (lambda: FakeClient(reply(good(reason="237 audit adults."))), NUMBER_NOT_IN_INPUT),
    (lambda: FakeClient(reply(good(cites=["made.up"]))), UNKNOWN_FIELD),
    (lambda: FakeClient(reply(good(action="promote"))), GUARDRAIL),
])
def test_every_failure_falls_back_to_the_rule(payload, make, reason):
    r = run_a2(payload, client=make())
    assert r.status == FALLBACK and r.fallback_reason == reason
    assert (r.output["action"], r.output["cap"]) == ("re-tune", 0.15)


def test_api_errors_fall_back(payload):
    import anthropic
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    for exc, want in ((anthropic.APITimeoutError(request=req), TIMEOUT),
                      (anthropic.APIConnectionError(request=req), CONNECTION),
                      (RuntimeError("boom"), API_ERROR)):
        r = run_a2(payload, client=FakeClient(raises=exc))
        assert r.status == FALLBACK and r.fallback_reason == want


def test_a_rejected_reply_is_kept_but_not_used(payload):
    r = run_a2(payload, client=FakeClient(reply(good(action="promote"))))
    assert "promote" in (r.rejected or "") and r.output["action"] == "re-tune"


def test_block_is_a_decisions_record(payload):
    r = run_a2(payload, client=FakeClient(reply(good())))
    b = merge_block(r, {"ms": 12.0})
    assert b["status"] == LIVE and b["output"]["action"] == "re-tune" and b["ms"] == 12.0
    json.dumps(b)


def test_input_is_not_mutated(payload):
    before = copy.deepcopy(payload)
    run_a2(payload, client=FakeClient(reply(good())))
    assert payload == before
