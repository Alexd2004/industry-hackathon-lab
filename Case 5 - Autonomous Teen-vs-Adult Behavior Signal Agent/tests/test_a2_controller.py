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
    SYSTEM, clamp_output, fallback_output, guarded_action, percent_forms, run_a2, user_message, validate_output,
)
from softsignal.agents.base import (
    AGE_CLAIM, API_ERROR, CONNECTION, FALLBACK, GUARDRAIL, INVALID, LIVE, NUMBER_NOT_IN_INPUT, OFFLINE, REFUSAL,
    TIMEOUT, UNKNOWN_FIELD, merge_block,
)
from softsignal.agents.contracts import BarrierError, a2_input, dotted_paths, hard_limits
from softsignal.agents.schemas import A2Output

THRESHOLDS = {"t_verify": 0.81, "t_soft": 0.42, "cap": 0.15, "flags": []}
AUDIT = {"mode": "SHADOW", "streak": 1, "audit_adults": 150, "audit_teens": 140, "round_audit_adults": 30,
         "pooled_adults": 60, "pooled_false_teen_rate": 0.12, "candidate_false_teen": 0.1}
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


@pytest.mark.parametrize("over,match", [
    ({"bounds": BOUNDS | {"cap_min": 0.4}}, "cap_min <= cap_max"),
    ({"bounds": BOUNDS | {"cap_max": 1.5}}, "cap_min <= cap_max"),
    ({"bounds": BOUNDS | {"cap_min": 0.0, "cap_max": 1.0}}, "looser than the code limits"),
    ({"bounds": BOUNDS | {"cap_max": 0.5}}, "looser than the code limits"),
    ({"bounds": BOUNDS | {"cap_min": 0.05}}, "looser than the code limits"),
    ({"bounds": BOUNDS | {"min_audit_adults": 0}}, "looser than the code limits"),
    ({"bounds": BOUNDS | {"min_audit_adults": 119}}, "looser than the code limits"),
    ({"audit": AUDIT | {"mode": "shadow"}}, "mode must be"),
    ({"guards": GUARDS | {"hold_required": True}, "audit": AUDIT | {"audit_adults": 10}},
     "breaks guards.hold_required"),
    ({"rule": RULE | {"action": "promote"}}, "breaks guards.promote_allowed"),
    ({"rule": RULE | {"cap": 0.5}}, "outside the bounds"),
    ({"rule": RULE | {"cap": 0.01}}, "outside the bounds"),
    ({"guards": GUARDS | {"hold_required": True}, "rule": {"action": "hold", "cap": 0.15}},
     "hold_required contradicts"),
    ({"audit": AUDIT | {"audit_adults": 10}}, "hold_required contradicts"),
    ({"guards": GUARDS | {"promote_allowed": True}, "audit": AUDIT | {"mode": "ACTIVE"}}, "promote_allowed needs"),
    ({"guards": GUARDS | {"promote_allowed": True, "hold_required": True}, "audit": AUDIT | {"audit_adults": 10}},
     "promote_allowed needs"),
])
def test_inconsistent_input_is_refused(over, match):
    with pytest.raises(ValueError, match=match):
        make_payload(**over)


def test_a_stricter_floor_or_narrower_cap_is_allowed():
    p = make_payload(bounds=BOUNDS | {"cap_min": 0.1, "cap_max": 0.25, "min_audit_adults": 150},
                     audit=AUDIT | {"audit_adults": 150})
    assert p["bounds"]["cap_max"] == 0.25


def test_hard_limits_are_the_loop_and_policy_constants():
    from softsignal.loop import CAP_MAX, CAP_MIN
    from softsignal.policy import load_policy

    assert hard_limits() == (CAP_MIN, CAP_MAX, int(load_policy()["min_audit_adults"]))


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


@pytest.mark.parametrize("cap", ["NaN", "Infinity", "-Infinity"])
def test_schema_rejects_a_non_finite_cap(cap):
    body = json.dumps(good()).replace("0.15", cap)
    with pytest.raises(ValidationError):
        A2Output.model_validate_json(body)


@pytest.mark.parametrize("cap", ["true", "false", '"0.2"', "null", "[0.2]"])
def test_schema_rejects_a_bool_or_string_cap(cap):
    body = json.dumps(good()).replace("0.15", cap)
    with pytest.raises(ValidationError):
        A2Output.model_validate_json(body)


def test_schema_accepts_an_integer_cap():
    assert A2Output.model_validate_json(json.dumps(good()).replace("0.15", "1")).cap == 1.0  # clamped later


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
    p = make_payload(guards=GUARDS | {"hold_required": True}, audit=AUDIT | {"audit_adults": 90},
                     rule={"action": "hold", "cap": 0.15})
    assert validate_output(good(action="re-tune", reason="90 audit adults."), p)[0] == GUARDRAIL
    assert validate_output(good(action="promote", reason="90 audit adults."), p)[0] == GUARDRAIL
    assert validate_output(good(action="hold", reason="90 audit adults."), p) == (None, [])


def test_promote_needs_the_loops_pooled_test():
    assert validate_output(good(action="promote"), make_payload())[0] == GUARDRAIL
    p = make_payload(guards=GUARDS | {"promote_allowed": True})
    assert validate_output(good(action="promote"), p) == (None, [])


def test_guards_are_rederived_from_the_counts_not_trusted():
    p = make_payload()
    p["audit"] = AUDIT | {"audit_adults": 10, "mode": "ACTIVE"}  # flags say all clear, the counts do not
    p["guards"] = {"hold_required": False, "promote_allowed": True}
    assert validate_output(good(action="re-tune", reason="10 audit adults."), p)[0] == GUARDRAIL
    p["audit"] = AUDIT | {"mode": "ACTIVE"}
    assert validate_output(good(action="promote"), p)[0] == GUARDRAIL  # not SHADOW


@pytest.mark.parametrize("reason,ok", [
    ("Cap 15% is kept, pooled 12% is within it.", True),
    ("Cap 0.15 and 12% pooled.", True),
    ("Cap 99% would be fine.", False),
    ("Cap 14% would be fine.", False),
])
def test_percent_form_of_an_input_fraction_is_accepted(payload, reason, ok):
    got = validate_output(good(reason=reason, cites=["thresholds.cap"]), payload)[0]
    assert (got is None) == ok and (ok or got == NUMBER_NOT_IN_INPUT)


def test_percent_forms_are_rounded_fractions(payload):
    forms = percent_forms(payload)
    assert 15.0 in forms and 12.0 in forms and 10.0 in forms and 81.0 in forms


def test_hold_is_allowed_after_the_floor(payload):
    assert validate_output(good(action="hold"), payload) == (None, [])  # A2 may be more careful than the rule


@pytest.mark.parametrize("cap,want", [(0.15, 0.15), (0.5, 0.3), (0.01, 0.08), (0.3, 0.3), (0.08, 0.08)])
def test_cap_is_clamped_to_bounds(payload, cap, want):
    out, notes = clamp_output(good(cap=cap), payload)
    assert out["cap"] == want and bool(notes) == (cap != want)
    assert all(n.startswith("CLAMPED") for n in notes)


# --- fallback -------------------------------------------------------------------------------------

def test_clamped_reason_matches_the_applied_cap(payload):
    out, _ = clamp_output(good(cap=0.5, reason="Cap 0.5 is wanted."), payload)
    assert out["cap"] == 0.3 and out["reason"].endswith("(cap clamped from 0.5 to 0.3)")
    out, _ = clamp_output(good(cap=0.5, reason="x" * 400), payload)
    assert len(out["reason"]) == 400 and out["reason"].endswith("to 0.3)")


def test_fallback_is_clamped_even_if_the_rule_is_not(payload):
    p = copy.deepcopy(payload)
    p["rule"]["cap"] = 0.9  # bypasses a2_input on purpose
    r = run_a2(p, client=None)
    assert r.output["cap"] == 0.3 and r.errors and r.errors[0].startswith("CLAMPED")


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


def test_the_prompt_input_is_exactly_the_payload_and_carries_no_forbidden_key(payload):
    from softsignal.agents.contracts import LABEL_KEYS, TEST_METRIC_KEYS, check_barrier

    user = user_message(payload)
    body = json.loads(user.split("<input>", 1)[1].rsplit("</input>", 1)[0])
    assert body == payload  # nothing added to the prompt beyond the contract fields
    check_barrier(body)  # parsed from the prompt itself, not from the payload object
    assert not set(body) & (LABEL_KEYS | TEST_METRIC_KEYS)
    assert user.count("<input>") == 1


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


# --- hard limits, rule guards and the model's own cap ----------------------------------------------

def hand_built(**over):
    """A payload that skipped a2_input (a caller bug): the checks must still hold the code limits."""
    p = copy.deepcopy(make_payload())
    for k, v in over.items():
        p[k] |= v
    return p


def test_loose_cap_bounds_in_a_hand_built_payload_do_not_loosen_the_clamp():
    p = hand_built(bounds={"cap_min": 0.0, "cap_max": 1.0})
    out, notes = clamp_output(good(cap=0.95), p)
    assert out["cap"] == 0.3 and notes
    out, _ = clamp_output(good(cap=0.0), p)
    assert out["cap"] == 0.08


@pytest.mark.parametrize("action,allowed,mode,adults,want", [
    ("re-tune", False, "SHADOW", 150, "re-tune"),
    ("hold", False, "SHADOW", 150, "hold"),
    ("promote", True, "SHADOW", 150, "promote"),
    ("promote", False, "SHADOW", 150, "re-tune"),
    ("promote", True, "ACTIVE", 150, "re-tune"),
    ("re-tune", False, "SHADOW", 10, "hold"),
    ("promote", True, "SHADOW", 10, "hold"),
])
def test_guarded_action(action, allowed, mode, adults, want):
    p = hand_built(guards={"promote_allowed": allowed, "hold_required": adults < 120},
                   audit={"mode": mode, "audit_adults": adults})
    assert guarded_action(action, p) == want


def test_fallback_obeys_the_guards_even_if_the_rule_does_not():
    p = hand_built(rule={"action": "re-tune", "cap": 0.15}, guards={"hold_required": True}, audit={"audit_adults": 10})
    r = run_a2(p, client=None)
    assert r.output["action"] == "hold" and r.fallback_reason == OFFLINE
    A2Output(**r.output)


def test_a_cap_the_model_states_is_not_an_invented_number(payload):
    assert validate_output(good(cap=0.2, reason="Raise the cap to 0.2 for 150 audit adults."), payload) == (None, [])
    assert validate_output(good(cap=0.2, reason="Raise the cap to 20% for 150 audit adults."), payload) == (None, [])
    assert validate_output(good(cap=0.2, reason="Raise the cap to 0.25."), payload)[0] == NUMBER_NOT_IN_INPUT


def test_a_stated_out_of_range_cap_is_clamped_live_not_a_fallback(payload):
    r = run_a2(payload, client=FakeClient(reply(good(cap=0.5, reason="Cap 0.5 wanted, 150 audit adults."))))
    assert r.status == LIVE and r.output["cap"] == 0.3 and r.errors == ["CLAMPED cap 0.5 -> 0.3"]
    assert r.output["reason"].endswith("(cap clamped from 0.5 to 0.3)")


def test_hold_rule_end_to_end_through_the_model_path():
    p = make_payload(guards=GUARDS | {"hold_required": True}, audit=AUDIT | {"audit_adults": 90},
                     rule={"action": "hold", "cap": 0.15})
    r = run_a2(p, client=FakeClient(reply(good(action="re-tune", reason="90 audit adults."))))
    assert r.status == FALLBACK and r.fallback_reason == GUARDRAIL and r.output["action"] == "hold"


def test_a_bool_cap_from_the_model_falls_back_not_live(payload):
    bad = json.dumps(good()).replace("0.15", "true")
    r = run_a2(payload, client=FakeClient(reply(text=bad)))
    assert r.status == FALLBACK and r.fallback_reason == INVALID


def test_clamp_note_cuts_the_reason_at_a_word_boundary(payload):
    long = "Cap 0.15 holds with 150 audit adults. " * 12  # over 400 characters
    out, _ = clamp_output(good(cap=0.5, reason=long.strip()), payload)
    note = " (cap clamped from 0.5 to 0.3)"
    assert len(out["reason"]) <= 400 and out["reason"].endswith(note)
    body = out["reason"].removesuffix(note)
    assert body and long.startswith(body)  # a whole-word prefix of the model's text
    assert long[len(body)] in " ."  # the cut did not land inside a number or word


def test_hard_limits_are_cached_per_policy_file():
    from softsignal.agents import contracts

    contracts._limits_from.cache_clear()
    hard_limits()
    hard_limits()
    info = contracts._limits_from.cache_info()
    assert info.hits >= 1 and info.misses == 1


def test_the_payload_builds_from_a_real_loop_state_and_rule_decision():
    """The contract fields exist on loop.py's State and rule_decision, and the guards agree with the rule."""
    from softsignal import loop
    from softsignal.policy import load_policy

    policy = load_policy()
    for adults, streak, mode, want in ((90, 0, "SHADOW", "hold"), (150, 0, "SHADOW", "re-tune"),
                                       (150, loop.PROMOTE_STREAK, "SHADOW", "promote"),
                                       (150, loop.PROMOTE_STREAK, "ACTIVE", "re-tune")):
        state = loop.new_state(policy)
        state.mode, state.streak = mode, streak
        rule = loop.rule_decision(state, policy, adults)
        assert rule["action"] == want
        th = state.live.th
        p = a2_input(
            round_id=3,
            thresholds={"t_verify": th.t_verify, "t_soft": th.t_soft, "cap": th.cap, "flags": list(th.flags)},
            audit={"mode": state.mode, "streak": state.streak, "audit_adults": adults, "audit_teens": adults,
                   "round_audit_adults": 30, "pooled_adults": 60, "pooled_false_teen_rate": 0.1,
                   "candidate_false_teen": None},
            bounds={"cap_min": loop.CAP_MIN, "cap_max": loop.CAP_MAX, "min_audit_adults": policy["min_audit_adults"],
                    "cap_default": policy["cap_false_teen"]},
            guards={"hold_required": adults < policy["min_audit_adults"],
                    "promote_allowed": state.mode == loop.SHADOW and state.streak >= loop.PROMOTE_STREAK
                    and adults >= policy["min_audit_adults"]},
            rule=rule)
        assert p["rule"]["action"] == want


# --- review 4: unit slips, limits failures, constants, a real loop run -----------------------------

@pytest.mark.parametrize("cap", [15.0, 15, 1.5, -0.1, 100.0])
def test_a_percent_scale_or_negative_cap_is_invalid_not_clamped(payload, cap):
    out = good(cap=cap, reason="Cap 15% kept, 150 audit adults.")
    assert validate_output(out, payload)[0] == INVALID
    r = run_a2(payload, client=FakeClient(reply(out)))
    assert r.status == FALLBACK and r.fallback_reason == INVALID
    assert (r.output["action"], r.output["cap"]) == ("re-tune", 0.15)  # the rule's cap, not 0.3


@pytest.mark.parametrize("cap", [0.0, 1.0, 0.5, 0.05])
def test_a_fraction_outside_the_bounds_is_still_clamped(payload, cap):
    r = run_a2(payload, client=FakeClient(reply(good(cap=cap))))
    assert r.status == LIVE and r.output["cap"] == min(0.3, max(0.08, cap))


def break_limits(monkeypatch):
    def boom():
        raise OSError("policy.yaml is gone")
    monkeypatch.setattr("softsignal.agents.a2_controller.cap_limits", boom)


def test_unreadable_limits_do_not_break_the_offline_fallback(payload, monkeypatch):
    break_limits(monkeypatch)
    r = run_a2(payload, client=None)
    assert r.status == FALLBACK and r.fallback_reason == OFFLINE
    assert (r.output["action"], r.output["cap"]) == ("re-tune", 0.15)
    assert any("limits unavailable" in e for e in r.errors)
    A2Output(**r.output)


def test_unreadable_limits_on_a_live_reply_fall_back_not_raise(payload, monkeypatch):
    break_limits(monkeypatch)
    r = run_a2(payload, client=FakeClient(reply(good())))  # validate_output cannot read the limits either
    assert r.status == FALLBACK and r.fallback_reason == INVALID
    assert (r.output["action"], r.output["cap"]) == ("re-tune", 0.15)


def test_clamp_failing_after_a_valid_reply_falls_back_not_raises(payload, monkeypatch):
    monkeypatch.setattr("softsignal.agents.a2_controller.validate_output", lambda out, p: (None, []))
    break_limits(monkeypatch)
    r = run_a2(payload, client=FakeClient(reply(good())))
    assert r.status == FALLBACK and r.fallback_reason == INVALID and r.rejected


def test_action_constants_match_the_loops():
    from softsignal import loop
    from softsignal.agents.contracts import A2_ACTIONS, INSUFFICIENT_INPUT
    from softsignal.agents.base import INSUFFICIENT

    assert A2_ACTIONS == (loop.HOLD, loop.RETUNE, loop.PROMOTE)
    assert INSUFFICIENT_INPUT == INSUFFICIENT


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    """A real R0-R7 loop run on the real data (the oracle, the stack, the hold rule and the promote test)."""
    from softsignal import loop
    from softsignal.agent_timer import AgentTimer as T
    from softsignal.data import load_data

    train, test = load_data(on_param_mismatch="error")
    env = loop.make_env(train, test, timer=T(tmp_path_factory.mktemp("a2") / "calls.jsonl", run="t"))
    rounds, records = loop.run_loop(env, None, loop.new_state(env.policy))
    return env, rounds, records


def test_the_payload_builds_from_every_round_of_a_real_loop_run(real_run):
    """The contract fields exist in what a real round produces, and the guards agree with the rule each round."""
    from softsignal import loop

    env, rounds, records = real_run
    pol, seen = env.policy, set()
    for k in range(1, len(rounds)):
        row, rec, prev = rounds.iloc[k], records[k], rounds.iloc[k - 1]
        ev, adults = rec["evidence"], int(row["n_audit_adults"])
        rule = {"action": rec["rule_decision"]["action"], "cap": rec["rule_decision"]["cap"]}
        p = a2_input(
            round_id=int(row["round"]),
            thresholds={"t_verify": float(prev["t_verify"]), "t_soft": float(prev["t_soft"]), "cap": float(prev["cap"]),
                        "flags": []},
            audit={"mode": prev["mode"], "streak": ev["streak"], "audit_adults": adults,
                   "audit_teens": int(row["n_labels"]) - adults, "round_audit_adults": ev["round_audit_adults"],
                   "pooled_adults": ev["pooled_adults"], "pooled_false_teen_rate": ev["pooled_ft"],
                   "candidate_false_teen": ev["cand_ft"]},
            bounds={"cap_min": loop.CAP_MIN, "cap_max": loop.CAP_MAX, "min_audit_adults": pol["min_audit_adults"],
                    "cap_default": pol["cap_false_teen"]},
            guards={"hold_required": adults < pol["min_audit_adults"],
                    "promote_allowed": prev["mode"] == loop.SHADOW and ev["streak"] >= loop.PROMOTE_STREAK
                    and adults >= pol["min_audit_adults"]},
            rule=rule, policy=pol)
        assert p["rule"]["action"] == row["action"] or ev["promote_refused"]
        assert validate_output(fallback_output(p), p) == (None, [])
        seen.add(rule["action"])
    assert {"hold", "re-tune"} <= seen  # the run covered both the hold rule and re-tuning


# --- review 5: floats are rounded for the prompt; the audit floor is the loop's policy -------------

REAL_FLOATS = {"thresholds": THRESHOLDS | {"t_verify": 0.8123456789012345, "t_soft": 0.4166666666666667},
               "audit": AUDIT | {"pooled_false_teen_rate": 0.1333333333333333, "candidate_false_teen": 0.0689655}}


def test_loop_floats_are_rounded_in_the_input_and_the_model_can_quote_them():
    p = make_payload(**REAL_FLOATS)
    assert p["thresholds"]["t_verify"] == 0.812 and p["thresholds"]["t_soft"] == 0.417
    assert p["audit"]["pooled_false_teen_rate"] == 0.133 and p["audit"]["candidate_false_teen"] == 0.069
    for reason in ("Pooled rate 0.133 is within the cap.", "Pooled rate 13.3% is within the cap.",
                   "Candidate 6.9% at t_verify 0.812."):
        assert validate_output(good(reason=reason, cites=["audit.pooled_false_teen_rate"]), p) == (None, [])
    assert validate_output(good(reason="Pooled rate 0.1333333333333333."), p)[0] == NUMBER_NOT_IN_INPUT


def test_rounding_leaves_counts_none_and_bools_alone():
    p = make_payload(audit=AUDIT | {"candidate_false_teen": None, "pooled_false_teen_rate": None})
    assert p["audit"]["candidate_false_teen"] is None and p["audit"]["audit_adults"] == 150
    assert p["thresholds"]["cap"] == 0.15 and p["rule"]["cap"] == 0.15 and p["bounds"] == BOUNDS


def test_the_floor_is_the_loops_policy_when_it_is_passed():
    lower = {"min_audit_adults": 50}
    bounds = BOUNDS | {"min_audit_adults": 50}
    with pytest.raises(ValueError, match="looser than the code limits"):  # without the policy, policy.yaml's 120 rules
        make_payload(bounds=bounds, audit=AUDIT | {"audit_adults": 60}, guards=GUARDS)
    p = make_payload(bounds=bounds, audit=AUDIT | {"audit_adults": 60}, guards=GUARDS, policy=lower)
    assert validate_output(good(action="re-tune", reason="60 audit adults."), p) == (None, [])  # not held at 120
    assert guarded_action("re-tune", p) == "re-tune"
    p = make_payload(bounds=bounds, audit=AUDIT | {"audit_adults": 40},
                     guards=GUARDS | {"hold_required": True}, rule={"action": "hold", "cap": 0.15}, policy=lower)
    assert validate_output(good(action="re-tune", reason="40 audit adults."), p)[0] == GUARDRAIL


def test_a_bounds_floor_below_the_loops_policy_is_refused():
    with pytest.raises(ValueError, match="looser than the code limits"):
        make_payload(policy={"min_audit_adults": 150})  # BOUNDS says 120


def test_cap_limits_read_no_file(monkeypatch):
    from softsignal.agents import contracts

    monkeypatch.setattr(contracts, "_limits_from", lambda *a: (_ for _ in ()).throw(AssertionError("read the file")))
    assert contracts.cap_limits() == (0.08, 0.3)
    assert hard_limits({"min_audit_adults": 77}) == (0.08, 0.3, 77)
