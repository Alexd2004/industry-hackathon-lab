"""Step 15, sub-step 3: A2 inside the round (loop.run_loop's decide callback and crew.run_crew's decider).

A fake decide stands in for A2, so the loop's own rules are what is tested: what is logged, what is applied, and
what the loop enforces again in code. No network.
"""
import json
from unittest import mock

import pytest

from softsignal import crew, loop
from softsignal.agent_timer import AgentTimer
from softsignal.agents.base import FALLBACK, LIVE
from softsignal.data import load_data
from softsignal.text_model import build_matrix
from softsignal.ui_loop import valid_decision

N_ROUNDS = 5  # R5 is the first round past the 120 audit adult floor (seed 42: 156 audit adults)


@pytest.fixture(scope="module")
def split_and_tm(tmp_path_factory):
    train, test = load_data(on_param_mismatch="error")
    return train, test, build_matrix(train[loop.ID_COL], cache_dir=tmp_path_factory.mktemp("cache"))


def make_env(split_and_tm, tmp_path, run="20261004T000000.000000Z-a2"):
    train, test, tm = split_and_tm
    return loop.make_env(train, test, timer=AgentTimer(tmp_path / "calls.jsonl", run=run), tm=tm)


def a2_block(status=LIVE, **out) -> dict:
    output = {"action": "hold", "cap": 0.15, "reason": "test", "cites": ["audit"]} | out
    return {"status": status, "output": output, "fallback_reason": None if status == LIVE else "offline"}


def scripted(by_round: dict, seen: list | None = None):
    """A decide that returns by_round[ctx.round] (a block, or an Exception to raise); no entry: A2 has no output."""

    def decide(ctx, blocks):
        if seen is not None:
            seen.append(ctx)
        item = by_round.get(ctx.round)
        if isinstance(item, Exception):
            raise item
        return {} if item is None else {"a2": item}

    return decide


SCRIPT = {
    1: a2_block(action="re-tune", cap=0.20),  # before the floor: the loop forces hold, A2's cap is still logged
    2: a2_block(status=FALLBACK),  # fell back: the rule decides
    3: RuntimeError("boom"),  # a failing decider never breaks the round
    4: a2_block(action="promote", cap=0.10),  # before the floor too: hold
    5: a2_block(action="re-tune", cap=0.10),  # past the floor: refit at A2's cap
}


@pytest.fixture(scope="module")
def applied_run(split_and_tm, tmp_path_factory):
    env = make_env(split_and_tm, tmp_path_factory.mktemp("applied"))
    caps, seen = [], []
    real = loop.refit

    def spy(env_, state, cap):
        caps.append(cap)
        return real(env_, state, cap)

    with mock.patch.object(loop, "refit", spy):
        rounds, records = loop.run_loop(env, N_ROUNDS, decide=scripted(SCRIPT, seen), apply_a2=True)
    return rounds, records, caps, seen


def test_a2_decision_and_diff_are_logged(applied_run):
    _, records, _, _ = applied_run
    r1 = records[1]
    assert r1["a2"]["output"]["action"] == "re-tune"
    assert r1["rule_decision"]["action"] == "hold" and r1["rule_decision"]["cap"] == 0.15
    assert r1["diff"] == {"cap": [0.15, 0.20], "action": ["hold", "re-tune"]}
    assert all(valid_decision(r) for r in records)


def test_diff_count_is_the_length_of_the_diff(applied_run):
    rounds, records, _, _ = applied_run
    assert rounds["diff_count"].tolist() == [len(r["diff"]) for r in records]
    assert rounds["diff_count"].tolist() == [0, 2, 0, 0, 2, 1]


def test_a2_is_applied_but_the_floor_is_enforced_again(applied_run):
    rounds, records, _, _ = applied_run
    r1 = records[1]
    assert r1["applied"]["source"] == "A2" and rounds.loc[1, "applied_source"] == "A2"
    assert r1["applied"]["decision"]["action"] == "hold"  # A2 asked to re-tune with 27 audit adults
    assert r1["applied"]["decision"]["cap"] == 0.20
    assert records[4]["applied"]["decision"]["action"] == "hold" and rounds.loc[4, "action"] == "hold"


def test_a_fallback_leaves_the_rule_applied(applied_run):
    rounds, records, _, _ = applied_run
    assert records[2]["applied"]["source"] == "rule" and rounds.loc[2, "applied_source"] == "rule"
    assert records[2]["diff"] == {}


def test_a_failing_decider_is_logged_and_the_rule_decides(applied_run):
    rounds, records, _, _ = applied_run
    assert records[3]["agent_error"].startswith("A2 RuntimeError: boom")
    assert records[3]["applied"]["source"] == "rule" and records[3]["a2"]["status"] is None
    assert rounds.loc[3, "diff_count"] == 0


def test_a2_cap_drives_the_refit_and_the_policy_cap_the_promote_gate(applied_run):
    rounds, records, caps, seen = applied_run
    assert caps == [0.10]  # one refit, at A2's cap, in R5 (R1-R4 hold)
    assert records[5]["applied"]["source"] == "A2" and records[5]["applied"]["decision"]["cap"] == 0.10
    # the row's cap is the live rule's: in SHADOW the starter stays live, only the candidate got A2's cap
    assert rounds.loc[5, "action"] == "re-tune" and rounds.loc[5, "cap"] == 0.15
    # the promote gate is the rule's: it was computed before A2 and from the policy cap, whatever A2 asked
    assert seen[4].guards["promote_allowed"] is False and seen[4].rule["cap"] == 0.15


def test_context_is_plain_json_from_before_the_apply_step(applied_run):
    _, _, _, seen = applied_run
    assert [c.round for c in seen] == [1, 2, 3, 4, 5]
    for c in seen:
        json.dumps([c.thresholds, c.audit, c.bounds, c.guards, c.rule])  # no numpy values
        assert c.audit["mode"] == "SHADOW" and c.bounds["cap_min"] == loop.CAP_MIN and c.bounds["cap_max"] == loop.CAP_MAX
    assert seen[0].guards == {"hold_required": True, "promote_allowed": False}
    assert seen[4].guards["hold_required"] is False and seen[4].audit["audit_adults"] >= 120
    assert seen[0].audit["audit_teens"] >= 0 and seen[0].thresholds["flags"] == [loop.STARTER]


def test_without_apply_a2_the_rule_is_applied_and_a2_is_only_logged(split_and_tm, tmp_path):
    env = make_env(split_and_tm, tmp_path)
    rounds, records = loop.run_loop(env, 2, decide=scripted({1: a2_block(action="re-tune", cap=0.20),
                                                              2: a2_block(action="re-tune", cap=0.20)}))
    for r in records[1:]:
        assert r["applied"]["source"] == "rule" and r["applied"]["decision"]["cap"] == 0.15
        assert r["a2"]["output"]["cap"] == 0.20 and r["diff"]["cap"] == [0.15, 0.20]
    assert rounds["applied_source"].tolist() == ["starter", "rule", "rule"]
    assert rounds["diff_count"].tolist() == [0, 2, 2]


def test_no_decider_changes_nothing(split_and_tm, tmp_path):
    env = make_env(split_and_tm, tmp_path)
    rounds, records = loop.run_loop(env, 1)
    assert records[1]["diff"] == {} and records[1]["a2"]["status"] is None and rounds.loc[1, "diff_count"] == 0


def test_crew_offline_a2_falls_back_to_the_rule_every_round(split_and_tm, tmp_path):
    rounds, records = crew.run_crew(make_env(split_and_tm, tmp_path), None, 2)
    assert records[0]["a2"]["status"] is None  # R0 has no decision to make
    for r in records[1:]:
        assert r["a2"]["status"] == FALLBACK and r["diff"] == {} and r["applied"]["source"] == "rule"
        assert r["a2"]["output"]["action"] == r["rule_decision"]["action"] == "hold"
    assert rounds["diff_count"].tolist() == [0, 0, 0]
