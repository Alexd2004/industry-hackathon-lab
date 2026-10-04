"""Step 15, sub-step 3: A2 inside the round (loop.run_loop's decide callback and crew.run_crew's decider).

A fake decide stands in for A2, so the loop's own rules are what is tested: what is logged, what is applied, and
what the loop enforces again in code. No network.
"""
import json
from unittest import mock

import pandas as pd
import pytest

from softsignal import crew, loop
from softsignal.agent_timer import AgentTimer
from softsignal.agents.base import FALLBACK, LIVE
from softsignal.data import load_data
from softsignal.text_model import build_matrix
from softsignal.metrics import ROUNDS_COLS
from softsignal.ui_loop import diff_table, load_rounds, valid_decision

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


# ---- the --mode switch ----
def run_main(module, monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", [module.__name__, *argv])
    module.main()


def test_loop_cli_default_is_rule_and_calls_no_agent(split_and_tm, monkeypatch, capsys):
    train, test, tm = split_and_tm
    monkeypatch.setattr(loop, "load_data", lambda **kw: (train, test))
    monkeypatch.setattr(loop, "build_matrix", lambda *a, **kw: tm)
    monkeypatch.setattr(crew, "run_crew", lambda *a, **kw: pytest.fail("rule mode must not run the crew"))
    run_main(loop, monkeypatch, "--rounds", "1", "--no-write")
    out = capsys.readouterr().out
    assert "applied_source" in out and "starter" in out and "appended run" not in out


def test_loop_cli_crew_mode_runs_the_crew_with_a2_applied(split_and_tm, monkeypatch, capsys):
    train, test, tm = split_and_tm
    monkeypatch.setattr(loop, "load_data", lambda **kw: (train, test))
    monkeypatch.setattr(loop, "build_matrix", lambda *a, **kw: tm)
    seen = {}
    real = crew.run_crew

    def spy(env, client, n_rounds, write=False, *a, **kw):
        seen.update(client=client, n_rounds=n_rounds, write=write, apply=kw.get("apply_a2", True))
        return real(env, None, n_rounds, write=False)  # offline, nothing written

    monkeypatch.setattr(crew, "run_crew", spy)
    run_main(loop, monkeypatch, "--mode", "crew", "--rounds", "1", "--no-write")
    assert seen["n_rounds"] == 1 and seen["write"] is False and seen["apply"] is True
    assert "applied_source" in capsys.readouterr().out


def test_loop_cli_rejects_an_unknown_mode(monkeypatch):
    with pytest.raises(SystemExit) as e:
        run_main(loop, monkeypatch, "--mode", "both")
    assert e.value.code == 2


class Stop(Exception):
    """Raised by a spy to end main() right after run_crew was called."""


@pytest.mark.parametrize("mode, applies", [("crew", True), ("rule", False)])
def test_crew_cli_mode_sets_whether_a2_is_applied(split_and_tm, tmp_path, monkeypatch, mode, applies):
    train, test, _ = split_and_tm
    monkeypatch.setattr(crew, "load_data", lambda **kw: (train, test))
    monkeypatch.setattr(crew, "make_env", lambda tr, te, **kw: make_env(split_and_tm, tmp_path))
    monkeypatch.setattr(crew, "make_client", lambda: None)
    seen = {}

    def spy(env, client, n_rounds, write, *paths, apply_a2):
        seen["apply"] = apply_a2
        raise Stop

    monkeypatch.setattr(crew, "run_crew", spy)
    with pytest.raises(Stop):
        run_main(crew, monkeypatch, "--mode", mode, "--no-write")
    assert seen["apply"] is applies


def test_crew_cli_record_needs_crew_mode(monkeypatch):
    with pytest.raises(SystemExit) as e:
        run_main(crew, monkeypatch, "--mode", "rule", "--record")
    assert e.value.code == 2


def test_rule_mode_applies_the_rule_but_still_logs_a2(split_and_tm, tmp_path):
    env = make_env(split_and_tm, tmp_path)
    decide = scripted({1: a2_block(action="re-tune", cap=0.20)})
    rounds, records = loop.run_loop(env, 1, decide=decide, apply_a2=False)
    assert records[1]["applied"]["source"] == "rule" and records[1]["a2"]["output"]["cap"] == 0.20
    assert rounds.loc[1, "diff_count"] == 2


# ---- the cap the refit uses ----
def test_evidence_logs_the_caps_when_a2_steers_the_refit(applied_run):
    _, records, _, _ = applied_run
    for r in records[1:5]:  # R1-R4 hold: no refit
        assert r["evidence"]["refit_cap"] is None and r["evidence"]["cap_differs"] is False
        assert r["evidence"]["policy_cap"] == 0.15
    ev = records[5]["evidence"]
    assert (ev["policy_cap"], ev["refit_cap"], ev["cap_differs"]) == (0.15, 0.10, True)


def test_rule_mode_never_differs_from_the_policy_cap(split_and_tm, tmp_path):
    env = make_env(split_and_tm, tmp_path)
    _, records = loop.run_loop(env, 2, decide=scripted({1: a2_block(cap=0.20), 2: a2_block(cap=0.20)}))
    assert all(r["evidence"]["cap_differs"] is False for r in records[1:])


@pytest.fixture(scope="module")
def promote_run(split_and_tm, tmp_path_factory):
    """A2 re-tunes at cap 0.10 from R5 and promotes the first time the rule's gate allows it, also at cap 0.10."""
    env = make_env(split_and_tm, tmp_path_factory.mktemp("promote"))
    caps = []
    real = loop.refit

    def spy(env_, state, cap):
        caps.append((state.round, cap))
        return real(env_, state, cap)

    def decide(ctx, blocks):
        if ctx.guards["hold_required"]:
            return {"a2": a2_block(action="hold", cap=0.10)}
        action = "promote" if ctx.guards["promote_allowed"] else "re-tune"
        return {"a2": a2_block(action=action, cap=0.10)}

    with mock.patch.object(loop, "refit", spy):
        rounds, records = loop.run_loop(env, 8, decide=decide, apply_a2=True)
    return rounds, records, caps


def test_a_promote_refits_at_the_policy_cap_not_a2s(promote_run):
    rounds, records, caps = promote_run
    promoted = [r for r in records if r["applied"]["decision"]["action"] == "promote"]
    if not promoted:
        pytest.skip("the pooled gate never allowed a promote with this A2 script")
    r = promoted[0]
    assert dict(caps)[r["round"]] == 0.15  # refit at the cap the gate tested
    assert r["applied"]["decision"]["cap"] == 0.15 and r["applied"]["source"] == "A2"
    assert r["evidence"]["refit_cap"] == 0.15 and r["evidence"]["cap_differs"] is False
    assert r["a2"]["output"]["cap"] == 0.10 and r["diff"]["cap"] == [0.15, 0.10]  # A2's own cap is still logged
    assert all(cap == 0.10 for rnd, cap in caps if rnd != r["round"])  # re-tune rounds still use A2's cap


# ---- the rule's cutoff in the record ----
def test_rule_cutoff_is_unknown_when_a2_chose_something_else(applied_run):
    _, records, _, _ = applied_run
    assert records[1]["rule_decision"]["cutoff"] is None  # A2 applied hold at cap 0.20, the rule said hold at 0.15
    assert records[5]["rule_decision"]["cutoff"] is None  # A2 re-tuned at 0.10, the rule at 0.15
    assert isinstance(records[1]["applied"]["decision"]["cutoff"], float)


def test_rule_cutoff_is_kept_when_the_rule_was_applied_or_agreed(applied_run, split_and_tm, tmp_path):
    _, records, _, _ = applied_run
    assert isinstance(records[2]["rule_decision"]["cutoff"], float)  # FALLBACK: the rule was applied
    agree = loop.run_loop(make_env(split_and_tm, tmp_path), 1, decide=scripted({1: a2_block()}), apply_a2=True)[1]
    assert agree[1]["applied"]["source"] == "A2" and agree[1]["diff"] == {}
    assert agree[1]["rule_decision"]["cutoff"] == agree[1]["applied"]["decision"]["cutoff"]


def test_a_record_with_an_unknown_rule_cutoff_still_loads_in_the_tab(applied_run):
    _, records, _, _ = applied_run
    assert all(valid_decision(r) for r in records)
    table = diff_table(records[1]).set_index("field")
    assert table.loc["cutoff", "rule"] == "-" and table.loc["cap", "changed"] == "YES"


# ---- a results file from before diff_count ----
OLD_COLS = [c for c in ROUNDS_COLS if c != "diff_count"]


def test_write_run_names_the_old_file_and_the_fix(split_and_tm, tmp_path):
    rounds, records = loop.run_loop(make_env(split_and_tm, tmp_path), 0)
    old = tmp_path / "rounds.csv"
    old.write_text(",".join(OLD_COLS) + "\n")
    with pytest.raises(ValueError, match=r"older schema: move or delete .*rounds\.csv"):
        loop.write_run(rounds, records, old, tmp_path / "decisions.jsonl")
    assert old.read_text() == ",".join(OLD_COLS) + "\n"  # nothing was written


def test_loop_tab_tells_the_user_how_to_fix_an_old_file(tmp_path):
    old = tmp_path / "rounds.csv"
    old.write_text(",".join(OLD_COLS) + "\n")
    with pytest.raises(ValueError, match=r"missing columns \['diff_count'\]\. It is from an older schema: regenerate it"):
        load_rounds(old, [])


# ---- candidate age ----
@pytest.fixture(scope="module")
def hold_run(split_and_tm, tmp_path_factory):
    """A2 re-tunes in R5 (the first refit) and then holds in R6 and R7, past the audit floor."""
    env = make_env(split_and_tm, tmp_path_factory.mktemp("hold"))
    script = {5: a2_block(action="re-tune", cap=0.15), 6: a2_block(action="hold", cap=0.15),
              7: a2_block(action="hold", cap=0.15)}
    return loop.run_loop(env, 7, decide=scripted(script), apply_a2=True)


def test_evidence_shows_how_old_the_scored_candidate_is(hold_run):
    _, records = hold_run
    ages = [r["evidence"]["cand_age"] for r in records[1:]]
    assert ages == [None, None, None, None, None, 1, 2]  # R5 refit, R6 and R7 score that same candidate


def test_a_hold_past_the_floor_skips_the_refit_but_not_the_evidence(hold_run):
    rounds, records = hold_run
    assert [r["applied"]["decision"]["action"] for r in records[5:]] == ["re-tune", "hold", "hold"]
    assert rounds["refit_s"].isna().tolist()[:5] == [True] * 5 and rounds.loc[5, "refit_s"] > 0
    assert pd.isna(rounds.loc[6, "refit_s"]) and pd.isna(rounds.loc[7, "refit_s"])  # held: no refit
    assert records[6]["evidence"]["cand_ft"] is not None and records[7]["evidence"]["cand_ft"] is not None
