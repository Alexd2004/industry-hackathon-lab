"""Review loop (step 11, loop.py half): decisions, verify band, hold rule, label rules, determinism, files."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

import softsignal.loop as lp
from softsignal.agent_timer import AgentTimer, load_records
from softsignal.data import load_data
from softsignal.features import ID_COL, TARGET
from softsignal.metrics import ROUNDS_COLS, prf, psi
from softsignal.oracle import OracleError
from softsignal.policy import Thresholds, load_policy
from softsignal.text_model import build_matrix
from softsignal.tier1 import STARTER_CUT, STARTER_W, activity_score, blend, flag, style_score
from softsignal.ui_loop import valid_decision

# these tests pin the original promote rule and floor; the challenger (policy.yaml's default) has its own tests below
POL = {**load_policy(), "promote_rule": "pooled", "min_audit_adults": 120}


def state(mode=lp.SHADOW, streak=0):
    return lp.State(live=lp.starter_rule(0.15), mode=mode, streak=streak)


# ---- pure pieces ----
def test_clamp_cap_keeps_the_plan_bounds():
    assert (lp.clamp_cap(0.01), lp.clamp_cap(0.15), lp.clamp_cap(0.9)) == (0.08, 0.15, 0.30)


@pytest.mark.parametrize("adults,mode,streak,action", [
    (0, lp.SHADOW, 0, lp.HOLD),
    (119, lp.SHADOW, 5, lp.HOLD),  # the floor beats a full streak
    (120, lp.SHADOW, 0, lp.RETUNE),
    (200, lp.SHADOW, 1, lp.RETUNE),
    (200, lp.SHADOW, 2, lp.PROMOTE),
    (200, lp.ACTIVE, 9, lp.RETUNE),  # already active: never promoted twice
])
def test_rule_decision(adults, mode, streak, action):
    assert lp.rule_decision(state(mode, streak), POL, adults) == {"action": action, "cap": 0.15}


def test_rule_decision_clamps_the_cap():
    assert lp.rule_decision(state(), {**POL, "cap_false_teen": 0.5}, 0)["cap"] == 0.30


def test_starter_rule_has_no_soft_band_and_uses_the_starter_cutoff():
    r = lp.starter_rule(0.15)
    assert r.model is None and r.th.t_verify == r.th.t_soft == STARTER_CUT


def test_starter_score_is_the_starter_blend_rounded():
    _, test = load_data(on_param_mismatch="error")
    expect = blend(style_score(test), activity_score(test), STARTER_W).round(9).to_numpy()
    assert np.array_equal(lp.starter_score(test), expect)


def test_pick_verify_takes_the_top_scores_within_budget():
    ids = [f"a{i}" for i in range(6)]
    s = np.array([0.9, 0.1, 0.8, 0.7, 0.95, 0.2])
    bands = np.array(["verify", "none", "verify", "verify", "verify", "none"])
    assert lp.pick_verify(s, bands, ids, 2, 1) == ["a4", "a0"]
    assert lp.pick_verify(s, bands, ids, 10, 1) == ["a4", "a0", "a2", "a3"]  # under budget: all flagged
    assert lp.pick_verify(s, bands, ids, 0, 1) == []


def test_pick_verify_ties_are_seeded_not_by_id_order():
    ids = [f"a{i:03d}" for i in range(200)]
    s, bands = np.full(200, 0.5), np.full(200, "verify")
    a, b = lp.pick_verify(s, bands, ids, 50, 1), lp.pick_verify(s, bands, ids, 50, 1)
    assert a == b and a != ids[:50] and a != lp.pick_verify(s, bands, ids, 50, 2)


def test_false_teen_needs_an_adult():
    assert lp.false_teen(np.array([1, 1, 1]), np.array([0.9, 0.2, 0.8]), 0.5) is None
    y, s = np.array([0, 0, 0, 0, 1]), np.array([0.9, 0.2, 0.8, 0.1, 0.9])
    assert lp.false_teen(y, s, 0.5) == 0.5


def test_psi_is_zero_for_the_same_scores_and_grows_with_a_shift():
    rng = np.random.default_rng(0)
    e = rng.normal(size=2000)
    assert psi(e, e) == pytest.approx(0.0, abs=1e-9)
    assert psi(e, rng.normal(size=2000)) < 0.05 < psi(e, rng.normal(1.0, 1.0, 2000))
    assert psi(np.full(50, 0.5), np.full(20, 0.5)) == 0.0  # tied scores share one bin
    assert psi(np.full(50, 0.5), np.arange(50) / 50) > 1.0  # spread against a constant is a big shift
    assert psi(e, rng.normal(size=100), bins=1) == 0.0  # one bin cannot shift


def test_psi_rejects_empty_and_non_finite():
    with pytest.raises(ValueError):
        psi([], [1.0])
    with pytest.raises(ValueError):
        psi([1.0, 2.0], [np.nan])


def test_make_env_rejects_an_unknown_threshold_source():
    with pytest.raises(ValueError, match="threshold_source"):
        lp.make_env(pd.DataFrame(), pd.DataFrame(), threshold_source="test")


def test_record_has_the_decisions_shape_and_no_agent_output():
    rec = lp.make_record("r", 3, {"cutoff": 0.4, "cap": 0.15, "action": "hold"}, "rule")
    assert valid_decision(rec) and rec["a2"]["status"] is None and rec["diff"] == {}


def test_write_run_appends_with_one_header(tmp_path):
    rounds = pd.DataFrame([{c: 0 for c in ROUNDS_COLS}])
    recs = [lp.make_record("r", 0, {"cutoff": 0.5, "cap": 0.15, "action": "starter"}, "starter")]
    r_path, d_path = tmp_path / "rounds.csv", tmp_path / "decisions.jsonl"
    lp.write_run(rounds, recs, r_path, d_path)
    lp.write_run(rounds, recs, r_path, d_path)
    assert len(pd.read_csv(r_path)) == 2 and r_path.read_text().count("run,round") == 1
    assert len([json.loads(x) for x in d_path.read_text().splitlines()]) == 2


def test_write_run_rejects_a_changed_header_and_writes_nothing(tmp_path):
    rounds = pd.DataFrame([{c: 0 for c in ROUNDS_COLS}])
    recs = [lp.make_record("r", 0, {"cutoff": 0.5, "cap": 0.15, "action": "starter"}, "starter")]
    r_path, d_path = tmp_path / "rounds.csv", tmp_path / "decisions.jsonl"
    r_path.write_text("run,round\nr,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        lp.write_run(rounds, recs, r_path, d_path)
    assert r_path.read_text(encoding="utf-8") == "run,round\nr,0\n"
    assert not d_path.exists() and not list(tmp_path.glob("*.tmp"))


def test_new_state_clamps_the_cap_once():
    assert lp.new_state({**POL, "cap_false_teen": 0.5}).live.th.cap == 0.30


# ---- whole loop on the real split ----
def make(n_rounds, tmp_path, source="audit"):
    train, test = load_data(on_param_mismatch="error")
    env = lp.make_env(train, test, policy=POL, threshold_source=source,
                      timer=AgentTimer(tmp_path / "calls.jsonl", run="t"))
    st = lp.new_state(env.policy)
    rounds, records = lp.run_loop(env, n_rounds, st)
    return env, test, rounds, records, st


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    return make(None, tmp_path_factory.mktemp("loop"))


def test_rounds_have_the_frozen_columns_and_r0_to_r7(full):
    _, _, rounds, records, _ = full
    assert list(rounds.columns) == ROUNDS_COLS
    assert rounds["round"].tolist() == list(range(8)) == [r["round"] for r in records]
    assert all(valid_decision(r) for r in records)


def test_r0_is_the_starter_rule_on_the_frozen_test_set(full):
    _, test, rounds, records, _ = full
    r0 = rounds.iloc[0]
    pred = flag(blend(style_score(test), activity_score(test), STARTER_W), STARTER_CUT)
    assert (r0["prec"], r0["rec"], r0["ft"], r0["mt"]) == pytest.approx(prf(test[TARGET], pred))
    assert (r0["action"], r0["applied_source"], r0["n_labels"]) == ("starter", "starter", 0)
    assert records[0]["applied"]["source"] == "starter"


def test_hold_rule_blocks_any_refit_below_the_floor(full):
    _, _, rounds, _, _ = full
    r = rounds[rounds["round"] > 0]
    below = r[r["n_audit_adults"] < POL["min_audit_adults"]]
    assert len(below) and (below["action"] == "hold").all() and below["refit_s"].isna().all()
    assert (below["t_verify"] == STARTER_CUT).all()  # the starter stays live
    above = r[r["n_audit_adults"] >= POL["min_audit_adults"]]
    assert len(above) and (above["action"] != "hold").all() and above["refit_s"].notna().all()


def test_verify_band_is_cut_to_the_review_budget(full):
    env, _, rounds, _, _ = full
    r = rounds[rounds["round"] > 0]
    assert (r["n_verify"] <= 75).all() and (r["n_flagged"] >= r["n_verify"]).all()
    assert (r["n_flagged"] > r["n_verify"]).any()  # the budget really binds under the starter rule


def test_counts_are_cumulative_and_match_the_oracle(full):
    env, _, rounds, _, _ = full
    r = rounds[rounds["round"] > 0]
    assert r["n_labels"].is_monotonic_increasing and r["n_audit_adults"].is_monotonic_increasing
    assert r["n_labels"].iloc[-1] == len(env.oracle.revealed("all"))
    assert r["n_audit_adults"].iloc[-1] == env.oracle.audit_counts()["adults"]


def test_no_test_id_reaches_the_oracle(full):
    env, test, _, _, st = full
    test_ids = set(test[ID_COL])
    assert not test_ids & set(env.oracle.revealed("all")[ID_COL])
    assert not test_ids & {i for rec in env.oracle.log for i in rec["verify_ids"] + rec["audit_ids"]}
    assert not test_ids & set(env.train_ids) and not test_ids & set(st.seen[ID_COL])


def test_audit_ft_and_psi_are_blank_only_where_undefined(full):
    _, _, rounds, _, _ = full
    r = rounds[rounds["round"] > 0]
    after_promote = set(r.loc[r["action"] == "promote", "round"] + 1)  # the history restarts on a new scorer
    assert r["audit_ft"].notna().all() and np.isnan(r["psi"].iloc[0])
    assert r.loc[~r["round"].isin(after_promote) & (r["round"] > 1), "psi"].notna().all()
    assert np.isnan(rounds.loc[0, "audit_ft"]) and np.isnan(rounds.loc[0, "psi"])


def test_mode_only_changes_through_the_promote_rule(full):
    _, _, rounds, _, _ = full
    assert rounds["mode"].isin(["SHADOW", "ACTIVE"]).all()
    active = rounds[rounds["mode"] == "ACTIVE"]
    if len(active):
        first = active["round"].min()
        assert rounds.loc[rounds["round"] == first, "action"].item() == "promote"
        assert (active["round"].diff().dropna() == 1).all() and active["round"].max() == 7  # no way back


def test_live_thresholds_only_move_once_active(full):
    _, _, rounds, _, _ = full
    shadow = rounds[rounds["mode"] == "SHADOW"]
    assert (shadow["t_verify"] == STARTER_CUT).all() and shadow["t_soft"].isna().all()


def test_timer_logs_one_run_round_per_round_and_a_refit_per_retune(full):
    env, _, rounds, _, _ = full
    recs = load_records(env.timer.path)
    assert sorted(r["round"] for r in recs if r["step"] == "run_round") == list(range(1, 8))
    assert sum(r["step"] == "refit" for r in recs) == int(rounds["refit_s"].notna().sum())


def test_same_seed_gives_the_same_rounds(tmp_path):
    a = make(5, tmp_path / "a")
    b = make(5, tmp_path / "b")
    pd.testing.assert_frame_equal(a[2].drop(columns="refit_s"), b[2].drop(columns="refit_s"))
    assert a[4].candidate.th == b[4].candidate.th


def test_biased_toggle_picks_a_different_candidate_cutoff(tmp_path):
    a, b = make(5, tmp_path / "a"), make(5, tmp_path / "b", source="all_verified")
    assert a[4].candidate.th.t_verify != b[4].candidate.th.t_verify
    assert a[1][TARGET].equals(b[1][TARGET])  # same frozen test set either way


# ---- promote path, on the real batches with refit stubbed ----
class ZeroModel:
    """Scores everything 0, so a candidate with t_verify above 0 never flags anyone (false-teen 0)."""

    def score(self, df):
        return np.zeros(len(df))


def candidate(flags=()):
    return lp.Rule(Thresholds(0.9, 0.5, 0.15, 0.0, 50, 50, flags), ZeroModel())


@pytest.fixture(scope="module")
def split():
    train, test = load_data(on_param_mismatch="error")
    return train, test, build_matrix(tuple(train[ID_COL].astype(str)))


def run_rounds(split, tmp_path, monkeypatch, n, flags=(), start_flags=(), min_adults=1, decide=None, apply_a2=False,
               caps=None, margins=None, windows=None):
    """n rounds with min_audit_adults=1 and a stubbed refit; the state starts SHADOW with a candidate in place.
    decide / apply_a2: A2's callback (see loop.run_round); caps, if a list, collects the cap of every refit."""
    train, test, tm = split
    monkeypatch.setattr(lp, "PROMOTE_MIN_ADULTS", min_adults)  # these test the promote logic, not the real floor
    seq = list(flags) if isinstance(flags, list) else [flags]  # a list gives each refit its own flags, the last repeats
    def stub_refit(env, st, cap, margin=None, window=None):
        if caps is not None:
            caps.append(cap)
        if margins is not None:
            margins.append(margin)
        if windows is not None:
            windows.append(window)
        return candidate(seq.pop(0) if len(seq) > 1 else seq[0])

    monkeypatch.setattr(lp, "refit", stub_refit)
    env = lp.make_env(train, test, policy={**POL, "min_audit_adults": 1}, tm=tm,
                      timer=AgentTimer(tmp_path / "calls.jsonl", run="t"))
    st = lp.new_state(env.policy)
    st.candidate = candidate(start_flags)
    results = [lp.run_round(st, b, env, None, decide, apply_a2) for _, b in zip(range(n), env.oracle)]
    rounds = pd.DataFrame([r.row for r in results])
    rounds.attrs["records"] = [r.record for r in results]
    return rounds, st, env


def test_two_good_rounds_promote_and_the_candidate_goes_live(split, tmp_path, monkeypatch):
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 3)
    assert rounds["action"].tolist() == [lp.RETUNE, lp.PROMOTE, lp.RETUNE]
    assert rounds["mode"].tolist() == [lp.SHADOW, lp.ACTIVE, lp.ACTIVE]  # no way back
    assert st.live is st.candidate and st.live.model is not None
    assert rounds["t_soft"].iloc[0] != rounds["t_soft"].iloc[0]  # NaN: starter still live in round 1
    assert rounds["t_soft"].iloc[1:].eq(0.5).all() and rounds["t_verify"].iloc[1:].eq(0.9).all()


def test_promote_is_refused_for_a_candidate_with_insufficient_flags(split, tmp_path, monkeypatch):
    # round 1 builds the streak and refits a clean candidate, round 2 would promote but its refit is flagged
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 3, flags=[(), ("insufficient_adults",)])
    assert rounds["action"].tolist() == [lp.RETUNE] * 3
    assert (rounds["mode"] == lp.SHADOW).all() and st.live.model is None
    ev = [r["evidence"] for r in rounds.attrs["records"]]
    assert [e["promote_refused"] for e in ev] == [False, True, False]
    assert [e["streak"] for e in ev] == [1, 2, 0]  # round 3 cannot count: its evidence candidate is the flagged one


def test_a_soft_capped_candidate_can_still_promote(split, tmp_path, monkeypatch):
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 2, flags=("soft_capped",))
    assert rounds["mode"].tolist() == [lp.SHADOW, lp.ACTIVE]


def test_rounds_with_too_few_audit_adults_do_not_build_the_streak(split, tmp_path, monkeypatch):
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 4, min_adults=10**6)
    assert (rounds["mode"] == lp.SHADOW).all() and st.streak == 0


def test_a_failed_reveal_leaves_the_state_untouched(split, tmp_path, monkeypatch):
    _, st, env = run_rounds(split, tmp_path, monkeypatch, 1)
    seen, rnd = len(st.seen), st.round
    batch = next(iter(env.oracle))

    def boom(*a, **k):
        raise OracleError("nope")

    monkeypatch.setattr(env.oracle, "reveal", boom)
    with pytest.raises(OracleError):
        lp.run_round(st, batch, env)
    assert len(st.seen) == seen and st.round == rnd


def test_psi_compares_against_stored_scores_and_restarts_after_a_promote(split, tmp_path, monkeypatch):
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 4)
    assert rounds["action"].tolist()[:2] == [lp.RETUNE, lp.PROMOTE]
    psis = rounds["psi"].tolist()
    assert np.isnan(psis[0]) and psis[1] >= 0.0  # round 1 has no reference, round 2 has the starter's scores
    assert np.isnan(psis[2])  # promoted in round 2: no scores from the new live rule yet
    assert psis[3] == 0.0  # round 4 against round 3, both all-zero scores
    assert 0 < len(st.live_scores) < len(st.seen)  # only rounds 3 and 4 are kept after the restart


def test_records_carry_the_promote_evidence_and_no_test_metrics(full):
    _, _, _, records, _ = full
    assert "evidence" not in records[0] and all(valid_decision(r) for r in records)
    for r in records[1:]:
        ev = r["evidence"]
        assert set(ev) == {"round_audit_adults", "cand_ft", "cand_t_verify", "cand_unsafe", "pooled_adults", "pooled_ft", "streak",
                           "promote_refused", "policy_cap", "refit_cap", "cap_differs", "cand_age",
                           "policy_margin", "refit_margin", "margin_differs", "refit_window", "window_differs"}
        assert ev["round_audit_adults"] > 0 and ev["streak"] >= 0 and ev["promote_refused"] is False
    hold = [r["evidence"] for r in records[1:] if r["rule_decision"]["action"] == lp.HOLD]
    assert hold and all(e["cand_ft"] is None and e["cand_t_verify"] is None and e["streak"] == 0 for e in hold)


def test_evidence_shows_the_streak_building_and_a_refused_promote(split, tmp_path, monkeypatch):
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 3)
    ev = [r["evidence"] for r in rounds.attrs["records"]]
    assert [e["streak"] for e in ev] == [1, 2, 2] and all(e["cand_ft"] == 0.0 for e in ev)
    assert ev[0]["cand_t_verify"] == 0.9 and not any(e["promote_refused"] for e in ev)
    rounds, _, _ = run_rounds(split, tmp_path / "x", monkeypatch, 3, flags=[(), ("insufficient_teens",)])
    assert [e["promote_refused"] for e in (r["evidence"] for r in rounds.attrs["records"])] == [False, True, False]


def test_a_failure_after_the_reveal_restores_the_state(split, tmp_path, monkeypatch):
    _, st, env = run_rounds(split, tmp_path, monkeypatch, 1)
    before = {f.name: getattr(st, f.name) for f in dataclasses.fields(st)}
    seen_rows = len(st.seen)

    def boom(*a, **k):
        raise RuntimeError("refit failed")

    monkeypatch.setattr(lp, "refit", boom)
    with pytest.raises(RuntimeError):
        lp.run_round(st, next(iter(env.oracle)), env)
    after = {f.name: getattr(st, f.name) for f in dataclasses.fields(st)}
    assert after.keys() == before.keys() and len(st.seen) == seen_rows
    assert all(after[k] is before[k] for k in before)  # same objects: streak 1, candidate, live_scores, ...


def test_a_candidate_with_insufficient_flags_does_not_build_the_streak(split, tmp_path, monkeypatch):
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 3, start_flags=("insufficient_adults",))
    ev = [r["evidence"] for r in rounds.attrs["records"]]
    assert [e["cand_unsafe"] for e in ev] == [True, False, False]
    assert [e["streak"] for e in ev] == [0, 1, 2]  # the first round is lost, then it builds as usual
    assert rounds["mode"].tolist() == [lp.SHADOW, lp.SHADOW, lp.ACTIVE]


def test_write_run_cuts_a_torn_last_line_and_starts_on_its_own_line(tmp_path):
    # an unterminated last line is what a killed writer leaves; readers already skip it, and completing it with a
    # newline would turn it into a malformed row, so write_run cuts it back to the last newline first
    rounds = pd.DataFrame([{c: 0 for c in ROUNDS_COLS}])
    recs = [lp.make_record("r", 0, {"cutoff": 0.5, "cap": 0.15, "action": "starter"}, "starter")]
    r_path, d_path = tmp_path / "rounds.csv", tmp_path / "decisions.jsonl"
    lp.write_run(rounds, recs, r_path, d_path)
    r_path.write_text(r_path.read_text() + "run,1,SHAD", encoding="utf-8")  # torn mid-append
    d_path.write_text(d_path.read_text() + '{"run": "r", "rou', encoding="utf-8")
    lp.write_run(rounds, recs, r_path, d_path)
    assert len(pd.read_csv(r_path)) == 2 and r_path.read_text().endswith("\n")
    assert [json.loads(x)["round"] for x in d_path.read_text().splitlines()] == [0, 0]


def test_real_rounds_only_promote_on_enough_pooled_adults(full):
    _, _, rounds, records, _ = full
    for r in records[1:]:
        ev = r["evidence"]
        if ev["pooled_adults"] is not None and ev["pooled_adults"] < lp.PROMOTE_MIN_ADULTS:
            assert ev["streak"] < lp.PROMOTE_STREAK  # a full window on too few adults cannot pass
    assert rounds["action"].ne(lp.PROMOTE).all() or rounds["mode"].eq(lp.ACTIVE).any()


@pytest.mark.parametrize("window,passes", [
    ([(30, 9), (30, 1)], True),  # one round alone is 0.30 (over 0.18), pooled 10/60 = 0.167
    ([(30, 12), (30, 12)], False),  # pooled 0.40
    ([(30, 5), (30, 5)], True),  # 10/60 = 0.167
    ([(30, 6), (30, 5)], False),  # 11/60 = 0.183, just over cap + slack (0.18)
    ([(20, 0), (19, 0)], False),  # 39 pooled adults, under the floor of 40
    ([(20, 0), (20, 0)], True),  # exactly the floor
    ([], False),
])
def test_pooled_test(window, passes):
    assert lp.pooled_test(window, 0.15)[2] is passes


def test_pooled_test_counts_adults_and_false_teens():
    assert lp.pooled_test([(27, 3), (30, 6)], 0.15)[:2] == (57, 9)


# ---- A2 inside the round (step 15) ----
def a2_says(action, cap, cap_margin=None, refit_window=None):
    block = {"status": "LIVE", "fallback_reason": None,
             "output": {"action": action, "cap": cap, "cap_margin": cap_margin, "refit_window": refit_window,
                        "reason": "test", "cites": ["audit"]}}
    return lambda ctx, blocks: {"a2": block}


def test_the_diff_agrees_with_the_rule_decision_after_a_refused_promote(split, tmp_path, monkeypatch):
    # round 2 would promote but its refit is flagged, so it becomes a re-tune; A2 (not applied) said re-tune too
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 3, flags=[(), ("insufficient_adults",)],
                              decide=a2_says("re-tune", 0.15))
    rec = rounds.attrs["records"][1]
    assert rec["rule_decision"]["action"] == lp.RETUNE and rec["evidence"]["promote_refused"] is True
    assert rec["diff"] == {} and rounds.loc[1, "diff_count"] == 0


def test_a_promote_refits_at_the_cap_the_evidence_candidate_was_refit_at(split, tmp_path, monkeypatch):
    caps = []

    def decide(ctx, blocks):  # re-tune at 0.10 while the rule's gate is filling, then ask to promote at 0.25
        if ctx.guards["promote_allowed"]:
            return a2_says("promote", 0.25)(ctx, blocks)
        return a2_says("re-tune", 0.10)(ctx, blocks)

    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 2, decide=decide, apply_a2=True, caps=caps)
    ev = [r["evidence"] for r in rounds.attrs["records"]]
    assert rounds["action"].tolist() == [lp.RETUNE, lp.PROMOTE] and caps == [0.10, 0.10]
    assert ev[1]["refit_cap"] == 0.10 and ev[1]["cap_differs"] is True and ev[1]["policy_cap"] == 0.15
    assert rounds.attrs["records"][1]["a2"]["output"]["cap"] == 0.25  # A2's own ask is still logged
    assert st.mode == lp.ACTIVE and st.candidate_cap == 0.10


def test_a_promote_with_no_recorded_candidate_cap_uses_the_policy_cap(split, tmp_path, monkeypatch):
    caps = []  # the helper puts a candidate in place without a refit, so no cap was recorded for it
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 2, caps=caps)
    assert rounds["action"].tolist() == [lp.RETUNE, lp.PROMOTE] and caps == [0.15, 0.15]


def test_no_context_is_built_without_a_decider(split, tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "_context", lambda *a, **kw: pytest.fail("a rule-only round must not build A2's context"))
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 1)
    assert rounds["applied_source"].tolist() == ["rule"]


def test_a_failing_context_is_logged_and_the_rule_decides(split, tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise ValueError("bad field")

    monkeypatch.setattr(lp, "_context", boom)
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 1, decide=a2_says("hold", 0.15), apply_a2=True)
    rec = rounds.attrs["records"][0]
    assert rec["agent_error"].startswith("A2 ValueError: bad field") and rec["applied"]["source"] == "rule"
    assert rec["a2"]["status"] is None and rounds.loc[0, "diff_count"] == 0


@pytest.mark.parametrize("mode", ["rule", "crew"])
def test_a_stale_header_fails_before_any_round_runs(tmp_path, monkeypatch, mode):
    old = tmp_path / "rounds.csv"
    old.write_text(",".join(c for c in ROUNDS_COLS if c != "diff_count") + "\n")
    monkeypatch.setattr(lp, "ROUNDS_CSV", old)
    monkeypatch.setattr(lp, "load_data", lambda **kw: pytest.fail("the data must not load before the header check"))
    monkeypatch.setattr("sys.argv", ["loop", "--mode", mode])
    with pytest.raises(ValueError, match="older schema"):
        lp.main()
    monkeypatch.setattr("sys.argv", ["loop", "--mode", mode, "--no-write"])
    with pytest.raises(pytest.fail.Exception, match="must not load"):  # --no-write skips the check and goes on to run
        lp.main()


# ---- A2's cap_margin (step 2) ----
def test_a2_margin_reaches_the_refit_clamped_and_is_logged(split, tmp_path, monkeypatch):
    margins = []
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 1, decide=a2_says("re-tune", 0.15, 0.4),
                               apply_a2=True, margins=margins)
    rec = rounds.attrs["records"][0]
    assert margins == [lp.MARGIN_MAX] and st.candidate_margin == lp.MARGIN_MAX  # 0.4 clamped in code
    assert rec["evidence"]["refit_margin"] == lp.MARGIN_MAX and rec["evidence"]["margin_differs"] is True
    assert rec["applied"]["decision"]["cap_margin"] == lp.MARGIN_MAX
    assert rec["diff"]["cap_margin"] == [POL["cap_margin"], 0.4]  # A2's own ask against the policy value


def test_a_null_margin_keeps_the_policy_value_and_no_diff(split, tmp_path, monkeypatch):
    margins = []
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 1, decide=a2_says("re-tune", 0.15),
                              apply_a2=True, margins=margins)
    rec = rounds.attrs["records"][0]
    assert margins == [POL["cap_margin"]] and rec["evidence"]["margin_differs"] is False
    assert "cap_margin" not in rec["diff"]


def test_the_rule_mode_never_uses_a2_margin(split, tmp_path, monkeypatch):
    margins = []
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 1, decide=a2_says("re-tune", 0.15, 0.04),
                              apply_a2=False, margins=margins)
    assert margins == [POL["cap_margin"]] and rounds.attrs["records"][0]["a2"]["output"]["cap_margin"] == 0.04


def test_a_promote_keeps_the_margin_the_evidence_candidate_was_refit_at(split, tmp_path, monkeypatch):
    margins = []

    def decide(ctx, blocks):
        if ctx.guards["promote_allowed"]:
            return a2_says("promote", 0.15, 0.01)(ctx, blocks)
        return a2_says("re-tune", 0.15, 0.03)(ctx, blocks)

    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 2, decide=decide, apply_a2=True, margins=margins)
    assert rounds["action"].tolist() == [lp.RETUNE, lp.PROMOTE] and margins == [0.03, 0.03]
    assert st.mode == lp.ACTIVE and st.candidate_margin == 0.03


def test_clamp_margin_never_exceeds_the_cap():
    assert lp.clamp_margin(0.2, 0.08) == lp.MARGIN_MAX and lp.clamp_margin(-1, 0.15) == 0.0
    assert lp.clamp_margin(0.5, 0.03) == 0.03


# ---- A2's refit_window (step 3) ----
def test_a2_window_reaches_the_refit_and_is_logged(split, tmp_path, monkeypatch):
    windows = []
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 4, decide=a2_says("re-tune", 0.15, refit_window=2),
                               apply_a2=True, windows=windows)
    ev = [r["evidence"] for r in rounds.attrs["records"]]
    assert windows[0] is None and windows[-1] == 2  # round 1 has no earlier round to drop: all rounds
    assert ev[-1]["refit_window"] == 2 and ev[-1]["window_differs"] is True
    assert rounds.attrs["records"][-1]["applied"]["decision"]["refit_window"] == 2
    assert st.candidate_window == 2


def test_a_window_of_one_is_clamped_to_the_minimum(split, tmp_path, monkeypatch):
    windows = []
    run_rounds(split, tmp_path, monkeypatch, 4, decide=a2_says("re-tune", 0.15, refit_window=1), apply_a2=True, windows=windows)
    assert windows[-1] == lp.WINDOW_MIN


def test_rule_mode_never_uses_the_window(split, tmp_path, monkeypatch):
    windows = []
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 4, decide=a2_says("re-tune", 0.15, refit_window=2), windows=windows)
    assert set(windows) == {None} and rounds.attrs["records"][-1]["evidence"]["window_differs"] is False


def test_a_promote_keeps_the_window_the_evidence_candidate_was_refit_at(split, tmp_path, monkeypatch):
    windows = []

    def decide(ctx, blocks):
        if ctx.guards["promote_allowed"]:
            return a2_says("promote", 0.15, refit_window=5)(ctx, blocks)
        return a2_says("re-tune", 0.15, refit_window=2)(ctx, blocks)

    # the first refit is flagged, so the gate only passes after a few rounds, when a window of 2 is already in use
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 6, flags=[("insufficient_adults",), ()],
                               decide=decide, apply_a2=True, windows=windows)
    promoted = list(rounds.index[rounds["action"] == lp.PROMOTE])
    assert len(promoted) == 1 and promoted[0] >= 3
    assert windows[promoted[0]] == 2 and 5 not in windows  # the tested window, not A2's 5


def test_effective_window_falls_back_to_all_rounds(split, tmp_path, monkeypatch):
    _, st, env = run_rounds(split, tmp_path, monkeypatch, 4)
    assert lp.effective_window(env, st, None) is None
    assert lp.effective_window(env, st, st.round) is None  # covers every round
    assert lp.effective_window(env, st, 2) == 2
    monkeypatch.setattr(lp, "MIN_WINDOW_ROWS", 10**6)  # too few rows in the window
    assert lp.effective_window(env, st, 2) is None


def test_labelled_rows_keep_only_the_last_rounds(split, tmp_path, monkeypatch):
    _, st, env = run_rounds(split, tmp_path, monkeypatch, 4)
    all_rows, last2 = lp.labelled_rows(env, st), lp.labelled_rows(env, st, 2)
    rev = env.oracle.revealed("all")
    assert len(last2) == int((rev["round"] > st.round - 2).sum()) and len(last2) < len(all_rows)



# ---- the challenger (policy.yaml promote_rule: challenger) ----
@pytest.mark.parametrize("live,cand,adults,wins", [
    ((0.80, 0.356), (0.88, 0.17), 30, True),    # the live rule is over the bar, the challenger is not
    ((0.80, 0.356), (0.95, 0.30), 30, False),   # both over the bar: no win, whatever the recall
    ((0.88, 0.15), (0.88, 0.12), 30, True),     # both within: as many teens caught is enough
    ((0.90, 0.15), (0.88, 0.12), 30, False),    # both within: fewer teens caught loses
    ((0.80, 0.356), (0.88, 0.17), 19, False),   # too few audit adults in the round to decide
    ((0.80, 0.356), (None, 0.10), 30, False),   # no audit teen to measure recall on
])
def test_challenger_wins(live, cand, adults, wins):
    assert lp.challenger_wins(live, cand, 0.15, adults) is wins


def test_audit_rates():
    y, s = np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1])
    assert lp.audit_rates(y, s, 0.5) == (0.5, 0.5)
    assert lp.audit_rates(np.array([0, 0]), np.array([0.9, 0.1]), 0.5) == (None, 0.5)


def test_one_won_round_promotes_under_the_challenger_rule():
    pol = {**POL, "promote_rule": "challenger", "min_audit_adults": 50}
    assert lp.rule_decision(state(lp.SHADOW, 1), pol, 60)["action"] == lp.PROMOTE
    assert lp.rule_decision(state(lp.SHADOW, 0), pol, 60)["action"] == lp.RETUNE
    assert lp.rule_decision(state(lp.SHADOW, 1), pol, 49)["action"] == lp.HOLD  # the floor still comes first
    assert lp.rule_decision(state(lp.SHADOW, 1), POL, 130)["action"] == lp.RETUNE  # pooled needs two rounds
