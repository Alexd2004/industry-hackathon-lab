"""Review loop (step 11, loop.py half): decisions, verify band, hold rule, label rules, determinism, files."""
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

POL = load_policy()


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


# ---- whole loop on the real split ----
def make(n_rounds, tmp_path, source="audit"):
    train, test = load_data(on_param_mismatch="error")
    env = lp.make_env(train, test, threshold_source=source, timer=AgentTimer(tmp_path / "calls.jsonl", run="t"))
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
    assert r["audit_ft"].notna().all() and r["psi"].iloc[1:].notna().all() and np.isnan(r["psi"].iloc[0])
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


def run_rounds(split, tmp_path, monkeypatch, n, flags=()):
    """n rounds with min_audit_adults=1 and a stubbed refit; the state starts SHADOW with a candidate in place."""
    train, test, tm = split
    monkeypatch.setattr(lp, "refit", lambda env, st, cap: candidate(flags))
    env = lp.make_env(train, test, policy={**POL, "min_audit_adults": 1}, tm=tm,
                      timer=AgentTimer(tmp_path / "calls.jsonl", run="t"))
    st = lp.new_state(env.policy)
    st.candidate = candidate()
    rows = [lp.run_round(st, b, env).row for _, b in zip(range(n), env.oracle)]
    return pd.DataFrame(rows), st, env


def test_two_good_rounds_promote_and_the_candidate_goes_live(split, tmp_path, monkeypatch):
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 3)
    assert rounds["action"].tolist() == [lp.RETUNE, lp.PROMOTE, lp.RETUNE]
    assert rounds["mode"].tolist() == [lp.SHADOW, lp.ACTIVE, lp.ACTIVE]  # no way back
    assert st.live is st.candidate and st.live.model is not None
    assert rounds["t_soft"].iloc[0] != rounds["t_soft"].iloc[0]  # NaN: starter still live in round 1
    assert rounds["t_soft"].iloc[1:].eq(0.5).all() and rounds["t_verify"].iloc[1:].eq(0.9).all()


def test_promote_is_refused_for_a_candidate_with_insufficient_flags(split, tmp_path, monkeypatch):
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 3, flags=("insufficient_adults",))
    assert rounds["action"].tolist() == [lp.RETUNE] * 3
    assert (rounds["mode"] == lp.SHADOW).all() and st.live.model is None


def test_a_soft_capped_candidate_can_still_promote(split, tmp_path, monkeypatch):
    rounds, _, _ = run_rounds(split, tmp_path, monkeypatch, 2, flags=("soft_capped",))
    assert rounds["mode"].tolist() == [lp.SHADOW, lp.ACTIVE]


def test_rounds_with_too_few_audit_adults_do_not_build_the_streak(split, tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "PROMOTE_MIN_ADULTS", 10**6)
    rounds, st, _ = run_rounds(split, tmp_path, monkeypatch, 4)
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
