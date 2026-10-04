"""Review policy (step 9): policy.yaml loading, thresholds at a cap, bands, edge cases, audit slice.

Small tests use synthetic scores. The real-data tests read the committed results/split.json and
cache/stack_oof.csv. That cache is gitignored, so they skip when it is missing, unless the environment
variable REQUIRE_STACK_CACHE is set (use it in CI), which turns the skip into a failure.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import softsignal.policy as pol
from softsignal.data import load_data
from softsignal.features import FEATURE_COLS, ID_COL, TARGET
from softsignal.metrics import prf
from softsignal.stack import STACK_OOF, csv_sha256, meta_path, write_oof


def scores_and_labels(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    s = np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    return s, y


def write_policy(tmp_path, **over):
    vals = {"cap_false_teen": 0.15, "review_budget": 0.25, "soft_recall": 0.9, "min_audit_adults": 120,
            "audit_per_batch": 60, "cap_margin": 0.0, **over}
    p = tmp_path / "policy.yaml"
    p.write_text("\n".join(f"{k}: {v}" for k, v in vals.items()), encoding="utf-8")
    return p


# ---- policy.yaml ----
def test_committed_policy_has_exactly_the_policy_keys_with_their_types():
    p = pol.load_policy()
    assert set(p) == set(pol.POLICY_KEYS) | set(pol.OPTIONAL_KEYS)
    assert all(type(p[k]) is kind for k, kind in pol.POLICY_KEYS.items())
    assert p["psi_drift"] == 0.25  # measured for A1, see policy.yaml
    assert p["min_a3_errors"] is None  # not measured yet: A3 stays insufficient_data until it is set


def test_load_policy_rejects_unknown_and_missing_keys(tmp_path):
    p = write_policy(tmp_path)
    p.write_text(p.read_text() + "\nextra: 1", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        pol.load_policy(p)
    p.write_text("cap_false_teen: 0.15", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        pol.load_policy(p)


@pytest.mark.parametrize("over", [{"cap_false_teen": 1.5}, {"soft_recall": -0.1}, {"min_audit_adults": 0},
                                  {"audit_per_batch": 2.5}, {"cap_false_teen": "x"}, {"review_budget": "true"},
                                  {"min_audit_adults": ".inf"}, {"cap_false_teen": ".nan"}, {"cap_margin": 0.5},
                                  {"cap_margin": -0.01}, {"psi_drift": 0}, {"psi_drift": "x"},
                                  {"min_a3_errors": 0}, {"min_a3_errors": 2.5}, {"min_a3_errors": "x"}])
def test_load_policy_rejects_bad_values(tmp_path, over):
    with pytest.raises(ValueError):
        pol.load_policy(write_policy(tmp_path, **over))


def test_load_policy_rejects_non_mapping(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("- 1\n- 2", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        pol.load_policy(p)


# ---- pick_thresholds ----
@pytest.mark.parametrize("cap", [0.05, 0.10, 0.15, 0.30])
def test_verify_threshold_keeps_false_teen_within_cap(cap):
    s, y = scores_and_labels()
    th = pol.pick_thresholds(s, y, cap=cap)
    _, _, ft, _ = prf(y, (s >= th.t_verify).astype(int))
    assert ft <= cap


def test_margin_aims_below_the_cap():
    s, y = scores_and_labels()
    plain = pol.pick_thresholds(s, y, cap=0.15)
    tight = pol.pick_thresholds(s, y, cap=0.15, margin=0.05)
    assert tight.t_verify >= plain.t_verify
    _, _, ft, _ = prf(y, (s >= tight.t_verify).astype(int))
    assert ft <= 0.10


def test_soft_threshold_reaches_soft_recall():
    s, y = scores_and_labels()
    th = pol.pick_thresholds(s, y, cap=0.05, soft_recall=0.9)
    _, rec, _, _ = prf(y, (s >= th.t_soft).astype(int))
    assert rec >= 0.9
    assert th.t_soft <= th.t_verify


@pytest.mark.parametrize("n_teens", [23, 57, 100, 101])
def test_soft_threshold_reaches_soft_recall_for_awkward_teen_counts(n_teens):
    # with 23 or 57 teens the non-conservative quantile methods fall below the target, "lower" does not
    rng = np.random.default_rng(n_teens)
    s = np.r_[rng.beta(2, 5, 100), rng.beta(5, 2, n_teens)]
    y = np.r_[np.zeros(100), np.ones(n_teens)].astype(int)
    target = pol.load_policy()["soft_recall"]
    th = pol.pick_thresholds(s, y, cap=0.05, soft_recall=target)
    _, rec, _, _ = prf(y, (s >= th.t_soft).astype(int))
    assert rec >= target


def test_tied_scores_keep_false_teen_within_cap():
    s = np.round(scores_and_labels()[0], 1)  # heavy ties: about 10 distinct values
    y = scores_and_labels()[1]
    for cap in (0.05, 0.15, 0.30):
        th = pol.pick_thresholds(s, y, cap=cap)
        _, _, ft, _ = prf(y, (s >= th.t_verify).astype(int))
        assert ft <= cap


def test_all_adults_tied_pushes_verify_just_above_the_tie():
    s = np.r_[np.full(50, 0.5), np.linspace(0.4, 1.0, 50)]
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    th = pol.pick_thresholds(s, y, cap=0.15)
    assert th.t_verify > 0.5  # cannot flag some of the tied adults without flagging all of them
    assert pol.SOFT_CAPPED not in th.flags or th.t_soft == th.t_verify


def test_soft_threshold_is_lowered_to_verify_when_it_would_sit_above():
    # teens all score above adults: the 90% teen recall point sits above the 15% cap cutoff
    s = np.r_[np.linspace(0.0, 0.5, 100), np.linspace(0.6, 1.0, 100)]
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    th = pol.pick_thresholds(s, y, cap=0.15, soft_recall=0.9)
    assert pol.SOFT_CAPPED in th.flags and th.t_soft == th.t_verify
    _, rec, _, _ = prf(y, (s >= th.t_soft).astype(int))
    assert rec >= 0.9  # lowering t_soft to t_verify keeps the recall target


def test_fewer_than_five_adults_returns_prior_with_flag():
    s, y = scores_and_labels()
    prior = pol.pick_thresholds(s, y)
    keep = np.r_[np.where(y == 1)[0], np.where(y == 0)[0][:4]]
    th = pol.pick_thresholds(s[keep], y[keep], cap=0.10, prior=prior)
    assert th.flags == (pol.INSUFFICIENT_ADULTS,)
    assert (th.t_verify, th.t_soft) == (prior.t_verify, prior.t_soft)
    # the record keeps the cap and margin its cutoffs were picked for, with the new counts
    assert (th.cap, th.margin) == (prior.cap, prior.margin)
    assert (th.n_adults, th.n_teens) == (4, int((y == 1).sum()))


def test_fewer_than_five_adults_without_prior_raises():
    s, y = scores_and_labels()
    keep = np.r_[np.where(y == 1)[0], np.where(y == 0)[0][:4]]
    with pytest.raises(ValueError, match="no prior"):
        pol.pick_thresholds(s[keep], y[keep])


def test_exactly_five_adults_is_enough():
    s, y = scores_and_labels()
    keep = np.r_[np.where(y == 1)[0], np.where(y == 0)[0][:5]]
    assert pol.INSUFFICIENT_ADULTS not in pol.pick_thresholds(s[keep], y[keep]).flags


def test_fewer_than_five_teens_keeps_prior_soft_threshold():
    s, y = scores_and_labels()
    prior = pol.pick_thresholds(s, y)
    keep = np.r_[np.where(y == 0)[0], np.where(y == 1)[0][:3]]
    th = pol.pick_thresholds(s[keep], y[keep], prior=prior)
    assert pol.INSUFFICIENT_TEENS in th.flags and th.t_soft == min(prior.t_soft, th.t_verify)


def test_fewer_than_five_teens_with_a_high_prior_soft_gives_both_flags_and_no_recall_guarantee():
    s, y = scores_and_labels()
    high = pol.Thresholds(t_verify=0.9, t_soft=0.95, cap=0.15, margin=0.0, n_adults=100, n_teens=100)
    keep = np.r_[np.where(y == 0)[0], np.where(y == 1)[0][:3]]
    th = pol.pick_thresholds(s[keep], y[keep], prior=high)
    assert th.flags == (pol.INSUFFICIENT_TEENS, pol.SOFT_CAPPED)
    assert th.t_soft == th.t_verify  # the soft band is empty, and its recall was not picked from these scores


def test_fewer_than_five_teens_without_prior_uses_verify():
    s, y = scores_and_labels()
    keep = np.r_[np.where(y == 0)[0], np.where(y == 1)[0][:3]]
    th = pol.pick_thresholds(s[keep], y[keep])
    assert pol.INSUFFICIENT_TEENS in th.flags and th.t_soft == th.t_verify


@pytest.mark.parametrize("kw", [{"cap": 1.2}, {"cap": 0.1, "margin": 0.2}, {"margin": -0.1}, {"soft_recall": 0.0},
                                {"soft_recall": 1.5}])
def test_pick_thresholds_rejects_bad_arguments(kw):
    s, y = scores_and_labels()
    with pytest.raises(ValueError):
        pol.pick_thresholds(s, y, **kw)


def test_pick_thresholds_rejects_nan_and_shape_mismatch():
    s, y = scores_and_labels()
    bad = s.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pol.pick_thresholds(bad, y)
    with pytest.raises(ValueError, match="shape"):
        pol.pick_thresholds(s[:-1], y)


def test_no_clamp_in_policy():
    # the 8-30% clamp lives in loop.py and A2; the ladder uses 5% and 10%
    s, y = scores_and_labels()
    assert pol.pick_thresholds(s, y, cap=0.05).cap == 0.05


# ---- bands ----
TH = pol.Thresholds(t_verify=0.7, t_soft=0.4, cap=0.15, margin=0.0, n_adults=100, n_teens=100)


def test_assign_bands_boundaries_are_inclusive_lower():
    got = pol.assign_bands([0.7, 0.699, 0.4, 0.399, 1.0, 0.0], TH)
    assert list(got) == ["verify", "soft", "soft", "none", "verify", "none"]


def test_new_account_without_a_score_goes_to_soft():
    assert list(pol.assign_bands([np.nan, 0.9, np.nan], TH)) == ["soft", "verify", "soft"]


def test_empty_batch_gives_empty_bands_and_zero_summary():
    bands = pol.assign_bands([], TH)
    assert len(bands) == 0
    out = pol.band_summary(bands, 0.25)
    assert out["n_verify"] == 0 and out["verify_share"] == 0.0 and out["budget_binding"] is False


def test_empty_verify_band_when_cutoff_above_every_score():
    th = pol.Thresholds(2.0, 0.4, 0.0, 0.0, 100, 100)
    bands = pol.assign_bands([0.1, 0.5, 0.9], th)
    assert "verify" not in set(bands)
    assert pol.band_summary(bands, 0.25)["n_verify"] == 0


def test_budget_binding_flags_but_never_truncates():
    bands = np.array(["verify"] * 6 + ["none"] * 4)
    out = pol.band_summary(bands, 0.25)
    assert out["n_verify"] == 6 and out["verify_share"] == 0.6 and out["budget_binding"] is True
    assert pol.band_summary(bands, 0.6)["budget_binding"] is False  # at the budget is not over it


def test_band_summary_with_labels():
    bands = np.array(["verify", "verify", "soft", "none"])
    out = pol.band_summary(bands, 0.5, y=[1, 0, 1, 0])
    assert out["rec_verify"] == 0.5 and out["ft_verify"] == 0.5
    assert out["rec_soft_up"] == 1.0 and out["ft_soft_up"] == 0.5


# ---- audit slice ----
def make_train(ids=("B1", "B2", "B3"), labels=(0, 1, 0)):
    train = pd.DataFrame({ID_COL: list(ids), TARGET: list(labels)})
    for c in FEATURE_COLS:
        train[c] = np.linspace(0.0, 1.0, len(train))
    return train


def reseal(p):
    """Point the meta file at the csv's current bytes, as if write_oof had written that csv."""
    meta = json.loads(meta_path(p).read_text())
    meta["csv_sha256"] = csv_sha256(p)
    meta_path(p).write_text(json.dumps(meta), encoding="utf-8")


def make_oof(tmp_path, train, ids=None, meta=True):
    """An OOF cache with its meta file (written by the real write_oof), ids optionally altered."""
    p = tmp_path / "oof.csv"
    write_oof(train, np.linspace(0, 1, len(train)), p)
    if ids is not None:
        pd.DataFrame({ID_COL: ids, "stack_oof": np.linspace(0, 1, len(ids))}).to_csv(p, index=False)
        reseal(p)  # so the id check, not the csv hash, is what rejects it
    if not meta:
        meta_path(p).unlink()
    return p


def test_audit_slice_joins_labels_by_id(tmp_path):
    train = make_train()
    p = make_oof(tmp_path, train)
    shuffled = pd.read_csv(p).iloc[[2, 0, 1]]  # file order differs from train order
    shuffled.to_csv(p, index=False)
    reseal(p)
    got = pol.load_audit_slice(train, p).set_index(ID_COL)
    assert got.loc["B2", TARGET] == 1
    assert got.loc["B3", "stack_oof"] == 1.0
    assert len(got) == 3


@pytest.mark.parametrize("ids", [["B1", "B2"], ["B1", "B2", "B3", "B4"], ["B1", "B2", "B2"]])
def test_audit_slice_rejects_wrong_ids_or_duplicates(tmp_path, ids):
    train = make_train()
    with pytest.raises(ValueError, match="rerun"):
        pol.load_audit_slice(train, make_oof(tmp_path, train, ids=ids))


def test_audit_slice_rejects_a_cache_without_a_sidecar(tmp_path):
    train = make_train()
    with pytest.raises(ValueError, match="cache key"):
        pol.load_audit_slice(train, make_oof(tmp_path, train, meta=False))


def test_audit_slice_rejects_a_cache_built_from_other_rows(tmp_path):
    # same ids, but a label or a feature changed after the cache was written
    train = make_train()
    p = make_oof(tmp_path, train)
    flipped = train.assign(**{TARGET: [1, 1, 0]})
    with pytest.raises(ValueError, match="cache key"):
        pol.load_audit_slice(flipped, p)
    moved = train.copy()
    moved.loc[0, FEATURE_COLS[0]] += 0.5
    with pytest.raises(ValueError, match="cache key"):
        pol.load_audit_slice(moved, p)


@pytest.mark.parametrize("bad", ["not json", "[]", "{}", '{"cache_key": "x", "use_text": true}'])
def test_audit_slice_rejects_a_damaged_sidecar(tmp_path, bad):
    train = make_train()
    p = make_oof(tmp_path, train)
    meta_path(p).write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="cache key"):
        pol.load_audit_slice(train, p)


def test_audit_slice_rejects_a_tabular_only_cache(tmp_path):
    train = make_train()
    p = tmp_path / "oof.csv"
    write_oof(train, np.linspace(0, 1, 3), p, use_text=False)
    with pytest.raises(ValueError, match="cache key"):
        pol.load_audit_slice(train, p)


def test_audit_slice_accepts_a_matching_cache(tmp_path):
    train = make_train()
    assert len(pol.load_audit_slice(train, make_oof(tmp_path, train))) == 3


def test_audit_slice_rejects_an_edited_csv_with_the_same_ids(tmp_path):
    train = make_train()
    p = make_oof(tmp_path, train)
    edited = pd.read_csv(p)
    edited["stack_oof"] = 1.0 - edited["stack_oof"]  # same ids, different scores, meta file untouched
    edited.to_csv(p, index=False)
    with pytest.raises(ValueError, match="cache key"):
        pol.load_audit_slice(train, p)


def test_audit_slice_rejects_a_meta_file_without_the_csv_hash(tmp_path):
    train = make_train()
    p = make_oof(tmp_path, train)
    meta = json.loads(meta_path(p).read_text())
    del meta["csv_sha256"]
    meta_path(p).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="cache key"):
        pol.load_audit_slice(train, p)


def test_audit_slice_rejects_a_meta_file_with_the_wrong_row_count(tmp_path):
    train = make_train()
    p = make_oof(tmp_path, train)
    meta = json.loads(meta_path(p).read_text())
    meta["n"] += 1
    meta_path(p).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="cache key"):
        pol.load_audit_slice(train, p)


# ---- caps reported by main ----
def test_caps_to_report_is_the_policy_cap_and_the_ladder_caps_once_each():
    assert pol.caps_to_report(0.15) == [0.15, 0.10, 0.05]
    assert pol.caps_to_report(0.10) == [0.10, 0.05]
    assert pol.caps_to_report(0.05) == [0.10, 0.05]
    assert pol.caps_to_report(0.20) == [0.20, 0.10, 0.05]


# ---- real data ----
@pytest.fixture
def stack_cache():
    if not STACK_OOF.exists():
        msg = "cache/stack_oof.csv not built (python -m softsignal.stack)"
        if os.environ.get("REQUIRE_STACK_CACHE"):
            pytest.fail(msg)
        pytest.skip(msg)


@pytest.mark.parametrize("cap", [0.05, 0.10, 0.15])
def test_real_oof_false_teen_within_cap(stack_cache, cap):
    train, _ = load_data(on_param_mismatch="error")
    audit = pol.load_audit_slice(train)
    assert len(audit) == len(train)
    target = pol.load_policy()["soft_recall"]
    th = pol.pick_thresholds(audit["stack_oof"], audit[TARGET], cap=cap, soft_recall=target)
    bands = pol.assign_bands(audit["stack_oof"], th)
    out = pol.band_summary(bands, 0.25, audit[TARGET])
    assert out["ft_verify"] <= cap
    assert out["rec_soft_up"] >= target
    assert th.t_soft <= th.t_verify


def test_malformed_yaml_raises_value_error(tmp_path):
    # one error type for every caller: the Results tab catches ValueError and keeps rendering
    bad = tmp_path / "policy.yaml"
    bad.write_text("cap_false_teen: [0.15\nreview_budget: 0.25\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        pol.load_policy(bad)
