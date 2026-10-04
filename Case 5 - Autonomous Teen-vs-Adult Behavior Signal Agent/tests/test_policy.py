"""Review policy (step 9): policy.yaml loading, thresholds at a cap, bands, edge cases, audit slice.

Small tests use synthetic scores. The real-data tests read the committed results/split.json and
cache/stack_oof.csv and skip when that cache is missing (it is gitignored).
"""
import numpy as np
import pandas as pd
import pytest

import softsignal.policy as pol
from softsignal.data import load_data
from softsignal.features import ID_COL, TARGET
from softsignal.metrics import prf
from softsignal.stack import STACK_OOF


def scores_and_labels(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    s = np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    return s, y


def write_policy(tmp_path, **over):
    vals = {"cap_false_teen": 0.15, "review_budget": 0.25, "soft_recall": 0.9, "min_audit_adults": 120,
            "audit_per_batch": 60, **over}
    p = tmp_path / "policy.yaml"
    p.write_text("\n".join(f"{k}: {v}" for k, v in vals.items()), encoding="utf-8")
    return p


# ---- policy.yaml ----
def test_committed_policy_has_the_five_keys():
    p = pol.load_policy()
    assert p == {"cap_false_teen": 0.15, "review_budget": 0.25, "soft_recall": 0.90,
                 "min_audit_adults": 120, "audit_per_batch": 60}


def test_load_policy_rejects_unknown_and_missing_keys(tmp_path):
    p = write_policy(tmp_path)
    p.write_text(p.read_text() + "\nextra: 1", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        pol.load_policy(p)
    p.write_text("cap_false_teen: 0.15", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        pol.load_policy(p)


@pytest.mark.parametrize("over", [{"cap_false_teen": 1.5}, {"soft_recall": -0.1}, {"min_audit_adults": 0},
                                  {"audit_per_batch": 2.5}, {"cap_false_teen": "x"}, {"review_budget": "true"}])
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


def test_soft_threshold_is_lowered_to_verify_when_it_would_sit_above():
    # teens all score above adults: the 90% teen recall point sits above the 15% cap cutoff
    s = np.r_[np.linspace(0.0, 0.5, 100), np.linspace(0.6, 1.0, 100)]
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    th = pol.pick_thresholds(s, y, cap=0.15, soft_recall=0.9)
    assert "soft_capped" in th.flags and th.t_soft == th.t_verify


def test_fewer_than_five_adults_returns_prior_with_flag():
    s, y = scores_and_labels()
    prior = pol.pick_thresholds(s, y)
    keep = np.r_[np.where(y == 1)[0], np.where(y == 0)[0][:4]]
    th = pol.pick_thresholds(s[keep], y[keep], cap=0.10, prior=prior)
    assert th.flags == ("insufficient_adults",)
    assert (th.t_verify, th.t_soft) == (prior.t_verify, prior.t_soft)
    assert th.cap == 0.10 and th.n_adults == 4


def test_fewer_than_five_adults_without_prior_raises():
    s, y = scores_and_labels()
    keep = np.r_[np.where(y == 1)[0], np.where(y == 0)[0][:4]]
    with pytest.raises(ValueError, match="no prior"):
        pol.pick_thresholds(s[keep], y[keep])


def test_exactly_five_adults_is_enough():
    s, y = scores_and_labels()
    keep = np.r_[np.where(y == 1)[0], np.where(y == 0)[0][:5]]
    assert "insufficient_adults" not in pol.pick_thresholds(s[keep], y[keep]).flags


def test_fewer_than_five_teens_keeps_prior_soft_threshold():
    s, y = scores_and_labels()
    prior = pol.pick_thresholds(s, y)
    keep = np.r_[np.where(y == 0)[0], np.where(y == 1)[0][:3]]
    th = pol.pick_thresholds(s[keep], y[keep], prior=prior)
    assert "insufficient_teens" in th.flags and th.t_soft == min(prior.t_soft, th.t_verify)


def test_fewer_than_five_teens_without_prior_uses_verify():
    s, y = scores_and_labels()
    keep = np.r_[np.where(y == 0)[0], np.where(y == 1)[0][:3]]
    th = pol.pick_thresholds(s[keep], y[keep])
    assert "insufficient_teens" in th.flags and th.t_soft == th.t_verify


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
def make_oof(tmp_path, ids):
    p = tmp_path / "oof.csv"
    pd.DataFrame({ID_COL: ids, "stack_oof": np.linspace(0, 1, len(ids))}).to_csv(p, index=False)
    return p


def test_audit_slice_joins_labels_by_id(tmp_path):
    train = pd.DataFrame({ID_COL: ["B1", "B2", "B3"], TARGET: [0, 1, 0]})
    got = pol.load_audit_slice(train, make_oof(tmp_path, ["B3", "B1", "B2"]))  # order differs
    assert got.set_index(ID_COL).loc["B2", TARGET] == 1
    assert got.set_index(ID_COL).loc["B3", "stack_oof"] == 0.0
    assert len(got) == 3


@pytest.mark.parametrize("ids", [["B1", "B2"], ["B1", "B2", "B3", "B4"], ["B1", "B2", "B2"]])
def test_audit_slice_rejects_stale_or_duplicate_cache(tmp_path, ids):
    train = pd.DataFrame({ID_COL: ["B1", "B2", "B3"], TARGET: [0, 1, 0]})
    with pytest.raises(ValueError, match="rerun"):
        pol.load_audit_slice(train, make_oof(tmp_path, ids))


# ---- real data ----
@pytest.mark.skipif(not STACK_OOF.exists(), reason="cache/stack_oof.csv not built (python -m softsignal.stack)")
@pytest.mark.parametrize("cap", [0.05, 0.10, 0.15])
def test_real_oof_false_teen_within_cap(cap):
    train, _ = load_data(on_param_mismatch="error")
    audit = pol.load_audit_slice(train)
    assert len(audit) == 2100
    th = pol.pick_thresholds(audit["stack_oof"], audit[TARGET], cap=cap)
    bands = pol.assign_bands(audit["stack_oof"], th)
    out = pol.band_summary(bands, 0.25, audit[TARGET])
    assert out["ft_verify"] <= cap
    assert out["rec_soft_up"] >= 0.9
    assert th.t_soft <= th.t_verify
