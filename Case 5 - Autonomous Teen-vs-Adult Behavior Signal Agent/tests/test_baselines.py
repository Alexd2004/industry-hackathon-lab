"""Keyword baseline (step 2) and tabular LR fallback (step 6) on the committed split.

Pinned numbers depend on the committed results/split.json and the sklearn version.
"""
import numpy as np
import pytest

from softsignal.baselines import cap_threshold, keyword_baseline, oof_scores, tabular_lr
from softsignal.data import load_data
from softsignal.features import FEATURE_COLS, TARGET
from softsignal.metrics import prf


@pytest.fixture(scope="module")
def split():
    return load_data(on_param_mismatch="error")


@pytest.fixture(scope="module")
def lr(split):
    return tabular_lr(*split)


def test_keyword_baseline_on_test(split):
    row = keyword_baseline(split[1])
    got = [round(row[k], 3) for k in ("prec", "rec", "ft", "mt", "auc")]
    assert got == [0.595, 0.473, 0.322, 0.527, 0.576]


@pytest.mark.parametrize("cap", [0.0, 0.05, 0.15, 0.3, 1.0])
def test_cap_threshold_never_exceeds_cap(cap):
    rng = np.random.default_rng(0)
    scores = np.round(rng.random(1000), 2)  # many ties on purpose
    y = rng.integers(0, 2, 1000)
    t = cap_threshold(scores, y, cap)
    _, _, ft, _ = prf(y, (scores >= t).astype(int))
    assert ft <= cap


def test_cap_threshold_is_as_loose_as_allowed():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y = np.zeros(10, dtype=int)
    t = cap_threshold(scores, y, 0.2)
    assert (scores >= t).sum() == 2


@pytest.mark.parametrize(
    "n,cap",
    [(100, 0.29), (100, 0.07), (1000, 0.15), (1, 0.15), (7, 0.3), (100, 0.15 - 1e-11), (3, 1 / 3)],
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cap_threshold_is_tight_at_float_boundaries(n, cap, dtype):
    scores = np.linspace(0, 1, n).astype(dtype)
    y = np.zeros(n, dtype=int)
    flagged = int((scores >= cap_threshold(scores, y, cap)).sum())
    _, _, ft, _ = prf(y, (scores >= cap_threshold(scores, y, cap)).astype(int))
    assert ft <= cap
    assert flagged == n or (flagged + 1) / n > cap  # one more adult would break the cap


@pytest.mark.parametrize("cap", [0.15, 0.25])
def test_cap_threshold_float32_ties(cap):
    rng = np.random.default_rng(1)
    scores = rng.random(1000).astype(np.float32)
    scores[:300] = np.float32(0.5)  # big tie block straddling the cutoff
    y = np.zeros(1000, dtype=int)
    _, _, ft, _ = prf(y, (scores >= cap_threshold(scores, y, cap)).astype(int))
    assert ft <= cap


def test_cap_threshold_full_cap_is_finite_and_flags_everyone():
    scores = np.array([0.1, 0.4, 0.9])
    t = cap_threshold(scores, np.array([0, 0, 1]), 1.0)
    assert np.isfinite(t) and (scores >= t).all()


def test_cap_threshold_all_ties_flags_none_below_full_cap():
    scores = np.full(10, 0.5)
    t = cap_threshold(scores, np.zeros(10, dtype=int), 0.5)
    assert (scores >= t).sum() == 0


def test_cap_threshold_ignores_teen_scores():
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    y = np.array([0, 0, 1, 1])
    assert cap_threshold(scores, y, 0.5) == cap_threshold(scores[:2], y[:2], 0.5)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_cap_threshold_rejects_bad_cap(bad):
    with pytest.raises(ValueError, match="cap"):
        cap_threshold(np.array([0.5]), np.array([0]), bad)


@pytest.mark.parametrize("scores", [[np.nan, 0.5], [np.inf, 0.5]])
def test_cap_threshold_rejects_non_finite(scores):
    with pytest.raises(ValueError, match="finite"):
        cap_threshold(np.array(scores), np.array([0, 0]), 0.15)


def test_cap_threshold_rejects_no_adults_and_bad_labels():
    with pytest.raises(ValueError, match="no adults"):
        cap_threshold(np.array([0.5, 0.6]), np.array([1, 1]), 0.15)
    with pytest.raises(ValueError, match="0 or 1"):
        cap_threshold(np.array([0.5, 0.6]), np.array([2, 2]), 0.15)
    with pytest.raises(ValueError, match="shape"):
        cap_threshold(np.array([0.5, 0.6]), np.array([0]), 0.15)


def test_stage_name_keeps_fractional_cap(split):
    assert tabular_lr(*split, cap=0.145).rows[0]["stage"] == "tabular_lr_cap14.5"


def test_model_uses_only_allowed_columns(lr):
    assert list(lr.model.feature_names_in_) == FEATURE_COLS


def test_oof_false_teen_holds_cap(lr, split):
    row = next(r for r in lr.rows if r["eval_set"] == "cv_oof")
    assert row["ft"] <= lr.cap
    assert not np.isnan(lr.oof).any()


def test_test_row_matches_measured(lr):
    row = next(r for r in lr.rows if r["eval_set"] == "test")
    assert round(row["rec"], 2) == 0.83
    assert round(row["ft"], 2) == 0.17
    assert round(row["auc"], 3) == 0.885


def test_forbidden_columns_do_not_change_scores(split):
    train = split[0]
    noisy = train.copy()
    noisy["age"] = np.where(noisy[TARGET] == 1, 15, 40)  # would leak if used
    assert np.allclose(oof_scores(train), oof_scores(noisy))


def test_coefficient_signs_are_sensible(lr):
    coef = lr.coefficients()
    assert coef["avg_word_len"] < 0
    assert coef["slang_emoji_rate"] > 0
    assert coef["pct_active_school_hours"] < 0
