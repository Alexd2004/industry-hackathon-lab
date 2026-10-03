"""Shared metrics: hand-checked confusion counts, zero-division and eval.csv rows."""
import math

import numpy as np
import pytest

from softsignal.metrics import EVAL_COLS, as_binary, auc, cap_threshold, confusion, eval_row, f1, prf

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
P = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])  # tp=3 fn=1 fp=1 tn=5


def test_confusion_counts():
    assert confusion(Y, P) == (3, 1, 1, 5)


def test_prf_hand_checked():
    prec, rec, ft, mt = prf(Y, P)
    assert (prec, rec, ft, mt) == pytest.approx((0.75, 0.75, 1 / 6, 0.25))


def test_prf_zero_division_is_zero():
    assert prf(np.array([0, 0]), np.array([0, 0])) == (0.0, 0.0, 0.0, 0.0)


def test_f1():
    assert f1(0.75, 0.75) == pytest.approx(0.75)
    assert f1(0.0, 0.0) == 0.0


def test_auc_of_binary_prediction_is_balanced_accuracy():
    _, rec, ft, _ = prf(Y, P)
    assert auc(Y, P) == pytest.approx((rec + 1 - ft) / 2)


def test_auc_single_class_is_nan():
    assert math.isnan(auc(np.array([1, 1]), np.array([0.2, 0.9])))


@pytest.mark.parametrize("bad", [[1, 2], [0.5, 1], [np.nan, 1]])
def test_rejects_non_binary(bad):
    with pytest.raises(ValueError, match="0 or 1"):
        confusion(np.array(bad), np.array([1, 1]))


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        confusion(np.array([0, 1]), np.array([0, 1, 1]))


def test_eval_row_has_frozen_columns_and_uses_score_for_auc():
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.65, 0.1, 0.2, 0.3, 0.4, 0.5])
    row = eval_row("s", "test", Y, P, score)
    assert list(row) == EVAL_COLS
    assert row["auc"] == pytest.approx(auc(Y, score))
    assert row["auc"] != pytest.approx(auc(Y, P))


def test_as_binary_requires_1d():
    with pytest.raises(ValueError, match="1-D"):
        as_binary(np.array([[0, 1], [1, 0]]))
    with pytest.raises(ValueError, match="1-D"):
        as_binary(np.int64(1))


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
    flags = (scores >= cap_threshold(scores, y, cap)).astype(int)
    flagged = int(flags.sum())
    _, _, ft, _ = prf(y, flags)
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


def test_cap_threshold_requires_1d_scores():
    with pytest.raises(ValueError, match="1-D"):
        cap_threshold(np.array([[0.1, 0.2]]), np.array([0, 0]), 0.15)


def test_cap_threshold_rejects_floats_wider_than_float64():
    scores = np.array([0.1, 0.5, 0.5, 0.9], dtype=np.longdouble)
    y = np.zeros(4, dtype=int)
    if scores.dtype.itemsize > 8:  # x86-64 Linux: 80-bit extended precision
        with pytest.raises(TypeError, match="float64"):
            cap_threshold(scores, y, 0.25)
    else:  # macOS arm64, Windows: longdouble is plain float64
        assert (scores >= cap_threshold(scores, y, 0.25)).sum() == 1
