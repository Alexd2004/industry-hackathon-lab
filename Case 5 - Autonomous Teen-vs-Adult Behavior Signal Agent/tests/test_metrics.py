"""Shared metrics: hand-checked confusion counts, zero-division and eval.csv rows."""
import math

import numpy as np
import pytest

from softsignal.metrics import EVAL_COLS, auc, confusion, eval_row, f1, prf

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
