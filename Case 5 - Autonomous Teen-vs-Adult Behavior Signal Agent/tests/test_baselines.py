"""Keyword baseline (step 2) and tabular LR fallback (step 6) on the committed split.

Pinned numbers depend on the committed results/split.json and the library versions;
they were measured with scikit-learn 1.9.1 and numpy 2.5.3.
"""
import numpy as np
import pytest

from softsignal.baselines import cap_label, keyword_baseline, nested_fold, oof_scores, tabular_lr
from softsignal.data import cv_folds, load_data
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


@pytest.mark.parametrize("cap,label", [(0.15, "15"), (0.145, "14.5"), (0.05, "5"), (1.0, "100")])
def test_cap_label_keeps_fractional_cap(cap, label):
    assert cap_label(cap) == label


def test_model_uses_only_allowed_columns(lr):
    assert list(lr.model.feature_names_in_) == FEATURE_COLS


def test_oof_false_teen_holds_cap(lr, split):
    assert not np.isnan(lr.oof).any()
    _, _, ft, _ = prf(split[0][TARGET], (lr.oof >= lr.threshold).astype(int))
    assert ft <= lr.cap


def test_rows_are_nested_cv_then_test(lr):
    assert [r["eval_set"] for r in lr.rows] == ["nested_cv", "test"]
    assert {r["stage"] for r in lr.rows} == {"tabular_lr_cap15"}


def test_nested_cv_row_is_sane(lr):
    row = lr.rows[0]
    assert not any(np.isnan(row[k]) for k in ("prec", "rec", "ft", "mt", "f1", "auc"))
    assert 0.10 <= row["ft"] <= 0.20  # honest estimate, so it may land a bit over the cap


def test_nested_fold_ignores_its_own_labels(split):
    train = split[0]
    fit_idx, val_idx = cv_folds(train)[0]
    flipped = train.copy()
    flipped.iloc[val_idx, flipped.columns.get_loc(TARGET)] = 1 - train[TARGET].iloc[val_idx]
    s1, p1 = nested_fold(train, fit_idx, val_idx)
    s2, p2 = nested_fold(flipped, fit_idx, val_idx)
    assert np.array_equal(s1, s2) and np.array_equal(p1, p2)


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
