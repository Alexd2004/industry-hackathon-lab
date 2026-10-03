import numpy as np
import pandas as pd
import pytest

from softsignal.data import load_data
from softsignal.metrics import EVAL_COLS, auc
from softsignal.tier1 import (
    ALT_BLEND_W, STARTER_CUT, STARTER_W, activity_score, blend, eval_point, evaluate_blend, style_score,
)


@pytest.fixture(scope="module")
def test_df():
    return load_data()[1]


def test_columns_and_row_count(test_df):
    rows = evaluate_blend(test_df, {"w": 0.5, "cutoff": 0.65})
    assert list(rows.columns) == EVAL_COLS
    assert len(rows) == 3
    assert set(rows["eval_set"]) == {"test"}


def test_no_cap_best_gives_two_rows(test_df):
    assert len(evaluate_blend(test_df, None)) == 2


def test_starter_row_matches_eval_point(test_df):
    row = evaluate_blend(test_df).iloc[0]
    ref = eval_point(test_df, STARTER_W, STARTER_CUT)
    for k in ("prec", "rec", "ft", "mt", "f1"):
        assert row[k] == pytest.approx(ref[k])


def test_alt_row_keeps_starter_cutoff(test_df):
    row = evaluate_blend(test_df).iloc[1]
    ref = eval_point(test_df, ALT_BLEND_W, STARTER_CUT)
    assert row["rec"] == pytest.approx(ref["rec"]) and row["ft"] == pytest.approx(ref["ft"])


def test_alt_weight_is_distinct_and_fixed():
    assert ALT_BLEND_W == 0.75
    assert abs(ALT_BLEND_W - STARTER_W) >= 0.3


def test_auc_uses_continuous_score(test_df):
    row = evaluate_blend(test_df).iloc[0]
    score = blend(style_score(test_df), activity_score(test_df), STARTER_W)
    assert row["auc"] == pytest.approx(auc(test_df["label_teen"], score))
    assert 0.5 < row["auc"] <= 1.0
