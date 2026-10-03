"""No forbidden column (age, label, self-declared profile, birthday proxies) may reach a model."""
import pandas as pd
import pytest

from softsignal.data import JOINED, load_data, split_xy
from softsignal.features import FEATURE_COLS, FORBIDDEN


def test_allow_list_is_disjoint_from_forbidden():
    assert len(FEATURE_COLS) == 16
    assert not FORBIDDEN & set(FEATURE_COLS)


def test_allow_list_columns_exist_in_csv():
    assert set(FEATURE_COLS) <= set(pd.read_csv(JOINED, nrows=1).columns)


@pytest.mark.parametrize("which", [0, 1])
def test_split_xy_returns_only_allowed_columns(which):
    X, y = split_xy(load_data()[which])
    assert list(X.columns) == FEATURE_COLS
    assert not FORBIDDEN & set(X.columns)
    assert len(X) == len(y)
