"""Split and loader behaviour. Every test uses tmp_path, never the real results/split.json."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from softsignal.data import ID_COL, TARGET, check_frame, cv_folds, load_data, make_split
from softsignal.features import N_TEST, SEED


def make_df(n=3000):
    """Balanced synthetic frame, same size as the real one so N_TEST fits."""
    return pd.DataFrame({ID_COL: [f"B{i}" for i in range(n)], TARGET: np.arange(n) % 2})


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "joined.csv"
    make_df().to_csv(path, index=False)
    return path


def test_check_frame_accepts_valid_frame():
    check_frame(make_df())


def test_check_frame_rejects_duplicate_id():
    df = make_df()
    df.loc[1, ID_COL] = df.loc[0, ID_COL]
    with pytest.raises(ValueError, match="unique"):
        check_frame(df)


@pytest.mark.parametrize("bad", [np.nan, 2, -1])
def test_check_frame_rejects_bad_label(bad):
    df = make_df()
    df[TARGET] = df[TARGET].astype(float)
    df.loc[0, TARGET] = bad
    with pytest.raises(ValueError, match="0 or 1"):
        check_frame(df)


def test_make_split_is_disjoint_and_complete():
    df = make_df()
    split = make_split(df)
    train, test = set(split["train_ids"]), set(split["test_ids"])
    assert not train & test
    assert train | test == set(df[ID_COL])
    assert split["n_train"] == len(train) and split["n_test"] == len(test) == N_TEST


def test_make_split_is_stratified():
    split = make_split(make_df())
    for counts in split["label_counts"].values():
        assert abs(counts["0"] - counts["1"]) <= 1


def test_make_split_same_seed_same_ids_other_seed_differs():
    df = make_df()
    assert make_split(df, seed=1) == make_split(df, seed=1)
    assert make_split(df, seed=1)["test_ids"] != make_split(df, seed=2)["test_ids"]


def test_load_data_creates_file_then_reuses_it(csv_path, tmp_path):
    split_file = tmp_path / "out" / "split.json"
    train, test = load_data(csv_path, split_file)
    assert split_file.exists()
    assert len(test) == N_TEST and len(train) + len(test) == 3000
    assert not set(train[ID_COL]) & set(test[ID_COL])
    train2, test2 = load_data(csv_path, split_file)
    assert test2[ID_COL].tolist() == test[ID_COL].tolist()


def test_load_data_rejects_split_file_for_other_data(csv_path, tmp_path):
    split_file = tmp_path / "split.json"
    load_data(csv_path, split_file)
    other = make_df()
    other[ID_COL] = "X" + other[ID_COL]
    other.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        load_data(csv_path, split_file)


def test_load_data_rejects_unstratified_split_file(csv_path, tmp_path):
    split_file = tmp_path / "split.json"
    df = make_df()
    zeros = df.loc[df[TARGET] == 0, ID_COL].tolist()
    test_ids = zeros[:N_TEST]
    train_ids = sorted(set(df[ID_COL]) - set(test_ids))
    split_file.write_text(json.dumps(
        {"seed": SEED, "n_test": N_TEST, "train_ids": train_ids, "test_ids": sorted(test_ids)}
    ))
    with pytest.raises(ValueError, match="not stratified"):
        load_data(csv_path, split_file)


def write_split_with_seed(csv_path, split_file, seed):
    split_file.write_text(json.dumps(make_split(pd.read_csv(csv_path), seed=seed)))


def test_param_mismatch_warns_by_default_and_uses_file(csv_path, tmp_path):
    split_file = tmp_path / "split.json"
    write_split_with_seed(csv_path, split_file, SEED + 1)
    with pytest.warns(UserWarning, match="Delete it"):
        _, test = load_data(csv_path, split_file)
    assert sorted(test[ID_COL]) == json.loads(split_file.read_text())["test_ids"]


def test_param_mismatch_raises_when_error(csv_path, tmp_path):
    split_file = tmp_path / "split.json"
    write_split_with_seed(csv_path, split_file, SEED + 1)
    with pytest.raises(ValueError, match="Delete it"):
        load_data(csv_path, split_file, on_param_mismatch="error")


def test_matching_params_do_not_warn(csv_path, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_data(csv_path, tmp_path / "split.json", on_param_mismatch="error")


def test_invalid_on_param_mismatch_value(csv_path, tmp_path):
    with pytest.raises(ValueError, match="on_param_mismatch"):
        load_data(csv_path, tmp_path / "split.json", on_param_mismatch="ignore")


def test_cv_folds_partition_train_and_stay_stratified():
    train = make_df(2100)
    folds = cv_folds(train, k=5)
    assert len(folds) == 5
    val_all = np.concatenate([val for _, val in folds])
    assert sorted(val_all) == list(range(len(train)))
    for fit, val in folds:
        assert not set(fit) & set(val)
        counts = train.iloc[val][TARGET].value_counts()
        assert abs(counts[0] - counts[1]) <= 1


def test_cv_folds_same_seed_same_folds():
    train = make_df(2100)
    a, b = cv_folds(train), cv_folds(train)
    assert all((x[1] == y[1]).all() for x, y in zip(a, b))
