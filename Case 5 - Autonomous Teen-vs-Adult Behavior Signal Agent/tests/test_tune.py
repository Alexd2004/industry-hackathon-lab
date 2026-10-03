import json

import numpy as np
import pandas as pd
import pytest

import softsignal.tier1 as t1
from softsignal.data import cv_folds, load_data
from softsignal.tier1 import (
    CUTOFFS, K_FOLDS, W_GRID, activity_score, blend, cv_summary, data_id, eval_point, is_current, load_best,
    pick_points, rules_id, save_best, style_score, sweep_cutoffs, sweep_grid, tune,
)

CAP = 0.15


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def tuned(data, tmp_path_factory):
    """One default tune() run, shared by the tests that only read its result."""
    train, test = data
    path = tmp_path_factory.mktemp("ck") / "best_params.json"
    return tune(train, test, checkpoint=path), path


def at(frame: pd.DataFrame, pt: dict) -> pd.Series:
    """The one row of frame at the (w, cutoff) of pt."""
    rows = frame[np.isclose(frame["w"], pt["w"]) & np.isclose(frame["cutoff"], pt["cutoff"])]
    assert len(rows) == 1
    return rows.iloc[0]


def test_grid_is_21_weights_times_17_cutoffs(data):
    assert len(W_GRID) == 21 and W_GRID[0] == 0.0 and W_GRID[-1] == 1.0 and 0.45 in W_GRID
    assert len(sweep_grid(data[0])) == 21 * 17


def test_w0_rows_equal_the_step3_sweep(data):
    train, _ = data
    grid = sweep_grid(train)
    w0 = grid[grid["w"] == 0.0].drop(columns="w").reset_index(drop=True)
    pd.testing.assert_frame_equal(w0, sweep_cutoffs(train, style_score(train), CUTOFFS))


def test_blend_matches_the_starter_formula(data):
    s, a = style_score(data[0]), activity_score(data[0])
    pd.testing.assert_series_equal(blend(s, a, 0.45), (1.0 - 0.45) * s + 0.45 * a)


def test_pick_points_tie_break_and_cap():
    grid = pd.DataFrame({
        "w": [0.0, 0.1, 0.2, 0.2], "cutoff": [0.5, 0.5, 0.5, 0.6],
        "rec": [0.8, 0.8, 0.8, 0.5], "ft": [0.14, 0.10, 0.10, 0.05], "f1": [0.7, 0.8, 0.8, 0.5],
    })
    picks = pick_points(grid, cap=CAP)
    assert picks["cap_best"] == {"w": 0.1, "cutoff": 0.5}  # equal recall: lower ft, then lower w
    assert picks["f1_best"] == {"w": 0.1, "cutoff": 0.5}
    assert pick_points(grid, cap=0.01)["cap_best"] is None


def test_pick_points_cap_col_uses_the_worst_fold_not_the_mean():
    grid = pd.DataFrame({
        "w": [0.0, 0.1, 0.2], "cutoff": [0.5, 0.5, 0.5],
        "rec": [0.9, 0.8, 0.5], "ft": [0.10, 0.10, 0.05], "ft_max": [0.20, 0.14, 0.08], "f1": [0.5, 0.5, 0.5],
    })
    assert pick_points(grid, CAP)["cap_best"] == {"w": 0.0, "cutoff": 0.5}  # mean ft passes
    assert pick_points(grid, CAP, cap_col="ft_max")["cap_best"] == {"w": 0.1, "cutoff": 0.5}  # worst fold fails w=0
    assert pick_points(grid, 0.05, cap_col="ft_max")["cap_best"] is None


def test_cv_summary_matches_a_loop_over_every_fold(data):
    train, _ = data
    cv = cv_summary(train)
    assert len(cv) == len(W_GRID) * len(CUTOFFS)
    folds = [sweep_grid(train.iloc[val].reset_index(drop=True)) for _, val in cv_folds(train, K_FOLDS)]
    for col, agg in (("rec", np.mean), ("ft", np.mean), ("f1", np.mean), ("ft_max", np.max), ("rec_min", np.min)):
        source = "ft" if col == "ft_max" else "rec" if col == "rec_min" else col
        expected = agg([f[source].to_numpy() for f in folds], axis=0)
        np.testing.assert_allclose(cv[col].to_numpy(), expected)


def test_cap_best_holds_the_cap_in_every_fold_recomputed_independently(data, tuned):
    train, _ = data
    out, _ = tuned
    pt = out["picks"]["cap_best"]
    for _, val in cv_folds(train, K_FOLDS):
        fold = train.iloc[val].reset_index(drop=True)
        assert eval_point(fold, pt["w"], pt["cutoff"])["ft"] <= CAP
    # and it has the best mean recall among the points that hold the cap in every fold
    cv = out["cv"]
    assert at(cv, pt)["rec"] == cv[cv["ft_max"] <= CAP]["rec"].max()


def test_worst_fold_rule_never_beats_the_mean_fold_rule_on_recall(tuned):
    out, _ = tuned
    cv = out["cv"]
    strict = at(cv, out["picks"]["cap_best"])["rec"]
    assert strict <= cv[cv["ft"] <= CAP]["rec"].max()


def test_f1_best_is_the_max_mean_fold_f1(tuned):
    out, _ = tuned
    cv = out["cv"]
    assert at(cv, out["picks"]["f1_best"])["f1"] == cv["f1"].max()


def test_tune_without_a_feasible_cap_pick_writes_nothing(data, tmp_path):
    train, test = data
    path = tmp_path / "best_params.json"
    out = tune(train, test, cap=0.001, checkpoint=path)
    assert out["picks"]["cap_best"] is None and out["cv_cap_best"] is None
    assert out["checkpoint_written"] is False and not path.exists()
    assert list(out["test"]["pick"]) == ["f1_best"]


def test_tune_writes_the_checkpoint_once_then_keeps_it(data, tuned):
    train, test = data
    out, path = tuned
    assert out["checkpoint_written"] is True
    assert tune(train, test, checkpoint=path)["checkpoint_written"] is False
    stored = load_best(path)
    assert stored["train"]["ft_max"] <= CAP and stored["k"] == K_FOLDS
    assert stored["rules"] == rules_id() and stored["data"] == data_id(train)
    assert is_current(stored, train)


def test_is_current_flags_a_different_cap_rules_or_data(data, tuned):
    train, _ = data
    _, path = tuned
    stored = load_best(path)
    assert not is_current(stored, train, cap=0.20)
    assert not is_current({**stored, "rules": "other"}, train)
    assert not is_current({**stored, "data": "other"}, train)
    assert not is_current({**stored, "k": K_FOLDS + 1}, train)


def test_tune_scores_test_once_per_pick_and_matches_eval_point(data, tuned):
    _, test = data
    out, _ = tuned
    for _, r in out["test"].iterrows():
        m = eval_point(test, r["w"], r["cutoff"])
        assert (r["prec"], r["rec"], r["ft"], r["mt"]) == (m["prec"], m["rec"], m["ft"], m["mt"])


def test_picks_use_train_only(data, tuned, tmp_path):
    train, test = data
    out, _ = tuned
    flipped = test.copy()
    flipped["label_teen"] = 1 - flipped["label_teen"]
    assert tune(train, flipped, checkpoint=tmp_path / "best_params.json")["picks"] == out["picks"]


def test_save_best_overwrites_only_on_strictly_higher_recall(tmp_path):
    path = tmp_path / "ck" / "best_params.json"
    pt = {"w": 0.5, "cutoff": 0.6}
    tm = lambda rec: {"rec": rec, "ft": 0.14, "ft_max": 0.14, "rec_min": rec}
    assert save_best(pt, tm(0.60), 0.15, 2100, path) is True
    assert save_best({"w": 0.6, "cutoff": 0.6}, tm(0.60), 0.15, 2100, path) is False  # equal: keep
    assert save_best({"w": 0.6, "cutoff": 0.6}, tm(0.55), 0.15, 2100, path) is False  # lower: keep
    assert json.loads(path.read_text())["w"] == 0.5
    assert save_best({"w": 0.7, "cutoff": 0.6}, tm(0.65), 0.15, 2100, path) is True
    assert json.loads(path.read_text())["w"] == 0.7
    assert save_best({"w": 0.1, "cutoff": 0.6}, tm(0.10), 0.20, 2100, path) is True  # different cap: new context
    assert save_best(pt, tm(0.1), 0.15, 2100, path, rules="other") is True  # scoring rules changed: replaced
    assert json.loads(path.read_text())["rules"] == "other"
    assert save_best(pt, tm(0.2), 0.15, 2100, path, rules="other") is True
    assert json.loads(path.read_text())["rules"] == "other"
    assert not list(path.parent.glob("*.tmp"))
    path.write_text("not json")
    assert save_best(pt, tm(0.1), 0.15, 2100, path) is True  # unreadable: replaced


def test_save_best_replaces_when_the_data_changed(tmp_path):
    path = tmp_path / "best_params.json"
    pt = {"w": 0.5, "cutoff": 0.5}
    tm = {"rec": 0.6, "ft": 0.1, "ft_max": 0.1, "rec_min": 0.6}
    assert save_best(pt, tm, 0.15, 2100, path, data="a") is True
    assert save_best(pt, tm, 0.15, 2100, path, data="a") is False
    assert save_best(pt, tm, 0.15, 2100, path, data="b") is True  # same size, different rows


def test_rules_id_is_stable():
    assert rules_id() == rules_id() and len(rules_id()) == 12


@pytest.mark.parametrize("name", ["blend", "flag", "pick_points", "cv_folds", "eval_point", "tune"])
def test_rules_id_covers_each_part_of_the_pick(name, monkeypatch):
    base = rules_id()

    def replacement(*args, **kwargs):  # different source text from the original
        return None

    monkeypatch.setattr(t1, name, replacement)
    assert rules_id() != base


def test_data_id_changes_with_the_rows(data):
    train, _ = data
    assert data_id(train) == data_id(train.copy())
    changed = train.copy()
    changed.loc[0, "avg_word_len"] += 1.0
    assert data_id(changed) != data_id(train)


def test_load_best_reads_the_stored_checkpoint(tmp_path):
    path = tmp_path / "best_params.json"
    assert load_best(path) is None
    path.write_text("{bad")
    assert load_best(path) is None
    path.write_text('{"w": 0.5}')
    assert load_best(path) == {"w": 0.5}
