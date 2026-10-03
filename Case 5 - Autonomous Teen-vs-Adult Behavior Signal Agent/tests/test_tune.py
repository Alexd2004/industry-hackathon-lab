import json

import pandas as pd

from softsignal.data import load_data
from softsignal.tier1 import (
    CUTOFFS, W_GRID, activity_score, blend, cv_summary, data_id, eval_point, load_best, pick_points, rules_id, save_best,
    style_score, style_sweep_report, sweep_cutoffs, sweep_grid, tune,
)


def test_grid_is_21_weights_times_17_cutoffs():
    assert len(W_GRID) == 21 and W_GRID[0] == 0.0 and W_GRID[-1] == 1.0 and 0.45 in W_GRID
    train, _ = load_data()
    assert len(sweep_grid(train)) == 21 * 17


def test_w0_rows_equal_the_step3_sweep():
    train, _ = load_data()
    grid = sweep_grid(train)
    w0 = grid[grid["w"] == 0.0].drop(columns="w").reset_index(drop=True)
    pd.testing.assert_frame_equal(w0, sweep_cutoffs(train, style_score(train), CUTOFFS))


def test_blend_matches_the_starter_formula():
    train, _ = load_data()
    s, a = style_score(train), activity_score(train)
    pd.testing.assert_series_equal(blend(s, a, 0.45), (1.0 - 0.45) * s + 0.45 * a)


def test_pick_points_tie_break_and_cap():
    grid = pd.DataFrame({
        "w": [0.0, 0.1, 0.2, 0.2], "cutoff": [0.5, 0.5, 0.5, 0.6],
        "rec": [0.8, 0.8, 0.8, 0.5], "ft": [0.14, 0.10, 0.10, 0.05], "f1": [0.7, 0.8, 0.8, 0.5],
    })
    picks = pick_points(grid, cap=0.15)
    assert picks["cap_best"] == {"w": 0.1, "cutoff": 0.5}  # equal recall: lower ft, then lower w
    assert picks["f1_best"] == {"w": 0.1, "cutoff": 0.5}
    assert pick_points(grid, cap=0.01)["cap_best"] is None


def test_cap_best_respects_cap_in_every_train_fold(tmp_path):
    train, test = load_data()
    out = tune(train, test, checkpoint=tmp_path / "best_params.json")
    pt = out["picks"]["cap_best"]
    cv = out["cv"]
    row = cv[(cv["w"] == pt["w"]) & (cv["cutoff"] == pt["cutoff"])].iloc[0]
    assert row["ft_max"] <= 0.15
    assert row["rec"] == cv[cv["ft_max"] <= 0.15]["rec"].max()


def test_cv_summary_matches_per_fold_metrics():
    train, _ = load_data()
    cv = cv_summary(train)
    assert len(cv) == len(W_GRID) * len(CUTOFFS)
    assert (cv["ft_max"] >= cv["ft"]).all() and (cv["rec_min"] <= cv["rec"]).all()
    from softsignal.data import cv_folds
    _, val = cv_folds(train)[0]
    first = eval_point(train.iloc[val].reset_index(drop=True), 0.75, 0.6)
    row = cv[(cv["w"] == 0.75) & (cv["cutoff"] == 0.6)].iloc[0]
    assert row["ft_max"] >= first["ft"] and row["rec_min"] <= first["rec"]


def test_tune_without_a_feasible_cap_pick_writes_nothing(tmp_path):
    train, test = load_data()
    path = tmp_path / "best_params.json"
    out = tune(train, test, cap=0.001, checkpoint=path)
    assert out["picks"]["cap_best"] is None and out["checkpoint_written"] is False and not path.exists()
    assert list(out["test"]["pick"]) == ["f1_best"]


def test_tune_writes_the_checkpoint_once_then_keeps_it(tmp_path):
    train, test = load_data()
    path = tmp_path / "best_params.json"
    assert tune(train, test, checkpoint=path)["checkpoint_written"] is True
    assert tune(train, test, checkpoint=path)["checkpoint_written"] is False
    assert load_best(path)["train"]["ft_max"] <= 0.15


def test_tune_scores_test_once_per_pick_and_matches_eval_point(tmp_path):
    train, test = load_data()
    out = tune(train, test, checkpoint=tmp_path / "best_params.json")
    for _, r in out["test"].iterrows():
        m = eval_point(test, r["w"], r["cutoff"])
        assert (r["prec"], r["rec"], r["ft"], r["mt"]) == (m["prec"], m["rec"], m["ft"], m["mt"])


def test_picks_use_train_only(tmp_path):
    train, test = load_data()
    base = tune(train, test, checkpoint=tmp_path / "best_params.json")["picks"]
    flipped = test.copy()
    flipped["label_teen"] = 1 - flipped["label_teen"]
    assert tune(train, flipped, checkpoint=tmp_path / "best_params.json")["picks"] == base


def test_blend_pick_recall_not_below_the_style_only_points_under_the_same_rule(tmp_path):
    train, test = load_data()
    out = tune(train, test, checkpoint=tmp_path / "best_params.json")
    cv = out["cv"]
    style_only = cv[(cv["w"] == 0.0) & (cv["ft_max"] <= 0.15)]["rec"].max()
    pt = out["picks"]["cap_best"]
    best = cv[(cv["w"] == pt["w"]) & (cv["cutoff"] == pt["cutoff"])]["rec"].iloc[0]
    assert best >= style_only


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


def test_rules_id_is_stable_and_stored():
    assert rules_id() == rules_id() and len(rules_id()) == 12


def test_rules_id_covers_blend_and_flag(monkeypatch):
    import softsignal.tier1 as t1

    base = t1.rules_id()
    monkeypatch.setattr(t1, "blend", lambda style, activity, w: style)
    assert t1.rules_id() != base


def test_data_id_changes_with_the_rows(tmp_path):
    train, _ = load_data()
    assert data_id(train) == data_id(train.copy())
    changed = train.copy()
    changed.loc[0, "avg_word_len"] += 1.0
    assert data_id(changed) != data_id(train)


def test_save_best_replaces_when_the_data_changed(tmp_path):
    path = tmp_path / "best_params.json"
    pt = {"w": 0.5, "cutoff": 0.5}
    tm = {"prec": 0.5, "rec": 0.6, "ft": 0.1, "mt": 0.4, "f1": 0.5}
    assert save_best(pt, tm, 0.15, 2100, path, data="a") is True
    assert save_best(pt, tm, 0.15, 2100, path, data="a") is False
    assert save_best(pt, tm, 0.15, 2100, path, data="b") is True  # same size, different rows


def test_load_best_reads_the_stored_checkpoint(tmp_path):
    path = tmp_path / "best_params.json"
    assert load_best(path) is None
    path.write_text("{bad")
    assert load_best(path) is None
    path.write_text('{"w": 0.5}')
    assert load_best(path) == {"w": 0.5}
