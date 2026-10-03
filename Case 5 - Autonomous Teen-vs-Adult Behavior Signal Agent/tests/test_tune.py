import json

import pandas as pd

from softsignal.data import load_data
from softsignal.tier1 import (
    CUTOFFS, W_GRID, blend, eval_point, pick_points, save_best, style_score, activity_score,
    style_sweep_report, sweep_cutoffs, sweep_grid, tune,
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


def test_cap_best_respects_cap_on_train():
    train, test = load_data()
    out = tune(train, test, checkpoint=_tmp_path())
    pt = out["picks"]["cap_best"]
    row = out["grid"][(out["grid"]["w"] == pt["w"]) & (out["grid"]["cutoff"] == pt["cutoff"])].iloc[0]
    assert row["ft"] <= 0.15
    assert row["rec"] == out["grid"][out["grid"]["ft"] <= 0.15]["rec"].max()


def _tmp_path():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp()) / "best_params.json"


def test_tune_scores_test_once_per_pick_and_matches_eval_point():
    train, test = load_data()
    out = tune(train, test, checkpoint=_tmp_path())
    for _, r in out["test"].iterrows():
        m = eval_point(test, r["w"], r["cutoff"])
        assert (r["prec"], r["rec"], r["ft"], r["mt"]) == (m["prec"], m["rec"], m["ft"], m["mt"])


def test_picks_use_train_only():
    train, test = load_data()
    base = tune(train, test, checkpoint=_tmp_path())["picks"]
    flipped = test.copy()
    flipped["label_teen"] = 1 - flipped["label_teen"]
    assert tune(train, flipped, checkpoint=_tmp_path())["picks"] == base


def test_blend_pick_recall_not_below_the_step3_style_only_pick():
    train, test = load_data()
    out = tune(train, test, checkpoint=_tmp_path())
    step3 = style_sweep_report(train, test)["picks"]["cap_best"]
    g = out["grid"]
    w0 = g[(g["w"] == 0.0) & (g["cutoff"] == step3)]["rec"].iloc[0]
    best = g[(g["w"] == out["picks"]["cap_best"]["w"]) & (g["cutoff"] == out["picks"]["cap_best"]["cutoff"])]["rec"].iloc[0]
    assert best >= w0


def test_save_best_overwrites_only_on_strictly_higher_recall(tmp_path):
    path = tmp_path / "ck" / "best_params.json"
    pt = {"w": 0.5, "cutoff": 0.6}
    tm = lambda rec: {"prec": 0.7, "rec": rec, "ft": 0.14, "mt": 0.3, "f1": 0.6}
    assert save_best(pt, tm(0.60), 0.15, 2100, path) is True
    assert save_best({"w": 0.6, "cutoff": 0.6}, tm(0.60), 0.15, 2100, path) is False  # equal: keep
    assert save_best({"w": 0.6, "cutoff": 0.6}, tm(0.55), 0.15, 2100, path) is False  # lower: keep
    assert json.loads(path.read_text())["w"] == 0.5
    assert save_best({"w": 0.7, "cutoff": 0.6}, tm(0.65), 0.15, 2100, path) is True
    assert json.loads(path.read_text())["w"] == 0.7
    assert save_best({"w": 0.1, "cutoff": 0.6}, tm(0.10), 0.20, 2100, path) is True  # different cap: new context
    path.write_text("not json")
    assert save_best(pt, tm(0.1), 0.15, 2100, path) is True  # unreadable: replaced
