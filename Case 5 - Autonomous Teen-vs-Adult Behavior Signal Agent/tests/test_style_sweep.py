import numpy as np
import pandas as pd

from softsignal.data import load_data
from softsignal.metrics import f1, prf
from softsignal.tier1 import CUTOFFS, best_cutoffs, flag, style_score, style_sweep_report, sweep_cutoffs


def test_grid_is_17_cutoffs_from_010_to_090():
    assert len(CUTOFFS) == 17
    assert CUTOFFS[0] == 0.10 and CUTOFFS[-1] == 0.90 and 0.55 in CUTOFFS


def test_f1_zero_when_no_hits():
    assert f1(0.0, 0.0) == 0.0
    assert abs(f1(0.5, 1.0) - 2 / 3) < 1e-12


def test_flag_not_lost_to_float_error():
    # 0.1 + 0.2 is 0.30000000000000004 and 0.7 - 0.4 is 0.29999999999999993 in floats
    assert (0.7 - 0.4) < 0.3
    assert flag(pd.Series([0.7 - 0.4]), 0.3)[0] == 1


def test_055_row_matches_the_starter_rule():
    train, _ = load_data()
    sweep = sweep_cutoffs(train, style_score(train))
    row = sweep[sweep["cutoff"] == 0.55].iloc[0]
    style = (
        0.35 * (train["avg_word_len"] < 4.4).astype(float)
        + 0.25 * (train["first_person_rate"] > 0.06).astype(float)
        + 0.20 * (train["exclaim_rate"] > 0.008).astype(float)
        + 0.15 * (train["slang_emoji_rate"] > 0.002).astype(float)
        + 0.20 * (train["school_token_rate"] > 0).astype(float)
    )
    prec, rec, ft, mt = prf(train["label_teen"].to_numpy(), (style >= 0.55).astype(int).to_numpy())
    assert (row["prec"], row["rec"], row["ft"], row["mt"]) == (prec, rec, ft, mt)


def test_best_cutoffs_ties_go_to_lower_and_cap_respected():
    sweep = pd.DataFrame(
        {"cutoff": [0.3, 0.4, 0.5, 0.6], "rec": [0.9, 0.8, 0.8, 0.5],
         "ft": [0.40, 0.14, 0.10, 0.05], "f1": [0.7, 0.8, 0.8, 0.5]}
    )
    picks = best_cutoffs(sweep, cap=0.15)
    assert picks == {"f1_best": 0.4, "cap_best": 0.4}


def test_cap_best_none_when_cap_unreachable():
    sweep = pd.DataFrame({"cutoff": [0.3], "rec": [0.9], "ft": [0.4], "f1": [0.7]})
    assert best_cutoffs(sweep)["cap_best"] is None


def test_report_picks_on_train_and_scores_test_once():
    train, test = load_data()
    rep = style_sweep_report(train, test)
    assert len(rep["train_sweep"]) == 17
    assert set(rep["test"]["pick"]) <= {"f1_best", "cap_best"}
    for _, r in rep["test"].iterrows():
        pred = flag(style_score(test), r["cutoff"])
        assert (r["prec"], r["rec"], r["ft"], r["mt"]) == prf(test["label_teen"].to_numpy(), pred)
