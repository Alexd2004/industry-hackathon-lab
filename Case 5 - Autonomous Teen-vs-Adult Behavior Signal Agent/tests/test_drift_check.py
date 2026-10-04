"""Drift check helpers: the drifted columns and the drifted test report. The full run is not tested here (it refits)."""
import numpy as np
import pandas as pd

from softsignal.drift_check import teen_columns
from softsignal.features import FEATURE_COLS, ID_COL, TARGET
from softsignal.loop import make_env
from softsignal.oracle import Drift
from tests.test_oracle import make_frame


def frame_with_signal(n=600, seed=1):
    df = make_frame(n, seed=seed)
    rng = np.random.default_rng(seed)
    df["sessions_per_day"] = df[TARGET] * 2.0 + rng.random(n)  # strongly teen-leaning
    df["share_news_views"] = -df[TARGET] * 2.0 + rng.random(n)  # adult-leaning
    return df


def test_teen_columns_are_the_most_teen_correlated_on_train():
    cols = teen_columns(frame_with_signal(), n=2)
    assert cols[0] == "sessions_per_day" and "share_news_views" not in cols
    assert all(c in FEATURE_COLS for c in cols)


def test_a_drifted_env_reports_on_drifted_test_rows(tmp_path):
    from softsignal.agent_timer import AgentTimer
    train, test = frame_with_signal(2100), make_frame(900, prefix="X")
    drift = Drift(2, 1.0, ("sessions_per_day",))
    pol = {"cap_false_teen": 0.15, "review_budget": 0.25, "soft_recall": 0.9, "min_audit_adults": 120,
           "audit_per_batch": 60, "cap_margin": 0.0, "psi_drift": 0.25, "min_a3_errors": 30}
    kw = dict(policy=pol, tm=object(), timer=AgentTimer(tmp_path / "c.jsonl", run="t"))
    clean, drifted = make_env(train, test, **kw), make_env(train, test, drift=drift, **kw)
    assert clean.test["sessions_per_day"].equals(test["sessions_per_day"])
    assert not drifted.test["sessions_per_day"].equals(test["sessions_per_day"])
    assert drifted.test[TARGET].equals(test[TARGET]) and drifted.test[ID_COL].equals(test[ID_COL])
