"""Drift check helpers: the drifted columns and the drifted test report. The full run is not tested here (it refits)."""
import numpy as np
import pandas as pd

from softsignal.drift_check import summary, teen_columns, window_decider
from softsignal.features import FEATURE_COLS, ID_COL, TARGET
from softsignal.loop import DecisionContext, make_env
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


def ctx():
    return DecisionContext(round=5, thresholds={}, audit={}, bounds={}, guards={},
                           rule={"action": "retune", "cap": 0.15})


def test_window_decider_copies_the_rule_and_asks_for_the_window_once_a1_says_real():
    decide = window_decider(2)
    not_real = decide(ctx(), {"a1": {"output": {"drift": "not_real"}}})["a2"]["output"]
    real = decide(ctx(), {"a1": {"output": {"drift": "real"}}})["a2"]["output"]
    after = decide(ctx(), {"a1": {"output": {"drift": "insufficient_data"}}})["a2"]["output"]
    assert not_real == {"action": "retune", "cap": 0.15}
    assert real == after == {"action": "retune", "cap": 0.15, "refit_window": 2}  # sticky once seen
    assert decide(ctx(), {})["a2"]["status"] == "LIVE"


def test_summary_counts_the_share_promoted_at_the_last_round():
    rows = [dict(seed=s, run="x", round=r, mode=m, a1_drift="real", rec=1.0, prec=1.0, ft=0.1, auc=0.9)
            for s, m in ((1, "ACTIVE"), (2, "SHADOW")) for r in (0, 1) for m in [m]]
    assert summary(pd.DataFrame(rows)).loc["x", "promoted"] == 0.5
