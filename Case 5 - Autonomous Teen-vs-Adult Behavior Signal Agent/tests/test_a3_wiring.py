"""Step 16, A3 sub-step 3: A3 inside the round (crew.a3_payload, the before_decision hook, the hand-off to A2).

The unit tests use stand-in state and oracle objects. One real run (offline, 6 rounds, min_a3_errors set) checks
the wiring end to end: which rounds A3 analyses, what its input holds, and what A2 is given. No network.
"""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from softsignal import crew, loop
from softsignal.agent_timer import AgentTimer
from softsignal.agents import a3_errors
from softsignal.agents.base import FALLBACK, INSUFFICIENT, LIVE, AgentResult
from softsignal.agents.contracts import INSUFFICIENT_INPUT
from softsignal.data import load_data
from softsignal.features import ID_COL, TARGET
from softsignal.text_model import build_matrix

from test_a3_errors import live_output

N_ROUNDS = 6  # the first stack candidate comes from the round 5 refit, so round 6 is the first with signals


# --- a3_payload on stand-ins -------------------------------------------------------------------------------

def stub(prior_ids, batch_ids, scores, labels, t_verify=0.5, min_errors=1, test_ids=()):
    """(env, state, batch, prior): state.live_scores holds the prior rows' scores then the batch's, as loop.py does."""
    feats = lambda ids: pd.DataFrame({ID_COL: ids})  # noqa: E731
    audit = pd.DataFrame({ID_COL: list(labels), TARGET: list(labels.values())})
    seen_before = []

    class Oracle:
        def revealed(self, source, before_round=None):
            seen_before.append((source, before_round))
            return audit

    env = SimpleNamespace(oracle=Oracle(), policy={"min_a3_errors": min_errors}, test=feats(list(test_ids)))
    state = SimpleNamespace(live=SimpleNamespace(model=None, th=SimpleNamespace(t_verify=t_verify)), candidate=None,
                            live_scores=np.array(scores, dtype=float))
    batch = SimpleNamespace(round=3, rows=feats(batch_ids))
    return env, state, batch, feats(prior_ids), seen_before


def test_only_earlier_rounds_audit_labels_and_arrival_scores_count():
    labels = {"p0": 0, "p1": 1, "p2": 0, "p3": 1}
    env, state, batch, prior, asked = stub(["p0", "p1", "p2", "p3"], ["b0", "b1"], [0.9, 0.1, 0.9, 0.1, 0.9, 0.9],
                                           labels)
    p = crew.a3_payload(env, state, batch, prior)
    assert asked == [("audit", 3)]  # this round's labels are never asked for
    assert p["round"] == 3 and p["audit"] == {"adults": 2, "teens": 2}
    assert p["n_errors"] == {"false_teen": 2, "missed_teen": 2}  # p0, p2 adults at 0.9; p1, p3 teens at 0.1


def test_only_audit_accounts_are_analysed():
    env, state, batch, prior, _ = stub(["p0", "p1", "p2"], ["b0"], [0.9, 0.9, 0.1, 0.5], {"p0": 0, "p2": 1})
    p = crew.a3_payload(env, state, batch, prior)  # p1 was revealed in no audit slice
    assert p["audit"] == {"adults": 1, "teens": 1}
    assert p["n_errors"] == {"false_teen": 1, "missed_teen": 1}


def test_after_a_promote_only_the_rounds_on_the_live_scale_count():
    labels = {f"p{i}": i % 2 for i in range(4)}
    # live_scores restarted at the promote: only p2 (0.9), p3 (0.1) and the batch (0.9, 0.9) were scored on the live
    # scale; p0 and p1 would be errors on the old scale (adult 0.9 would not be, teen 0.9 would not be) and must not count
    env, state, batch, prior, _ = stub(["p0", "p1", "p2", "p3"], ["b0", "b1"], [0.9, 0.1, 0.9, 0.9], labels)
    p = crew.a3_payload(env, state, batch, prior)
    assert p["audit"] == {"adults": 1, "teens": 1}  # p2 (adult) and p3 (teen)
    assert p["n_errors"] == {"false_teen": 1, "missed_teen": 1}  # p2 adult at 0.9, p3 teen at 0.1


def test_round_zero_has_nothing_to_analyse():
    env, state, _, prior, asked = stub([], [], [], {})
    p = crew.a3_payload(env, state, None, prior)
    assert p["round"] == 0 and p["audit"] == {"adults": 0, "teens": 0}
    assert a3_errors.insufficient_reason(p) is not None


def test_no_stack_model_means_no_signals_and_insufficient_data():
    env, state, batch, prior, _ = stub(["p0", "p1"], ["b0"], [0.9, 0.1, 0.5], {"p0": 0, "p1": 1})
    p = crew.a3_payload(env, state, batch, prior)
    assert p["n_errors"] == {"false_teen": 1, "missed_teen": 1} and p["false_teen"]["signals"] == []
    assert "no explanation signals" in a3_errors.insufficient_reason(p)


def test_a_frozen_test_account_in_the_audit_rows_raises():
    env, state, batch, prior, _ = stub(["p0"], ["b0"], [0.9, 0.5], {"p0": 0}, test_ids=["p0"])
    with pytest.raises(Exception, match="frozen test"):
        crew.a3_payload(env, state, batch, prior)


# --- one real run ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    train, test = load_data(on_param_mismatch="error")
    tm = build_matrix(train[loop.ID_COL], cache_dir=tmp_path_factory.mktemp("cache"))
    timer = AgentTimer(tmp_path_factory.mktemp("calls") / "calls.jsonl", run="20261004T000000.000000Z-a3")
    env = loop.make_env(train, test, timer=timer, tm=tm)
    env.policy["min_a3_errors"] = 1
    payloads, to_a2 = {}, {}
    real_payload, real_a2 = crew.a3_payload, crew.a2_input

    def payload_spy(env_, state, batch, prior):
        p = real_payload(env_, state, batch, prior)
        payloads[p["round"]] = p
        return p

    def a2_spy(round_id, *args, **kw):
        to_a2[round_id] = kw["a3"]
        return real_a2(round_id, *args, **kw)

    def fake_live(payload, client=None, timer=None, round_id=None):
        """Offline A3, except that a round with something to analyse answers LIVE with a valid reply."""
        if a3_errors.insufficient_reason(payload) is not None:
            return real_run(payload, client, timer, round_id)
        kind = next(k for k in ("false_teen", "missed_teen") if payload[k]["signals"])
        return AgentResult("A3", LIVE, live_output(payload, kind), None, "h")

    real_run = a3_errors.run_a3
    with mock.patch.object(crew, "a3_payload", payload_spy), mock.patch.object(crew, "a2_input", a2_spy), \
            mock.patch.object(a3_errors, "run_a3", fake_live):
        rounds, records = crew.run_crew(env, None, N_ROUNDS)
    return env, payloads, to_a2, records


def test_every_round_has_an_a3_block(run):
    _, _, _, records = run
    assert [r["a3"]["status"] for r in records] == [FALLBACK] * 6 + [LIVE]


def test_nothing_to_analyse_before_the_first_stack_and_no_model_call(run):
    _, payloads, _, records = run
    for rnd in range(0, 6):
        assert records[rnd]["a3"]["fallback_reason"] == INSUFFICIENT
        assert records[rnd]["a3"]["output"]["status"] == INSUFFICIENT


def test_a3_input_holds_earlier_rounds_audit_labels_only(run):
    env, payloads, _, _ = run
    for rnd, p in payloads.items():
        assert p["audit"] == env.oracle.audit_counts(before_round=rnd)  # counts from rounds < rnd, not this one
        assert p["min_errors"] == 1 and p["agent"] == "A3"


def test_round_six_has_signals_from_the_candidate_and_is_analysed(run):
    _, payloads, _, records = run
    p = payloads[6]
    assert sum(p["n_errors"].values()) > 0 and any(p[k]["signals"] for k in ("false_teen", "missed_teen"))
    assert records[6]["a3"]["status"] == LIVE and records[6]["a3"]["output"]["status"] == "ok"
    assert all(not payloads[r]["false_teen"]["signals"] for r in range(0, 6))  # no stack model yet


def test_a2_gets_a3_only_when_it_analysed_something(run):
    _, _, to_a2, records = run
    for rnd in range(1, 6):
        assert to_a2[rnd] == INSUFFICIENT_INPUT
    assert to_a2[6] == records[6]["a3"]["output"] and to_a2[6]["status"] == "ok"


def test_no_test_id_or_label_in_any_a3_input(run):
    from softsignal.agents.contracts import frozen_test_ids

    _, payloads, _, _ = run
    test = frozen_test_ids()
    for p in payloads.values():
        text = str(p)
        assert not any(i in text for i in list(test)[:200])
        assert "label" not in text and "blogger_id" not in text


def test_a3_exceptions_do_not_break_a1_or_the_round(tmp_path_factory):
    train, test = load_data(on_param_mismatch="error")
    tm = build_matrix(train[loop.ID_COL], cache_dir=tmp_path_factory.mktemp("cache"))
    env = loop.make_env(train, test, timer=AgentTimer(tmp_path_factory.mktemp("c") / "calls.jsonl", run="x-a3"), tm=tm)
    with mock.patch.object(crew, "a3_payload", side_effect=RuntimeError("boom")):
        _, records = crew.run_crew(env, None, 1)
    for r in records:
        assert r["a3"]["status"] == FALLBACK and r["a3"]["fallback_reason"] == crew.A3_ERROR
        assert r["a3"]["output"] == a3_errors.fallback_output() and "boom" in r["a3"]["errors"][0]
        assert r["a1"]["status"] is not None  # A1 still ran
