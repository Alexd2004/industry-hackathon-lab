"""A1 drift watcher (Tier 3, step 16): contract, fallback rule, checks, every path. No test calls the network."""
import copy
import json
from types import SimpleNamespace

import anthropic
import httpx2
import numpy as np
import pandas as pd
import pytest

from softsignal.agent_timer import AgentTimer, load_records, round_agent_summary
from softsignal.agents import base
from softsignal.agents.a1_drift import NOT_REAL, REAL, SYSTEM, fallback_output, run_a1, user_message, validate_output
from softsignal.agents.base import (
    AGE_CLAIM, API_ERROR, CONNECTION, FALLBACK, INSUFFICIENT, INVALID, LIVE, NUMBER_NOT_IN_INPUT, OFFLINE, REFUSAL,
    TIMEOUT, UNKNOWN_FIELD,
)
from softsignal.agents.contracts import PSI_GROUPS, a1_fields, a1_history, a1_input, floor_psi, group_psi
from softsignal.agents.schemas import A1Output
from softsignal.data import load_data
from softsignal.features import ACTIVITY_COLS, FEATURE_COLS, ID_COL, TARGET, TEXT_COLS
from softsignal.metrics import psi
from softsignal.oracle import Oracle
from softsignal.policy import load_policy

from agent_fakes import REQ, FakeClient, reply

AUDIT = {"adults": 57, "teens": 63}


def rows(n=300, seed=0, shift=None) -> pd.DataFrame:
    """Feature rows drawn from one population; shift={col: delta} moves one column (a real drift)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(0, 1, (n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    df.insert(0, ID_COL, [f"T{seed}-{i}" for i in range(n)])
    for col, delta in (shift or {}).items():
        df[col] = df[col] + delta
    return df


def payload(ref=None, batch=None, score=0.042, history=(), rnd=3, psi_drift=0.25, audit=AUDIT):
    ref = rows(600, seed=1) if ref is None else ref
    batch = rows(300, seed=2) if batch is None else batch
    return a1_input(ref, batch, score, audit, list(history), rnd, psi_drift)


def run(p, client, tmp_path, rnd=3):
    timer = AgentTimer(tmp_path / "calls.jsonl", run="test-run")
    return run_a1(p, client=client, timer=timer, round_id=rnd), load_records(timer.path)


# --- contract -----------------------------------------------------------------------------------

def test_input_is_counts_and_psi_only():
    p = payload(history=[{"round": 2, "score": 0.07, "activity_max": 0.16, "text_max": 0.13}])
    assert set(p) == {"agent", "round", "n_reference", "n_batch", "psi", "psi_drift", "psi_conventions", "n_features",
                      "audit", "history"}
    assert p["psi_conventions"] == {"stable": 0.10, "large": 0.25} and p["n_features"] == {"activity": 9, "text": 7}
    assert set(p["psi"]) == {"score", *(f"{g}_{k}" for g in PSI_GROUPS for k in ("max", "mean", "top_feature"))}
    text = json.dumps(p)
    assert "T1-" not in text and "T2-" not in text and f'"{TARGET}"' not in text  # no ids, no label


def test_group_psi_is_metrics_psi_per_column():
    ref, batch = rows(600, seed=1), rows(300, seed=2, shift={"pct_active_late_night": 0.3})
    g = group_psi(ref, batch)
    per = {c: psi(ref[c], batch[c]) for c in ACTIVITY_COLS}
    assert g["activity_top_feature"] == "pct_active_late_night" == max(per, key=per.get)
    assert g["activity_max"] == floor_psi(max(per.values()))  # floored: the shown value gives the exact verdict
    assert g["activity_mean"] == floor_psi(float(np.mean(list(per.values()))))
    assert g["text_max"] == floor_psi(max(psi(ref[c], batch[c]) for c in TEXT_COLS))


def test_input_never_carries_a_label_even_if_the_rows_do():
    ref, batch = rows(600, seed=1), rows(300, seed=2)
    a = a1_input(ref, batch, 0.04, AUDIT, [], 3, 0.25)
    b = a1_input(ref.assign(**{TARGET: 1}), batch.assign(**{TARGET: 0}), 0.04, AUDIT, [], 3, 0.25)
    assert a == b


def test_no_reference_means_no_psi_and_insufficient_data():
    none = pd.DataFrame(columns=FEATURE_COLS)
    p = a1_input(none, rows(), None, AUDIT, [], 1, 0.25)
    assert p["n_reference"] == 0 and p["psi"] == {"score": None}
    assert fallback_output(p)["drift"] == INSUFFICIENT


def test_fields_are_every_numeric_value_by_path():
    p = payload(history=[{"round": 2, "score": None, "activity_max": 0.16, "text_max": 0.13}])
    f = a1_fields(p)
    assert f["psi.activity_max"] == p["psi"]["activity_max"] and f["audit.adults"] == 57 and f["psi_drift"] == 0.25
    assert f["history.round2.activity_max"] == 0.16 and "history.round2.score" not in f  # None: nothing to cite
    assert "psi.activity_top_feature" not in f  # a name, not a number
    assert a1_history(p) == {"round": 3, "score": p["psi"]["score"], "activity_max": p["psi"]["activity_max"],
                             "text_max": p["psi"]["text_max"]}


def test_numpy_numbers_are_json_safe():
    p = a1_input(rows(600, 1), rows(300, 2), np.float64(0.05), {"adults": np.int64(3), "teens": np.int64(4)}, [],
                 np.int64(2), 0.25)
    assert type(p["round"]) is int and type(p["audit"]["adults"]) is int
    json.dumps(p) and user_message(p)


# --- the fallback rule (policy.yaml psi_drift) ----------------------------------------------------------

def test_committed_threshold_is_the_measured_value():
    assert load_policy()["psi_drift"] == 0.25


def test_no_drift_reads_not_real_and_a_shift_reads_real():
    calm = fallback_output(payload())
    assert calm["drift"] == NOT_REAL and calm["evidence"][1] == {"field": "psi_drift", "value": 0.25}
    shifted = payload(batch=rows(300, seed=2, shift={"night_notification_open_rate": 0.3}))
    out = fallback_output(shifted)
    assert out["drift"] == REAL and out["evidence"][0]["field"] == "psi.activity_max"
    assert "night_notification_open_rate" in out["reason"]


def test_threshold_is_inclusive_and_the_score_psi_counts():
    p = payload(score=0.9)
    p["psi"] |= {"activity_max": 0.1, "text_max": 0.1}
    assert fallback_output(p)["drift"] == REAL and fallback_output(p)["evidence"][0]["field"] == "psi.score"
    p["psi"]["score"] = 0.25  # exactly at psi_drift
    assert fallback_output(p)["drift"] == REAL
    p["psi"]["score"] = None  # the round after a promote: features only
    assert fallback_output(p)["drift"] == NOT_REAL


def test_no_threshold_is_insufficient_data():
    p = payload(psi_drift=None)
    assert fallback_output(p)["drift"] == INSUFFICIENT and "psi_drift" in fallback_output(p)["reason"]


@pytest.mark.parametrize("seed", range(6))
def test_fallback_passes_its_own_checks(seed):
    shift = {"pct_active_school_hours": 0.4} if seed % 2 else None
    p = payload(ref=rows(300 * (seed + 1), seed=10 + seed), batch=rows(300, seed=20 + seed, shift=shift),
                score=None if seed == 3 else round(0.01 * seed, 3))
    out = fallback_output(p)
    A1Output(**out)
    assert validate_output(out, p) == (None, [])


def test_real_batches_read_not_real_with_the_committed_threshold():
    # the measurement behind psi_drift: seed 42's batches against the batches before them
    train, test = load_data(on_param_mismatch="error")
    seen, verdicts = [], []
    for b in Oracle(train, test[ID_COL].tolist()):
        ref = pd.concat(seen, ignore_index=True) if seen else pd.DataFrame(columns=FEATURE_COLS)
        verdicts.append(fallback_output(a1_input(ref, b.rows, None, AUDIT, [], b.round, 0.25))["drift"])
        seen.append(b.rows)
    assert verdicts == [INSUFFICIENT] + [NOT_REAL] * 6


# --- checks -----------------------------------------------------------------------------------------

def live_output(p) -> dict:
    return {"drift": NOT_REAL, "evidence": [{"field": "psi.activity_max", "value": p["psi"]["activity_max"]}],
            "reason": f"Largest feature PSI {p['psi']['activity_max']} is under {p['psi_drift']}; batches are random draws."}


def test_valid_output_passes():
    p = payload()
    assert validate_output(live_output(p), p) == (None, [])


@pytest.mark.parametrize("change, reason", [
    ({"evidence": [{"field": "psi.made_up", "value": 0.1}]}, UNKNOWN_FIELD),
    ({"evidence": [{"field": "psi.activity_max", "value": 0.987}]}, UNKNOWN_FIELD),  # right field, wrong value
    ({"reason": "PSI 0.987 shows a shift."}, NUMBER_NOT_IN_INPUT),
    ({"reason": "These users look 15 years old."}, AGE_CLAIM),
])
def test_bad_outputs_are_rejected(change, reason):
    p = payload()
    got, errors = validate_output(live_output(p) | change, p)
    assert got == reason and errors


# --- run_a1: every path ----------------------------------------------------------------------------------

def test_live_reply_is_used_with_the_decision_path_timeout(tmp_path):
    p = payload()
    client = FakeClient(reply(live_output(p)))
    result, records = run(p, client, tmp_path)
    assert (result.status, result.output, result.fallback_reason) == (LIVE, live_output(p), None)
    assert client.options == [{"timeout": base.TIMEOUT_S, "max_retries": 0}] and base.TIMEOUT_S == 4.0
    (kw,) = client.calls
    assert kw["system"] == SYSTEM and kw["messages"] == [{"role": "user", "content": user_message(p)}]
    assert kw["output_config"]["format"]["schema"] == anthropic.transform_schema(A1Output)
    assert '"fields":' in kw["messages"][0]["content"]  # the citable paths are in the prompt
    assert [(r["kind"], r["status"]) for r in records] == [("model", LIVE)]


def test_offline_uses_the_threshold_rule(tmp_path):
    p = payload()
    result, records = run(p, None, tmp_path)
    assert (result.status, result.fallback_reason, result.output) == (FALLBACK, OFFLINE, fallback_output(p))
    assert round_agent_summary(records, 3, "test-run")["A1"]["status"] == FALLBACK


def test_round_zero_and_one_are_insufficient_without_a_call(tmp_path):
    none = pd.DataFrame(columns=FEATURE_COLS)
    client = FakeClient(reply({"drift": "real", "evidence": [], "reason": "x"}))
    for p in (a1_input(none, none, None, AUDIT, [], 0, 0.25), a1_input(none, rows(), None, AUDIT, [], 1, 0.25)):
        result, _ = run(p, client, tmp_path, rnd=p["round"])
        assert (result.status, result.fallback_reason, result.output["drift"]) == (FALLBACK, INSUFFICIENT, INSUFFICIENT)
    assert client.calls == []


@pytest.mark.parametrize("make, reason", [
    (lambda p: FakeClient(raises=anthropic.APITimeoutError(request=REQ)), TIMEOUT),
    (lambda p: FakeClient(raises=anthropic.APIConnectionError(request=REQ)), CONNECTION),
    (lambda p: FakeClient(raises=anthropic.RateLimitError("slow", response=httpx2.Response(429, request=REQ), body=None)), API_ERROR),
    (lambda p: FakeClient(raises=TypeError("Could not resolve authentication method")), API_ERROR),
    (lambda p: FakeClient(reply(stop_reason="refusal", text='{"drift": "re')), REFUSAL),
    (lambda p: FakeClient(reply(live_output(p), stop_reason="max_tokens")), INVALID),
    (lambda p: FakeClient(reply(live_output(p) | {"drift": "maybe"})), INVALID),
    (lambda p: FakeClient(reply(live_output(p) | {"reason": "x" * 400})), INVALID),
    (lambda p: FakeClient(SimpleNamespace(stop_reason="end_turn")), INVALID),
    (lambda p: FakeClient(reply(live_output(p) | {"reason": "PSI 0.987 shows a shift."})), NUMBER_NOT_IN_INPUT),
    (lambda p: FakeClient(reply(live_output(p) | {"evidence": [{"field": "audit.adults", "value": 99}]})), UNKNOWN_FIELD),
])
def test_every_failure_falls_back_to_the_threshold_rule(tmp_path, make, reason):
    p = payload()
    before = copy.deepcopy(p)
    result, records = run(p, make(p), tmp_path)
    assert (result.status, result.fallback_reason) == (FALLBACK, reason)
    assert result.output == fallback_output(p) and result.errors and p == before
    assert all(r["status"] == FALLBACK for r in records)
