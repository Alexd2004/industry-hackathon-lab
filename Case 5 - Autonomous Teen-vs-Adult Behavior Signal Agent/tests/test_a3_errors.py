"""A3 error analyst (Tier 3, step 16): schema, input contract, checks and every run path.

No test calls the network or reads the real data. Live replies come from a fake client.
"""
import copy
from types import SimpleNamespace

import anthropic
import httpx2
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from softsignal.agent_timer import AgentTimer, load_records, round_agent_summary
from softsignal.agents import base
from softsignal.agents.a3_errors import (
    SYSTEM, fallback_output, insufficient_reason, run_a3, user_message, validate_output,
)
from softsignal.agents.base import (
    AGE_CLAIM, API_ERROR, CONNECTION, FALLBACK, FORBIDDEN_COLUMN, INSUFFICIENT, INVALID, LIVE, NUMBER_NOT_IN_INPUT,
    OFFLINE, REFUSAL, TIMEOUT, UNKNOWN_FIELD, UNSUPPORTED,
)
from softsignal.agents.contracts import (
    A3_COLS, A3_ERROR_TYPES, A3_MAX_SIGNALS, LABEL_KEYS, TEST_METRIC_KEYS, BarrierError, a3_fields, a3_input,
    check_barrier,
)
from softsignal.agents.schemas import A3Output
from softsignal.features import ID_COL

from agent_fakes import REQ, FakeClient, reply

CHIPS = [("logit_text_score", "writes like a teen"), ("night_notification_open_rate", "opens notifications at night"),
         ("pct_active_school_hours", "quiet in school hours"), ("share_news_views", "reads little news")]
T_VERIFY = 0.6


def frame(n=40, seed=0) -> tuple[pd.DataFrame, dict]:
    """explain_frame-shaped audit rows and their labels: alternating adults and teens, scores spread over 0-1."""
    rng = np.random.default_rng(seed)
    rows, labels = [], {}
    for i in range(n):
        picks = rng.choice(len(CHIPS), 3, replace=False)
        vals = np.round(rng.uniform(0.2, 6, 3), 4)
        row = {ID_COL: f"A{i:04d}", "score": float(rng.uniform(0, 1)),
               "words": ", ".join(rng.choice(["lol", "school", "im", "omg"], 2, replace=False))}
        for k, (j, v) in enumerate(zip(picks, vals), start=1):
            f, phrase = CHIPS[j]
            row |= {f"f{k}": f, f"c{k}": f"{phrase} {v:+.2f}", f"v{k}": v}
        rows.append(row)
        labels[row[ID_COL]] = i % 2
    return pd.DataFrame(rows), labels


@pytest.fixture
def parts():
    return frame()


@pytest.fixture
def payload(parts):
    df, labels = parts
    return a3_input(df, labels, T_VERIFY, round_id=4, min_errors=5, test_ids=[])


def expected_errors(df, labels, t=T_VERIFY):
    y = df[ID_COL].map(labels).to_numpy()
    s = df["score"].to_numpy()
    return int(((y == 0) & (s >= t)).sum()), int(((y == 1) & (s < t)).sum())


def test_error_definition_uses_t_verify_for_both_types(parts, payload):
    false_teen, missed_teen = expected_errors(*parts)
    assert (payload["n_errors"]["false_teen"], payload["n_errors"]["missed_teen"]) == (false_teen, missed_teen)
    assert payload["false_teen"]["n_accounts"] == false_teen and payload["missed_teen"]["n_accounts"] == missed_teen
    assert false_teen > 0 and missed_teen > 0  # the fixture exercises both


def test_a_teen_between_t_soft_and_t_verify_is_missed():
    df, labels = frame(2)
    df["score"] = [0.1, 0.59]  # A0000 adult, A0001 teen scored just under t_verify
    p = a3_input(df, labels, T_VERIFY, 3, 1, test_ids=[])
    assert p["n_errors"] == {"false_teen": 0, "missed_teen": 1}


def test_score_equal_to_t_verify_is_flagged():
    df, labels = frame(2)
    df["score"] = [T_VERIFY, T_VERIFY]  # adult at t_verify is a false teen, teen at t_verify is caught
    p = a3_input(df, labels, T_VERIFY, 3, 1, test_ids=[])
    assert p["n_errors"] == {"false_teen": 1, "missed_teen": 0}


def test_rates_use_their_own_base(parts, payload):
    false_teen, missed_teen = expected_errors(*parts)
    assert payload["audit"] == {"adults": 20, "teens": 20}
    assert payload["false_teen"]["rate"] == round(false_teen / 20, 3)
    assert payload["missed_teen"]["rate"] == round(missed_teen / 20, 3)


def test_no_labels_ids_or_test_metrics_in_payload(parts, payload):
    df, labels = parts
    text = str(payload)
    assert not any(i in text for i in labels)
    check_barrier(payload)

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    assert not (set(keys(payload)) & (LABEL_KEYS | TEST_METRIC_KEYS))


def test_extra_frame_columns_never_copied(parts):
    df, labels = parts
    df = df.assign(label_teen=1, age=15, band="verify", in_audit=True)
    p = a3_input(df, labels, T_VERIFY, 4, 5, test_ids=[])
    assert "'age'" not in str(p) and "label_teen" not in str(p) and "'band'" not in str(p)


def test_frozen_test_account_raises(parts):
    df, labels = parts
    with pytest.raises(BarrierError):
        a3_input(df, labels, T_VERIFY, 4, 5, test_ids=["A0003"])


def test_missing_column_raises(parts):
    df, labels = parts
    with pytest.raises(ValueError, match="missing columns"):
        a3_input(df.drop(columns=["words"]), labels, T_VERIFY, 4, 5, test_ids=[])


def test_labels_must_match_frame_exactly(parts):
    df, labels = parts
    with pytest.raises(ValueError, match="exactly"):
        a3_input(df, {k: v for k, v in list(labels.items())[1:]}, T_VERIFY, 4, 5, test_ids=[])
    with pytest.raises(ValueError, match="exactly"):
        a3_input(pd.concat([df, df.iloc[:1]]), labels, T_VERIFY, 4, 5, test_ids=[])
    with pytest.raises(ValueError, match="0 or 1"):
        a3_input(df, {**labels, "A0000": 2}, T_VERIFY, 4, 5, test_ids=[])


def test_nan_t_verify_raises(parts):
    df, labels = parts
    with pytest.raises(ValueError, match="NaN"):
        a3_input(df, labels, float("nan"), 4, 5, test_ids=[])


def test_no_t_verify_means_no_errors(parts):
    df, labels = parts
    p = a3_input(df, labels, None, 4, 5, test_ids=[])
    assert p["t_verify"] is None and p["n_errors"] == {"false_teen": 0, "missed_teen": 0}
    assert p["false_teen"]["signals"] == [] and p["false_teen"]["rate"] == 0.0


def test_empty_frame_gives_zero_counts():
    p = a3_input(pd.DataFrame(columns=A3_COLS), {}, T_VERIFY, 0, 5, test_ids=[])
    assert p["audit"] == {"adults": 0, "teens": 0} and p["n_errors"] == {"false_teen": 0, "missed_teen": 0}
    assert p["false_teen"]["rate"] is None and p["false_teen"]["score_median"] is None


def test_signals_are_counted_and_sorted(parts, payload):
    df, labels = parts
    y = df[ID_COL].map(labels).to_numpy()
    s = df["score"].to_numpy()
    rows = df[(y == 0) & (s >= T_VERIFY)]
    block = payload["false_teen"]
    assert block["n_accounts"] == len(rows)
    sig = block["signals"]
    assert 0 < len(sig) <= A3_MAX_SIGNALS
    counts = [x["n_accounts"] for x in sig]
    assert counts == sorted(counts, reverse=True)
    for x in sig:
        have = sum(any(f == x["feature"] and c.rsplit(" ", 1)[0] == x["signal"]
                       for f, c in ((r.f1, r.c1), (r.f2, r.c2), (r.f3, r.c3))) for r in rows.itertuples())
        assert x["n_accounts"] == have and x["share_pct"] == round(100 * have / len(rows))
        assert x["id"].startswith(x["feature"] + "__") and "." not in x["id"]


def test_top_words_counts(parts, payload):
    df, labels = parts
    y = df[ID_COL].map(labels).to_numpy()
    s = df["score"].to_numpy()
    rows = df[(y == 1) & (s < T_VERIFY)]
    for w in payload["missed_teen"]["top_words"]:
        assert w["n_accounts"] == sum(w["word"] in str(x).split(", ") for x in rows["words"])


def test_fields_cover_every_number_and_skip_none(payload):
    f = a3_fields(payload)
    assert f["audit.adults"] == 20.0 and f["t_verify"] == T_VERIFY and f["min_errors"] == 5.0
    assert f["false_teen.n_accounts"] == float(payload["n_errors"]["false_teen"])
    for kind in A3_ERROR_TYPES:
        for s in payload[kind]["signals"]:
            assert f[f"{kind}.signals.{s['id']}.n_accounts"] == float(s["n_accounts"])
    none = a3_input(pd.DataFrame(columns=A3_COLS), {}, None, 0, None, test_ids=[])
    assert "t_verify" not in a3_fields(none) and "min_errors" not in a3_fields(none)
    assert "false_teen.rate" not in a3_fields(none)


def test_payload_is_deterministic_and_json_plain(parts):
    import json

    df, labels = parts
    a = a3_input(df, labels, T_VERIFY, 4, 5, test_ids=[])
    b = a3_input(df.sample(frac=1, random_state=1), labels, T_VERIFY, 4, 5, test_ids=[])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_schema_accepts_a_valid_reply_and_rejects_bad_ones():
    good = {"status": "ok", "patterns": [{"error_type": "false_teen", "description": "late night openers",
                                           "n_accounts": 3, "evidence": [{"field": "false_teen.n_accounts",
                                                                         "value": 3}]}],
            "suggested_param_changes": [{"param": "cap", "direction": "down", "reason": "3 false teens"}]}
    assert A3Output.model_validate(good).status == "ok"
    for mutate in (lambda d: d["suggested_param_changes"][0].update(param="age"),
                   lambda d: d["patterns"][0].update(n_accounts=0),
                   lambda d: d["patterns"][0].update(n_accounts=2.5),
                   lambda d: d["patterns"][0].update(evidence=[]),
                   lambda d: d.update(status="maybe"),
                   lambda d: d.update(extra=1)):
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(ValidationError):
            A3Output.model_validate(bad)


# --- checks and run paths ---------------------------------------------------------------------------------

def live_output(p, kind="false_teen"):
    """A reply that passes every check: one pattern citing the type's own count, one advisory change."""
    n = p[kind]["n_accounts"]
    sig = p[kind]["signals"][0]
    return {"status": "ok",
            "patterns": [{"error_type": kind,
                          "description": f"{sig['signal']} shows in {sig['n_accounts']} of {n} {kind} accounts.",
                          "n_accounts": n,
                          "evidence": [{"field": f"{kind}.n_accounts", "value": n},
                                       {"field": f"{kind}.signals.{sig['id']}.n_accounts",
                                        "value": sig["n_accounts"]}]}],
            "suggested_param_changes": [{"param": "cap", "direction": "down",
                                         "reason": f"{p['n_errors'][kind]} errors at t_verify {p['t_verify']}."}]}


def run(p, client, tmp_path, rnd=4):
    timer = AgentTimer(tmp_path / "calls.jsonl", run="test-run")
    return run_a3(p, client=client, timer=timer, round_id=rnd), load_records(timer.path)


def with_pattern(p, **change):
    out = live_output(p)
    out["patterns"][0].update(change)
    return out


def test_valid_output_passes(payload):
    assert validate_output(live_output(payload), payload) == (None, [])
    assert validate_output(live_output(payload, "missed_teen"), payload) == (None, [])


def test_insufficient_status_with_empty_lists_passes(payload):
    assert validate_output(fallback_output(), payload) == (None, [])


@pytest.mark.parametrize("make, reason", [
    (lambda p: with_pattern(p, description="Teens aged 15 write like this."), AGE_CLAIM),
    (lambda p: with_pattern(p, description="The job field explains it."), FORBIDDEN_COLUMN),
    (lambda p: with_pattern(p, description="account_age_days is low."), FORBIDDEN_COLUMN),
    (lambda p: with_pattern(p, description="Seen in 99 accounts."), NUMBER_NOT_IN_INPUT),
    (lambda p: {**live_output(p), "suggested_param_changes": [
        {"param": "cap", "direction": "up", "reason": "Raise it by 7777."}]}, NUMBER_NOT_IN_INPUT),
    (lambda p: with_pattern(p, n_accounts=p["false_teen"]["n_accounts"] - 1), UNSUPPORTED),
    (lambda p: with_pattern(p, error_type="missed_teen"), UNSUPPORTED),  # cites false_teen counts for missed_teen
    (lambda p: with_pattern(p, evidence=[{"field": "audit.adults", "value": 99}]), UNKNOWN_FIELD),
    (lambda p: with_pattern(p, evidence=[{"field": "made_up.path", "value": 1}]), UNKNOWN_FIELD),
    (lambda p: with_pattern(p, evidence=[{"field": "audit.adults", "value": 20}]), UNSUPPORTED),  # not a count
    (lambda p: {**live_output(p), "patterns": []}, UNSUPPORTED),
    (lambda p: {**fallback_output(), "suggested_param_changes": live_output(p)["suggested_param_changes"]}, INVALID),
    (lambda p: {**live_output(p), "suggested_param_changes": live_output(p)["suggested_param_changes"] * 2},
     UNSUPPORTED),
])
def test_bad_outputs_are_rejected(payload, make, reason):
    got, errors = validate_output(make(payload), payload)
    assert got == reason and errors


def test_insufficient_reasons(parts):
    df, labels = parts
    ok = a3_input(df, labels, T_VERIFY, 4, 5, test_ids=[])
    assert insufficient_reason(ok) is None
    assert insufficient_reason(a3_input(df, labels, T_VERIFY, 4, None, test_ids=[])) == "no min_a3_errors in policy.yaml"
    assert insufficient_reason(a3_input(df, labels, None, 4, 5, test_ids=[])) == "no live t_verify"
    assert "fewer than" in insufficient_reason(a3_input(df, labels, T_VERIFY, 4, 500, test_ids=[]))
    empty = a3_input(pd.DataFrame(columns=A3_COLS), {}, T_VERIFY, 0, 5, test_ids=[])
    assert "no audit labels" in insufficient_reason(empty)
    n = sum(ok["n_errors"].values())  # the floor is inclusive
    assert insufficient_reason(a3_input(df, labels, T_VERIFY, 4, n, test_ids=[])) is None
    assert insufficient_reason(a3_input(df, labels, T_VERIFY, 4, n + 1, test_ids=[])) is not None


def test_live_reply_is_used_with_the_decision_path_timeout(payload, tmp_path):
    client = FakeClient(reply(live_output(payload)))
    result, records = run(payload, client, tmp_path)
    assert (result.status, result.output, result.fallback_reason) == (LIVE, live_output(payload), None)
    assert client.options == [{"timeout": base.TIMEOUT_S, "max_retries": 0}]
    (kw,) = client.calls
    assert kw["system"] == SYSTEM and kw["messages"] == [{"role": "user", "content": user_message(payload)}]
    assert kw["output_config"]["format"]["schema"] == anthropic.transform_schema(A3Output)
    assert '"fields":' in kw["messages"][0]["content"]
    assert [(r["kind"], r["status"]) for r in records] == [("model", LIVE)]


def test_prompt_holds_no_id_label_or_forbidden_input(parts, payload):
    text = user_message(payload)
    assert not any(i in text for i in parts[1])
    for word in ("label_teen", "is_teen", "blogger_id"):
        assert word not in text


def test_offline_is_no_analysis(payload, tmp_path):
    result, records = run(payload, None, tmp_path)
    assert (result.status, result.fallback_reason, result.output) == (FALLBACK, OFFLINE, fallback_output())
    assert round_agent_summary(records, 4, "test-run")["A3"]["status"] == FALLBACK


def test_insufficient_input_makes_no_call(parts, tmp_path):
    df, labels = parts
    client = FakeClient(reply(fallback_output()))
    for p in (a3_input(df, labels, T_VERIFY, 4, None, test_ids=[]),
              a3_input(df, labels, T_VERIFY, 4, 500, test_ids=[]),
              a3_input(pd.DataFrame(columns=A3_COLS), {}, T_VERIFY, 0, 5, test_ids=[])):
        result, _ = run(p, client, tmp_path, rnd=p["round"])
        assert (result.status, result.fallback_reason, result.output["status"]) == (FALLBACK, INSUFFICIENT, INSUFFICIENT)
    assert client.calls == []


@pytest.mark.parametrize("make, reason", [
    (lambda p: FakeClient(raises=anthropic.APITimeoutError(request=REQ)), TIMEOUT),
    (lambda p: FakeClient(raises=anthropic.APIConnectionError(request=REQ)), CONNECTION),
    (lambda p: FakeClient(raises=anthropic.RateLimitError("slow", response=httpx2.Response(429, request=REQ),
                                                           body=None)), API_ERROR),
    (lambda p: FakeClient(reply(stop_reason="refusal", text='{"status": "o')), REFUSAL),
    (lambda p: FakeClient(reply(live_output(p), stop_reason="max_tokens")), INVALID),
    (lambda p: FakeClient(reply(live_output(p) | {"status": "maybe"})), INVALID),
    (lambda p: FakeClient(SimpleNamespace(stop_reason="end_turn")), INVALID),
    (lambda p: FakeClient(reply(with_pattern(p, description="Seen in 99 accounts."))), NUMBER_NOT_IN_INPUT),
    (lambda p: FakeClient(reply(with_pattern(p, description="The gender gap."))), FORBIDDEN_COLUMN),
])
def test_every_failure_is_no_analysis(payload, tmp_path, make, reason):
    before = copy.deepcopy(payload)
    result, records = run(payload, make(payload), tmp_path)
    assert (result.status, result.fallback_reason) == (FALLBACK, reason)
    assert result.output == fallback_output() and result.errors and payload == before
    assert all(r["status"] == FALLBACK for r in records)


def test_system_prompt_is_fully_rendered_and_states_the_signal_caveat():
    assert "{" not in SYSTEM and "}" not in SYSTEM  # an f-string placeholder must not reach the model unfilled
    assert "up to 4 items" in SYSTEM  # A3_MAX_PATTERNS, the schema's own limit
    assert "fit on these same accounts" in SYSTEM and "do not show why" in SYSTEM.lower().replace("\n", " ")
    assert "caused" in SYSTEM  # no causal claims about a signal
