"""A4 verify-band triager (Tier 3, step 16): contract, information barrier, checks, fallbacks, live path.

No test calls the network. Live replies come from a fake client, and one test drives the real anthropic SDK
through a mocked HTTP transport so the request shape the API receives is checked too.
"""
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
from softsignal.agents.a4_triage import (
    SYSTEM, fallback_output, run_a4, user_message, validate_output,
)
from softsignal.agents.base import (
    AGE_CLAIM, API_ERROR, CONNECTION, FALLBACK, INSUFFICIENT, INVALID, LIVE, NUMBER_NOT_IN_INPUT, OFFLINE, REFUSAL,
    TIMEOUT, UNKNOWN_FIELD, age_claims, merge_block, numbers_in, numbers_not_in_input,
)
from softsignal.agents.contracts import (
    A4_COLS, LABEL_KEYS, TEST_METRIC_KEYS, BarrierError, a4_input, check_barrier, frozen_test_ids,
)
from softsignal.agents.schemas import A4_MAX_NOTE_CHARS, A4Output
from softsignal.data import load_data
from softsignal.features import ID_COL, TARGET
from softsignal.ui_loop import plain, valid_decision

from agent_fakes import REQ, FakeClient, reply

CHIPS = [("logit_text_score", "writes like a teen"), ("night_notification_open_rate", "opens notifications at night"),
         ("pct_active_school_hours", "quiet in school hours"), ("share_news_views", "reads little news")]


def frame(n=8, n_none=2, seed=0) -> pd.DataFrame:
    """explain.apply_bands-shaped rows: n verify accounts then n_none unflagged ones, plus columns A4 must drop."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n + n_none):
        picks = rng.choice(len(CHIPS), 3, replace=False)
        vals = np.round(rng.uniform(0.2, 6, 3), 4)
        row = {ID_COL: f"T{i:04d}", "score": float(rng.uniform(0.9, 1) if i < n else rng.uniform(0, 0.3)),
               "band": "verify" if i < n else "none", "words": ", ".join(rng.choice(["lol", "school", "im", "omg"], 2, replace=False)),
               TARGET: int(rng.integers(0, 2)), "age": 15, "in_audit": True, "action": "x", "reason": "y"}
        for k, (j, v) in enumerate(zip(picks, vals), start=1):
            f, phrase = CHIPS[j]
            row |= {f"f{k}": f, f"c{k}": f"{phrase} {v:+.2f}", f"v{k}": v}
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def payload():
    df = frame()
    return a4_input(df, df.loc[df["band"] == "verify", ID_COL], round_id=3, test_ids=[])


def good_output(payload) -> dict:
    s = payload["signals"][0]
    return {"batch_reason": f"{payload['n_accounts']} accounts, mostly {s['signal']} ({s['n_accounts']} accounts). "
                            f"Scores run {payload['score_min']} to {payload['score_max']}.",
            "based_on": [s["feature"]]}


# --- contract and information barrier -------------------------------------------------------------

def test_input_is_a_batch_summary_only(payload):
    assert set(payload) == {"agent", "round", "band", "n_accounts", "score_min", "score_median", "score_max",
                            "signals", "top_words"}  # no per-account rows: no ranks 1..n, no single-account values
    assert set(payload["signals"][0]) == {"feature", "signal", "n_accounts", "share_pct", "mean_contribution",
                                          "n_leading"}
    text = json.dumps(payload)
    assert "T000" not in text  # no account ids
    for key in (TARGET, "age", "in_audit", "reason", "action"):  # label and extra frame columns
        assert f'"{key}"' not in text


def test_input_never_carries_a_label_even_if_the_frame_does():
    df = frame()
    a = a4_input(df, df.loc[df["band"] == "verify", ID_COL], 1, test_ids=[])
    df[TARGET] = 1 - df[TARGET]
    b = a4_input(df, df.loc[df["band"] == "verify", ID_COL], 1, test_ids=[])
    assert a == b


def test_input_refuses_frozen_test_accounts():
    _, test = load_data(on_param_mismatch="error")
    df = frame()
    df.loc[0, ID_COL] = test[ID_COL].iloc[0]
    with pytest.raises(BarrierError, match="frozen test"):
        a4_input(df, df.loc[df["band"] == "verify", ID_COL], 1)  # default: results/split.json
    with pytest.raises(BarrierError):
        a4_input(df, [df.loc[0, ID_COL]], 1, test_ids=[df.loc[0, ID_COL]])


def test_frozen_test_ids_are_the_split_test_set():
    train, test = load_data(on_param_mismatch="error")
    ids = frozen_test_ids()
    assert ids == set(test[ID_COL]) and not ids & set(train[ID_COL])


def test_input_needs_verify_band_accounts_from_the_frame():
    df = frame()
    with pytest.raises(ValueError, match="verify band"):
        a4_input(df, [df.loc[df["band"] == "none", ID_COL].iloc[0]], 1, test_ids=[])
    with pytest.raises(ValueError, match="exactly once"):
        a4_input(df, ["NOPE"], 1, test_ids=[])
    with pytest.raises(ValueError, match="missing columns"):
        a4_input(df.drop(columns="words"), [], 1, test_ids=[])


def test_input_aggregates_are_counts_of_the_same_fields(payload):
    df = frame()
    v = df[df["band"] == "verify"]
    assert payload["n_accounts"] == len(v)
    assert (payload["score_min"], payload["score_median"], payload["score_max"]) == (
        round(v["score"].min(), 2), round(v["score"].median(), 2), round(v["score"].max(), 2))
    assert sum(s["n_leading"] for s in payload["signals"]) == len(v)  # every account has one largest signal
    for s in payload["signals"]:
        cells = [(r[f"v{k}"]) for _, r in v.iterrows() for k in (1, 2, 3)
                 if r[f"f{k}"] == s["feature"] and r[f"c{k}"].rsplit(" ", 1)[0] == s["signal"]]
        assert s["n_accounts"] == len(cells) and s["mean_contribution"] == round(float(np.mean(cells)), 2)
        assert s["share_pct"] == round(100 * len(cells) / len(v))
        assert s["n_leading"] == sum(r["f1"] == s["feature"] and r["c1"].rsplit(" ", 1)[0] == s["signal"]
                                     for _, r in v.iterrows())
    counts = [w["n_accounts"] for w in payload["top_words"]]
    assert counts == sorted(counts, reverse=True)
    assert [s["n_accounts"] for s in payload["signals"]] == sorted((s["n_accounts"] for s in payload["signals"]), reverse=True)


def test_check_barrier_finds_nested_forbidden_keys():
    check_barrier({"a": [{"b": 1}], "c": "label_teen is a value, not a key"})
    for key in sorted(LABEL_KEYS | TEST_METRIC_KEYS):
        with pytest.raises(BarrierError):
            check_barrier({"a": [{"b": {key: 0}}]})


def test_test_metric_keys_match_the_loop_tab():
    from softsignal import ui_loop
    assert ui_loop.TEST_METRIC_KEYS is TEST_METRIC_KEYS


def test_empty_verify_band_gives_no_accounts():
    p = a4_input(frame(), [], 0, test_ids=[])
    assert p["n_accounts"] == 0 and p["signals"] == [] and p["score_min"] is None


# --- numbers in input ------------------------------------------------------------------------------

def test_numbers_in_reads_numbers_not_ids_or_keys():
    text = "75 accounts (scores 0.97 to 1.0), 93% share, -0.63 and +7.98, B4155846, c1, R1, 1,000, .5, 3rd."
    assert numbers_in(text) == [75, 0.97, 1.0, 93, -0.63, 7.98, 1000, 0.5, 3]


def test_invented_numbers_are_caught(payload):
    s = payload["signals"][0]
    assert numbers_not_in_input(f"{s['n_accounts']} accounts", payload) == []
    assert numbers_not_in_input(f"{payload['score_min']} and {s['share_pct']}%", payload) == []
    assert numbers_not_in_input("exactly 987654 accounts", payload) == ["987654"]
    assert numbers_not_in_input("about 3.2", {"mean_contribution": 3.21}) == ["3.2"]  # compared as written
    assert numbers_not_in_input("down 0.63", {"v": -0.63}) == []  # by absolute value (documented)


# --- validation and the deterministic fallback --------------------------------------------------------

def test_valid_output_passes(payload):
    assert validate_output(good_output(payload), payload) == (None, [])


@pytest.mark.parametrize("change, reason", [
    ({"based_on": ["age"]}, UNKNOWN_FIELD),
    ({"based_on": ["logit_text_score", "logit_text_score"]}, UNKNOWN_FIELD),
    ({"batch_reason": "Exactly 987654 accounts."}, NUMBER_NOT_IN_INPUT),
    ({"batch_reason": "Most are 15 years old."}, AGE_CLAIM),
])
def test_bad_outputs_are_rejected(payload, change, reason):
    got, errors = validate_output(good_output(payload) | change, payload)
    assert got == reason and errors


@pytest.mark.parametrize("n", [1, 2, 8, 75])
def test_fallback_passes_its_own_checks(n):
    df = frame(n=n, seed=n)
    p = a4_input(df, df.loc[df["band"] == "verify", ID_COL], 1, test_ids=[])
    out = fallback_output(p)
    A4Output(**out)
    assert validate_output(out, p) == (None, [])
    assert out["batch_reason"].startswith(f"{n} accounts sent to verification")


def test_fallback_note_stays_under_the_limit_with_long_signals(payload):
    p = copy.deepcopy(payload)
    for s in p["signals"]:
        s["signal"] = "a very long readable signal phrase " * 6
    out = fallback_output(p)
    assert len(out["batch_reason"]) <= A4_MAX_NOTE_CHARS and validate_output(out, p) == (None, [])


@pytest.fixture(scope="module")
def real_band(tmp_path_factory):
    """75 verify-band accounts of a real train batch, scored by a stack fit on the rest of train."""
    from softsignal.explain import apply_bands, explain_frame
    from softsignal.stack import Stack
    from softsignal.text_model import build_matrix

    train, _ = load_data(on_param_mismatch="error")
    rest, batch = train.iloc[300:], train.iloc[:300]
    tm = build_matrix(train[ID_COL], cache_dir=tmp_path_factory.mktemp("cache"))
    model = Stack.fit(rest, tm=tm, train_ids=train[ID_COL])
    f = apply_bands(explain_frame(model, batch.drop(columns=TARGET)), 0.4, 0.6)
    ids = f.loc[f["band"] == "verify"].sort_values("score", ascending=False)[ID_COL].head(75)
    return f, ids, a4_input(f, ids, 1)


def test_fallback_on_the_real_model_and_data(real_band):
    _, ids, p = real_band
    out = fallback_output(p)
    assert p["n_accounts"] == len(ids) == 75 and validate_output(out, p) == (None, [])
    assert set(out["based_on"]) <= {s["feature"] for s in p["signals"]}


def test_invented_batch_figures_are_rejected_on_real_data(real_band):
    # review finding 1: with per-account ranks in the input, every count 1..75 passed the check
    f, ids, p = real_band
    allowed = set(numbers_in(json.dumps(p)))
    free_count = next(k for k in range(2, 75) if k not in allowed)
    free_pct = next(k for k in range(5, 95) if k not in allowed)
    one_value = next(v for v in f.loc[f[ID_COL].isin(ids), "v1"].round(2) if v not in allowed)
    feature = p["signals"][0]["feature"]
    for note in (f"Roughly {free_count} of the 75 accounts use the word lol.",
                 f"About {free_pct}% of these accounts were active late at night.",
                 f"Their mean contribution is {one_value:.2f}."):
        assert validate_output({"batch_reason": note, "based_on": [feature]}, p)[0] == NUMBER_NOT_IN_INPUT, note
    reason, errors = validate_output(
        {"batch_reason": "At least 3 accounts appear to be 14 years old.", "based_on": [feature]}, p)
    assert reason == AGE_CLAIM and "14 years old" in errors[0]


@pytest.mark.parametrize("text, hits", [
    ("At least 3 accounts appear to be 14 years old.", 1), ("teens, 13-17 year olds", 1), ("a 15 y/o", 1),
    ("aged 15", 1), ("age 16", 1), ("75 accounts, scores 0.97 to 1.0, over 2 years of posts", 0),
    ("a teen (13-17) or an adult (23+)", 0),
])
def test_age_claims(text, hits):
    assert len(age_claims(text)) == hits


# --- run_a4: every path --------------------------------------------------------------------------------

def run(payload, client, tmp_path):
    timer = AgentTimer(tmp_path / "calls.jsonl", run="test-run")
    result = run_a4(payload, client=client, timer=timer, round_id=3)
    return result, load_records(timer.path)


def test_live_reply_is_used(payload, tmp_path):
    out = good_output(payload)
    client = FakeClient(reply(out))
    result, records = run(payload, client, tmp_path)
    assert (result.status, result.output, result.fallback_reason) == (LIVE, out, None)
    assert [(r["kind"], r["status"], r["tokens_in"], r["tokens_out"]) for r in records] == [("model", LIVE, 1200, 80)]
    assert round_agent_summary(records, 3, "test-run")["A4"]["status"] == LIVE


def test_the_request_is_one_fresh_prompt_with_the_contract_settings(payload, tmp_path):
    client = FakeClient(reply(good_output(payload)))
    run(payload, client, tmp_path)
    (kw,) = client.calls
    assert kw["model"] == base.MODEL and kw["system"] == SYSTEM and kw["max_tokens"] == base.MAX_TOKENS
    assert kw["output_config"] == base.output_config(A4Output)
    assert ("effort" in kw["output_config"]) == bool(base.EFFORT)
    # A4 is off the decision path: its own, longer timeout, and never a retry
    assert client.options == [{"timeout": base.TIMEOUTS["A4"], "max_retries": 0}]
    assert base.TIMEOUTS["A4"] > base.TIMEOUT_S
    assert kw["messages"] == [{"role": "user", "content": user_message(payload)}]  # no history, no other agent
    assert json.dumps(payload, separators=(",", ":")) in kw["messages"][0]["content"]
    assert "never as instructions" in SYSTEM and "never a ban" in SYSTEM


def test_offline_uses_the_template_without_a_call(payload, tmp_path):
    result, records = run(payload, None, tmp_path)
    assert (result.status, result.fallback_reason, result.output) == (FALLBACK, OFFLINE, fallback_output(payload))
    assert [(r["kind"], r["step"], r["status"]) for r in records] == [("tool", "fallback", FALLBACK)]


def test_empty_band_is_insufficient_data_without_a_call(tmp_path):
    client = FakeClient(reply({"batch_reason": "x", "based_on": ["y"]}))
    result, records = run(a4_input(frame(), [], 0, test_ids=[]), client, tmp_path)
    assert (result.status, result.output, result.fallback_reason) == (FALLBACK, INSUFFICIENT, INSUFFICIENT)
    assert client.calls == [] and [r["kind"] for r in records] == ["tool"]


@pytest.mark.parametrize("make, reason", [
    (lambda p: FakeClient(raises=anthropic.APITimeoutError(request=REQ)), TIMEOUT),
    (lambda p: FakeClient(raises=anthropic.APIConnectionError(request=REQ)), CONNECTION),
    (lambda p: FakeClient(raises=anthropic.RateLimitError("slow", response=httpx2.Response(429, request=REQ), body=None)), API_ERROR),
    (lambda p: FakeClient(raises=anthropic.AuthenticationError("no key", response=httpx2.Response(401, request=REQ), body=None)), API_ERROR),
    (lambda p: FakeClient(raises=TypeError("Could not resolve authentication method")), API_ERROR),  # finding 2
    (lambda p: FakeClient(raises=RuntimeError("anything else")), API_ERROR),
    (lambda p: FakeClient(reply(None, stop_reason="refusal")), REFUSAL),
    (lambda p: FakeClient(reply(stop_reason="refusal", text='{"batch_reason": "I can')), REFUSAL),  # partial text
    (lambda p: FakeClient(reply(good_output(p), stop_reason="max_tokens")), INVALID),
    (lambda p: FakeClient(reply(None)), INVALID),
    (lambda p: FakeClient(reply(text="not json")), INVALID),
    (lambda p: FakeClient(reply({"batch_reason": "", "based_on": []})), INVALID),
    (lambda p: FakeClient(reply({"batch_reason": "x" * 700, "based_on": ["logit_text_score"]})), INVALID),
    (lambda p: FakeClient(reply(good_output(p) | {"extra": 1})), INVALID),
    (lambda p: FakeClient(SimpleNamespace(stop_reason="end_turn")), INVALID),  # a malformed reply object
    (lambda p: FakeClient(reply(good_output(p) | {"batch_reason": "About 999 teens here."})), NUMBER_NOT_IN_INPUT),
    (lambda p: FakeClient(reply(good_output(p) | {"batch_reason": "They seem 15 years old."})), AGE_CLAIM),
    (lambda p: FakeClient(reply(good_output(p) | {"based_on": ["gender"]})), UNKNOWN_FIELD),
])
def test_every_failure_falls_back_to_the_template(payload, tmp_path, make, reason):
    before = copy.deepcopy(payload)
    result, records = run(payload, make(payload), tmp_path)
    assert (result.status, result.fallback_reason) == (FALLBACK, reason)
    assert result.output == fallback_output(payload) and result.errors
    assert payload == before  # A4 never changes its input
    assert records[-1]["status"] == FALLBACK and records[-1]["step"] == "fallback"
    assert all(r["status"] == FALLBACK for r in records)  # a rejected model reply is not logged LIVE
    assert round_agent_summary(records, 3, "test-run")["A4"]["status"] == FALLBACK


def test_a_rejected_note_is_kept_for_review_but_not_used(payload, tmp_path):
    bad = good_output(payload) | {"batch_reason": "About 999 teens here."}
    result, _ = run(payload, FakeClient(reply(bad)), tmp_path)
    assert result.output == fallback_output(payload) and json.loads(result.rejected) == bad
    assert result.block()["rejected"] == result.rejected
    ok, _ = run(payload, FakeClient(reply(good_output(payload))), tmp_path)
    assert ok.rejected is None


def test_numpy_round_ids_do_not_break_the_prompt(payload):
    df = frame()
    p = a4_input(df, df.loc[df["band"] == "verify", ID_COL], np.int64(4), test_ids=[])
    assert p["round"] == 4 and type(p["round"]) is int
    user_message(payload | {"round": np.int64(4)})  # review: json.dumps raised on numpy ints


def test_real_sdk_request_through_a_mocked_transport(payload, tmp_path):
    seen = {}
    out = good_output(payload)

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": base.MODEL,
            "content": [{"type": "text", "text": json.dumps(out)}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 1500, "output_tokens": 90}})

    client = anthropic.Anthropic(api_key="test-key", max_retries=0, timeout=4.0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    result, records = run(payload, client, tmp_path)
    assert (result.status, result.output) == (LIVE, out) and records[0]["tokens_in"] == 1500
    body = seen["body"]
    assert body["model"] == base.MODEL and body["output_config"].get("effort") == (base.EFFORT or None)
    assert body["output_config"]["format"]["type"] == "json_schema" and body["max_tokens"] == base.MAX_TOKENS
    assert set(body["output_config"]["format"]["schema"]["properties"]) == {"batch_reason", "based_on"}
    assert "temperature" not in body and len(body["messages"]) == 1


# --- client and decisions.jsonl block ---------------------------------------------------------------

def test_make_client_is_offline_without_credentials(monkeypatch, tmp_path):
    for k in base.CREDENTIAL_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("SOFTSIGNAL_OFFLINE", raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "anthropic"))
    assert base.make_client() is None
    creds = tmp_path / "anthropic" / "credentials"
    creds.mkdir(parents=True)  # review finding 2: an empty CLI folder is not a login
    assert not base.has_credentials() and base.make_client() is None
    (creds / "default.json").write_text("{}")
    assert base.has_credentials()
    (creds / "default.json").unlink()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = base.make_client()
    assert client is not None and client.max_retries == 0 and client.timeout == base.TIMEOUT_S == 4.0
    monkeypatch.setenv("SOFTSIGNAL_OFFLINE", "1")
    assert base.make_client() is None


def test_block_is_a_valid_decisions_record_and_renders(payload, tmp_path):
    from streamlit.testing.v1 import AppTest

    result, records = run(payload, FakeClient(reply(good_output(payload))), tmp_path)
    block = merge_block(result, round_agent_summary(records, 3, "test-run")["A4"])
    assert {"status", "output", "fallback_reason", "input_hash", "ms", "tokens_in", "errors"} <= set(block)
    record = {"run": "test-run", "round": 3, "a4": block, "rule_decision": {"cutoff": 0.5}, "diff": {},
              "applied": {"decision": {"cutoff": 0.5}, "source": "rule"}}
    assert valid_decision(json.loads(json.dumps(record)))

    def card(rec):
        from softsignal import ui_loop
        ui_loop.agent_card("a4", rec)

    at = AppTest.from_function(card, kwargs={"rec": record}).run()
    assert not at.exception
    assert any(plain(good_output(payload)["batch_reason"]) == c.value for c in at.caption)
    from softsignal.explain import FEATURE_NAMES
    assert any(c.value == "Based on: " + plain(FEATURE_NAMES[block["output"]["based_on"][0]]) for c in at.caption)


def test_insufficient_a4_card_says_the_band_was_empty(tmp_path):
    from streamlit.testing.v1 import AppTest

    result, _ = run(a4_input(frame(), [], 0, test_ids=[]), None, tmp_path)
    record = {"run": "r", "round": 0, "a4": result.block(), "rule_decision": {}, "diff": {},
              "applied": {"decision": {}, "source": "starter"}}

    def card(rec):
        from softsignal import ui_loop
        ui_loop.agent_card("a4", rec)

    at = AppTest.from_function(card, kwargs={"rec": record}).run()
    assert any("no accounts were sent to verification" in c.value for c in at.caption)


def test_merge_block_keeps_both_error_lists():
    from softsignal.agents.base import AgentResult
    r = AgentResult("A4", FALLBACK, "x", TIMEOUT, "h", ["timeout"])
    merged = merge_block(r, {"status": LIVE, "ms": 5.0, "errors": ["triage: TimeoutError"]})
    assert merged["status"] == FALLBACK and merged["ms"] == 5.0
    assert merged["errors"] == ["triage: TimeoutError", "timeout"]


def test_a4_cols_are_the_ranked_list_columns_it_needs():
    from softsignal.metrics import RANKED_COLS
    assert set(A4_COLS) <= set(RANKED_COLS)


def test_model_text_is_shown_as_typed_not_rendered():
    from streamlit.testing.v1 import AppTest

    note = "See [this](http://evil.example) ![x](http://evil.example/a.png) $x$ **now**"
    record = {"run": "r", "round": 1, "rule_decision": {}, "diff": {}, "applied": {"decision": {}, "source": "rule"},
              "a4": {"status": LIVE, "output": {"batch_reason": note, "based_on": ["logit_text_score"]},
                     "fallback_reason": None}}

    def card(rec):
        from softsignal import ui_loop
        ui_loop.agent_card("a4", rec)

    at = AppTest.from_function(card, kwargs={"rec": record}).run()
    shown = next(c.value for c in at.caption if "evil" in c.value)
    assert "](http" not in shown and r"\[this\]" in shown and r"\$x\$" in shown and r"\*\*now\*\*" in shown
