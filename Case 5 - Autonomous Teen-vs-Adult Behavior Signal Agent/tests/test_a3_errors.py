"""A3 error analyst (Tier 3, step 16): schema and input contract (sub-step 1).

No test calls the network or reads the real data.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from softsignal.agents.contracts import (
    A3_COLS, A3_ERROR_TYPES, A3_MAX_SIGNALS, LABEL_KEYS, TEST_METRIC_KEYS, BarrierError, a3_fields, a3_input,
    check_barrier,
)
from softsignal.agents.schemas import A3Output
from softsignal.features import ID_COL

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
