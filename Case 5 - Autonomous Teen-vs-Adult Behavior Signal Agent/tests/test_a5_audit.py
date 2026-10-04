"""A5 honesty auditor (Tier 3, step 16): sources, the number-matching script, the code gate, every path. No network."""
import copy
import csv
import inspect
import json

import pandas as pd
import pytest

from softsignal.agent_timer import AgentTimer, load_records
from softsignal.agents import base
from softsignal.agents.a5_audit import (
    CANNOT, CHECK_COLS, PROJECTED_V, SLIDE_TIMEOUT_S, SUPPORTED, SYSTEM, UNSUPPORTED_V, as_block_output, check_claims,
    main, risk_tags, row_holds, run_a5, token_matches, user_message, validate_output,
)
from softsignal.agents.base import (
    AGE_CLAIM, FALLBACK, INSUFFICIENT, LIVE, NUMBER_NOT_IN_INPUT, OFFLINE, UNKNOWN_FIELD, UNSUPPORTED,
)
from softsignal.agents.contracts import (
    CLAIMS_FILE, BarrierError, a2_input, a5_input, a5_sources, load_checklist, load_claims, round_claims,
)
from softsignal.data import ROOT
from softsignal.metrics import ROUNDS_COLS

from agent_fakes import FakeClient, reply

RESULTS = ROOT / "results"
SEED = [  # claims/claims.md, in order: (verdict, source)
    (SUPPORTED, "policy_grid.csv:cap=0.15"),
    (SUPPORTED, "policy_grid.csv:cap=0.15"),
    (SUPPORTED, "policy_grid.csv:cap=0.08"),
    (PROJECTED_V, "eval_placeholder.csv:1"),
    (SUPPORTED, "rounds_recorded.csv:R7"),
    (PROJECTED_V, "eval_placeholder.csv:6"),
    (CANNOT, None),  # no number
    (CANNOT, "files: eval.csv, sanity.txt, ablations.csv"),  # no present row holds it, and these files are missing
]


@pytest.fixture(scope="module")
def checklist():
    return load_checklist()


@pytest.fixture(scope="module")
def slides(checklist):
    """The slide pass input over the committed results files."""
    return a5_input(load_claims(), a5_sources(RESULTS), checklist, "slides")


def tiny(claims, rows, checklist=(), kind="measured", file="x.csv"):
    src = [{"file": file, "status": "present",
            "rows": [{"id": f"{file}:{i}", "kind": kind, "values": v} for i, v in enumerate(rows, 1)]}]
    return a5_input(claims, src, list(checklist), "slides")


# --- matching a quoted number with a file value ---------------------------------------------------

@pytest.mark.parametrize("token, value, ok", [
    ("92%", 0.9199, True), ("92%", 0.926, False), ("17.1%", 0.171, True), ("17.1%", 0.1716, False),
    ("0.955", 0.9551, True), ("0.955", 0.9556, False), ("7", 7, True), ("7", 0.07, False), ("15%", 15, True),
    ("15.0%", 0.15, True), ("900", 900.4, True),
])
def test_numbers_match_at_the_precision_the_claim_writes(token, value, ok):
    assert token_matches(token, value) is ok


def test_a_row_must_hold_every_number_of_the_claim():
    row = {"values": {"rec": 0.92, "cap": 0.15, "n": 900, "stage": "x"}}
    assert row_holds("92% at a 15% cap on 900 accounts", row, "x.csv")
    assert not row_holds("92% at a 10% cap", row, "x.csv") and not row_holds("no numbers", row, "x.csv")
    assert row_holds("line says 0.95", {"values": {"line": "auc 0.95"}}, "sanity.txt")  # text files: numbers in the line


# --- sources --------------------------------------------------------------------------------------

def test_sources_mark_projected_rows_and_missing_files(tmp_path):
    pd.DataFrame({"stage": ["a", "b"], "eval_set": ["test", "projected"], "rec": [0.9, 0.8]}).to_csv(tmp_path / "eval.csv", index=False)
    pd.DataFrame({"run": ["r1", "r2", "r2"], "round": [0, 0, 1], "rec": [0.1, 0.2, 0.3]}).to_csv(
        tmp_path / "rounds_recorded.csv", index=False)
    (tmp_path / "sanity.txt").write_text("header\ncoef 2.36\n\nno digits here\n")
    src = {s["file"]: s for s in a5_sources(tmp_path)}
    assert [r["kind"] for r in src["eval.csv"]["rows"]] == ["measured", "projected"]
    assert [r["id"] for r in src["rounds_recorded.csv"]["rows"]] == ["rounds_recorded.csv:R0", "rounds_recorded.csv:R1"]
    assert [r["values"]["rec"] for r in src["rounds_recorded.csv"]["rows"]] == [0.2, 0.3]  # the latest run only
    assert not any("run" in r["values"] for r in src["rounds_recorded.csv"]["rows"])  # run ids left out
    assert [r["id"] for r in src["sanity.txt"]["rows"]] == ["sanity.txt:line 2"]
    assert src["policy_grid.csv"]["status"] == "missing" and src["ablations.csv"]["status"] == "missing"


def test_without_eval_csv_the_placeholder_stands_in_and_is_all_projected():
    src = {s["file"]: s for s in a5_sources(RESULTS)}
    assert src["eval.csv"]["status"] == "missing"  # said out loud, even with the placeholder standing in
    assert {r["kind"] for r in src["eval_placeholder.csv"]["rows"]} == {"projected"}
    assert "policy_grid.csv:cap=0.15" in {r["id"] for r in src["policy_grid.csv"]["rows"]}


def test_a5_reads_test_metrics_but_never_a_label():
    payload = tiny(["recall 92%"], [{"rec": 0.92, "ft": 0.17, "auc": 0.95}])  # test metrics: allowed for A5
    assert payload["sources"][0]["rows"][0]["values"]["rec"] == 0.92
    with pytest.raises(BarrierError):
        tiny(["x 1"], [{"label_teen": 1}])


def test_nothing_from_a5_can_reach_a2():
    assert "a5" not in inspect.signature(a2_input).parameters  # A2's builder has no way in for A5's output


# --- the script on the committed results (the seed claims) ---------------------------------------------

def test_seed_claims_get_their_verdicts(slides):
    got = [(v["verdict"], v["source"]) for v in check_claims(slides)["verdicts"]]
    assert got == SEED


def test_closest_row_wins_over_a_coincidental_match(slides):
    # 50% / 32% also round from a policy_grid row (49.6% / 31.6%); the placeholder row holds them exactly
    keyword = check_claims(slides)["verdicts"][3]
    assert keyword["verdict"] == PROJECTED_V and keyword["source"] == "eval_placeholder.csv:1"


def test_risk_tags_follow_the_checklist(slides, checklist):
    tags = [v["risks"] for v in check_claims(slides)["verdicts"]]
    assert tags[0] == ["over_cap"] and tags[4] == ["loop_below_static"] and tags[6] == ["agents_accuracy"]
    assert risk_tags("A ban is never the action.", checklist) == ["legal_age"]


def test_the_script_passes_its_own_gate(slides):
    assert validate_output(check_claims(slides), slides) == (None, [])
    assert validate_output(as_block_output(check_claims(slides), slides), slides) == (None, [])  # recorded form


# --- the code gate -------------------------------------------------------------------------------------

def edited(payload, i, **kw):
    out = copy.deepcopy(check_claims(payload))
    out["verdicts"][i].update(kw)
    return out


@pytest.mark.parametrize("i, kw, reason", [
    (7, {"verdict": SUPPORTED, "source": "policy_grid.csv:cap=0.15"}, UNSUPPORTED),  # the row lacks "100"
    (5, {"verdict": SUPPORTED}, UNSUPPORTED),  # eval_placeholder.csv:6 is projected
    (3, {"verdict": SUPPORTED, "source": "policy_grid.csv:cap=0.3"}, UNSUPPORTED),  # a closer projected row exists
    (0, {"source": "made_up.csv:1"}, UNKNOWN_FIELD),
    (6, {"source": "nowhere.csv"}, UNKNOWN_FIELD),
    (0, {"risks": ["not_a_risk"]}, UNKNOWN_FIELD),
    (0, {"note": "Checked against 12345 rows."}, NUMBER_NOT_IN_INPUT),
    (0, {"note": "These users are 14 years old."}, AGE_CLAIM),
])
def test_the_gate_refuses_unbacked_verdicts(slides, i, kw, reason):
    got, errors = validate_output(edited(slides, i, **kw), slides)
    assert got == reason and errors


def test_the_gate_wants_exactly_one_verdict_per_claim(slides):
    out = check_claims(slides)
    assert validate_output({"verdicts": out["verdicts"][:-1]}, slides)[0] == UNKNOWN_FIELD
    assert validate_output({"verdicts": out["verdicts"] + out["verdicts"][:1]}, slides)[0] == UNKNOWN_FIELD
    recorded = as_block_output(out, slides)
    recorded[0]["claim"] = "a different claim"
    assert validate_output(recorded, slides)[0] == UNKNOWN_FIELD


# --- run_a5: every path ----------------------------------------------------------------------------------

def run(p, client, tmp_path, **kw):
    timer = AgentTimer(tmp_path / "calls.jsonl", run="t")
    return run_a5(p, client=client, timer=timer, round_id=2, **kw), load_records(timer.path)


def test_offline_runs_the_script_in_the_plan_shape(slides, tmp_path):
    result, records = run(slides, None, tmp_path)
    assert (result.status, result.fallback_reason) == (FALLBACK, OFFLINE)
    assert [(v["verdict"], v["source"]) for v in result.output] == SEED
    assert set(result.output[0]) == {"claim_id", "claim", "verdict", "source", "risks", "note"}  # plan: claim, verdict, source
    assert [(r["kind"], r["status"]) for r in records] == [("tool", FALLBACK)]


def test_a_live_reply_is_used_and_overrules_the_script_with_a_reason(slides, tmp_path):
    live = edited(slides, 2, verdict=UNSUPPORTED_V, source="policy_grid.csv",
                  note="225 is the sent count at every cap, so the slide should name the cap.")
    client = FakeClient(reply(live))
    result, records = run(slides, client, tmp_path)
    assert result.status == LIVE and result.output[2]["verdict"] == UNSUPPORTED_V
    (kw,) = client.calls
    assert kw["system"] == SYSTEM and kw["messages"][0]["content"] == user_message(slides)
    assert '"script_check":' in kw["messages"][0]["content"]  # the script's verdicts are the model's starting point
    assert client.options == [{"timeout": base.TIMEOUTS["A5"], "max_retries": 0}]
    assert [(r["kind"], r["status"]) for r in records] == [("model", LIVE)]


def test_an_unbacked_live_verdict_falls_back_to_the_script(slides, tmp_path):
    bad = edited(slides, 5, verdict=SUPPORTED)  # projected row called supported
    result, _ = run(slides, FakeClient(reply(bad)), tmp_path)
    assert (result.status, result.fallback_reason) == (FALLBACK, UNSUPPORTED)
    assert [(v["verdict"], v["source"]) for v in result.output] == SEED and json.loads(result.rejected) == bad


def test_the_slide_pass_gets_a_longer_timeout(slides, tmp_path):
    client = FakeClient(reply(check_claims(slides)))
    run(slides, client, tmp_path, timeout=SLIDE_TIMEOUT_S)
    assert client.options == [{"timeout": SLIDE_TIMEOUT_S, "max_retries": 0}]


def test_no_claims_is_insufficient_data(tmp_path):
    result, _ = run(tiny([], [{"a": 1}]), FakeClient(reply({"verdicts": []})), tmp_path)
    assert (result.status, result.fallback_reason, result.output) == (FALLBACK, INSUFFICIENT, [])


# --- per round and the slide pass -------------------------------------------------------------------------

def test_round_claims_are_the_headline_and_a2s_reason():
    row = dict.fromkeys(ROUNDS_COLS, 0) | {"round": 7, "mode": "ACTIVE", "action": "promote", "rec": 0.887,
                                            "ft": 0.169, "n_audit_adults": 213}
    assert round_claims(row, {}) == ["Round 7: ACTIVE, promote; test recall 88.7% at 16.9% false-teen; "
                                     "213 audit adults so far."]
    two = round_claims(row, {"a2": {"output": {"reason": "Promote: 213 audit adults."}}})
    assert two[1] == "A2: Promote: 213 audit adults."
    rounds = pd.DataFrame([row | {"run": "20261004T000000Z-x", "refit_s": 2.13}], columns=ROUNDS_COLS)
    p = a5_input(two, a5_sources(RESULTS, rounds, "rounds.csv", files=("rounds",)), [], "round", 7)
    assert [s["file"] for s in p["sources"]] == ["rounds.csv"]  # per round: this run's rows only
    assert not {"run", "refit_s"} & set(p["sources"][0]["rows"][0]["values"])  # per-run values: no stable hash
    assert [v["verdict"] for v in check_claims(p)["verdicts"]] == [SUPPORTED, SUPPORTED]


def test_slide_pass_writes_claims_check(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTSIGNAL_OFFLINE", "1")
    out = tmp_path / "claims_check.csv"
    main(["--out", str(out)])
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0]) == CHECK_COLS and len(rows) == len(load_claims(CLAIMS_FILE))
    assert [(r["verdict"], r["source"] or None) for r in rows] == SEED and rows[0]["status"] == FALLBACK


def test_committed_claims_check_matches_the_script():
    rows = list(csv.DictReader((RESULTS / "claims_check.csv").open()))
    assert [(r["verdict"], r["source"] or None) for r in rows] == SEED


# --- review round 1: metrics, missing files, sources, failure paths -----------------------------------------------

def test_a_named_metric_must_be_the_column_the_number_is_in(slides):
    p = tiny(["The stack AUC is 0.50."], [{"t_verify": 0.5, "auc": 0.82}])
    assert check_claims(p)["verdicts"][0]["verdict"] == UNSUPPORTED_V  # 0.50 is a cutoff here, not an AUC
    assert check_claims(tiny(["The cutoff is 0.50."], [{"t_verify": 0.5, "auc": 0.82}]))["verdicts"][0]["verdict"] == SUPPORTED
    probe = a5_input(["With shuffled labels the stack AUC drops to 0.50."], slides["sources"], [], "slides")
    v = check_claims(probe)["verdicts"][0]
    assert (v["verdict"], v["source"]) == (CANNOT, "files: eval.csv, sanity.txt, ablations.csv")


def test_with_every_file_present_an_unheld_claim_is_unsupported_and_names_its_source():
    p = tiny(["Recall is 97%."], [{"rec": 0.92}, {"rec": 0.88}])
    v = check_claims(p)["verdicts"][0]
    assert (v["verdict"], v["source"]) == (UNSUPPORTED_V, "files: x.csv")
    assert validate_output({"verdicts": [{**v, "source": None}]}, p)[0] == UNKNOWN_FIELD  # every verdict names a source


@pytest.mark.parametrize("raises, reason", [
    (lambda: __import__("anthropic").APITimeoutError(request=__import__("agent_fakes").REQ), "timeout"),
    (lambda: TypeError("Could not resolve authentication method"), "api_error"),
])
def test_a5_failures_fall_back_to_the_script(slides, tmp_path, raises, reason):
    result, _ = run(slides, FakeClient(raises=raises()), tmp_path)
    assert (result.status, result.fallback_reason) == (FALLBACK, reason)
    assert [(v["verdict"], v["source"]) for v in result.output] == SEED


def test_a_refusal_with_partial_text_falls_back(slides, tmp_path):
    result, _ = run(slides, FakeClient(reply(stop_reason="refusal", text='{"verdicts": [')), tmp_path)
    assert (result.status, result.fallback_reason) == (FALLBACK, "refusal")


def test_the_per_round_timeout_is_the_live_one():
    assert base.TIMEOUTS["A5"] == base.TIMEOUT_S == 4.0 and SLIDE_TIMEOUT_S == 60.0
