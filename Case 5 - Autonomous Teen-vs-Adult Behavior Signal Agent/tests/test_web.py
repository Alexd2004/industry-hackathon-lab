"""Web console (softsignal/web): the JSON the page reads and the server's routes. No network, no model call."""
import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from softsignal.metrics import ROUNDS_COLS
from softsignal.web import payload, server


def row(run, rnd, **kw):
    base = {c: None for c in ROUNDS_COLS}
    base.update(run=run, round=rnd, mode="SHADOW", action="hold", applied_source="rule", diff_count=0, cap=0.15,
                t_verify=0.5, n_flagged=0, n_verify=0, n_labels=0, n_audit_adults=0, prec=0.7, rec=0.8, ft=0.35,
                mt=0.2, auc=0.82)
    return base | kw


def write_rounds(path, rows):
    pd.DataFrame(rows, columns=ROUNDS_COLS).to_csv(path, index=False, lineterminator="\n")


# --- runs --------------------------------------------------------------------------------------------------------

def test_runs_from_all_three_sources(tmp_path):
    write_rounds(tmp_path / "rounds_recorded.csv", [row("r1", 0), row("r1", 1, applied_source="A2")])
    (tmp_path / "decisions_recorded.jsonl").write_text(
        json.dumps({"run": "r1", "round": 1, "a2": {"status": "REPLAY", "output": {"action": "hold"}}}) + "\n")
    write_rounds(tmp_path / "rounds_rule.csv", [row("r2", r) for r in range(8)])
    write_rounds(tmp_path / "rounds.csv", [row("r3", 0)])
    runs = payload.load_runs(tmp_path)
    assert [r["source"] for r in runs] == ["live", "recorded", "rule"]
    rec = runs[1]
    assert rec["kind"] == "crew" and rec["badge"] == "REPLAY" and not rec["complete"]
    assert rec["rounds"][1]["agents"]["a2"]["status"] == "REPLAY"
    assert runs[2]["badge"] == "RULE ONLY" and runs[2]["complete"]


def test_torn_last_line_and_repeated_rounds(tmp_path):
    path = tmp_path / "rounds.csv"
    write_rounds(path, [row("r", 0), row("r", 1, rec=0.5), row("r", 1, rec=0.6)])
    with open(path, "a") as f:
        f.write("r,2,SHADOW,ho")  # a writer mid-append
    got = payload.read_rounds(path)
    assert got["round"].tolist() == [0, 1] and got["rec"].tolist() == [0.8, 0.6]


def test_last_decision_line_of_a_round_wins(tmp_path):
    path = tmp_path / "d.jsonl"
    lines = [{"run": "r", "round": 1, "a5": None}, {"run": "r", "round": 1, "a5": {"status": "LIVE"}}]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines) + "not json\n")
    assert payload.read_decisions(path)[("r", 1)]["a5"] == {"status": "LIVE"}


def test_clean_makes_json_safe_values():
    assert payload.clean(float("nan")) is None and payload.clean(float("inf")) is None
    json.dumps(payload.records(pd.DataFrame({"a": [1.0, float("nan")]})), allow_nan=False)


# --- weights -----------------------------------------------------------------------------------------------------

def test_starter_weights_are_shares_of_the_score():
    w = payload.starter_weights()
    assert len(w) == payload.N_WEIGHTS
    assert w[0]["feature"] == "avg_word_len" and w[0]["group"] == "writing"
    assert all(a["share"] >= b["share"] for a, b in zip(w, w[1:]))


def test_stack_weights_from_contributions(tmp_path):
    path = tmp_path / "contrib.csv"
    pd.DataFrame({"feature": ["logit_text_score", "logit_text_score", "sessions_per_day"],
                  "contrib": [2.0, -2.0, 1.0]}).to_csv(path, index=False)
    w = payload.stack_weights(path)
    assert [x["feature"] for x in w] == ["logit_text_score", "sessions_per_day"]
    assert w[0]["share"] == pytest.approx(2 / 3) and w[1]["group"] == "activity"


# --- server ------------------------------------------------------------------------------------------------------

@pytest.fixture
def base_url(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REVIEWS", tmp_path / "reviews.json")
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def call(url, data=None):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as res:
        return res.status, res.read()


def test_page_and_runs_are_served(base_url):
    status, body = call(base_url + "/")
    assert status == 200 and b"SoftSignal" in body
    status, body = call(base_url + "/api/runs")
    assert status == 200 and "runs" in json.loads(body)


def test_static_paths_cannot_leave_the_folder(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base_url + "/../payload.py")
    assert e.value.code == 404


def test_reviews_round_trip(base_url):
    _, body = call(base_url + "/api/review", {"account": "B1", "verdict": "agree"})
    assert json.loads(body) == {"B1": "agree"}
    _, body = call(base_url + "/api/review", {"account": "B1", "verdict": None})
    assert json.loads(body) == {}
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base_url + "/api/review", {"account": "B1", "verdict": "maybe"})
    assert e.value.code == 400


def test_bad_run_mode_is_refused(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base_url + "/api/run", {"mode": "both"})
    assert e.value.code == 400


# --- each run's own list -------------------------------------------------------------------------------------------

def test_starter_list_bands_at_the_cutoff_and_explains_with_fired_rules():
    from softsignal.data import load_data
    from softsignal.metrics import RANKED_COLS
    from softsignal.web.run_list import starter_list

    _, test = load_data()
    out = starter_list(test, 0.5)
    assert list(out.columns) == RANKED_COLS and len(out) == len(test)
    assert set(out["band"]) <= {"verify", "none"}  # no soft band: t_soft == t_verify
    assert ((out["score"] >= 0.5) == (out["band"] == "verify")).all()
    assert out["score"].is_monotonic_decreasing and out["words"].fillna("").eq("").all()
    top = out.iloc[0]
    assert top["c1"].endswith("+0.19") and top["f1"] == "avg_word_len"  # 0.35 x (1 - 0.45), the heaviest rule


def test_run_list_reads_the_file_for_the_source(tmp_path):
    from softsignal.web.run_list import write_list
    from softsignal.metrics import RANKED_COLS

    row = {c: "" for c in RANKED_COLS} | {"rank": 1, "blogger_id": "B1", "score": 0.9, "band": "verify",
                                          "f1": "logit_text_score", "v1": 1.0, "v2": None, "v3": None}
    write_list(pd.DataFrame([row], columns=RANKED_COLS), tmp_path / "run_lists" / "r9.csv")
    got = payload.run_list("live:r9", tmp_path)
    assert got["model"] == "stack" and got["rows"][0]["blogger_id"] == "B1"
    assert payload.run_list("rule:x", tmp_path)["rows"] is None  # no committed list for this source here
    assert payload.run_list("nonsense", tmp_path)["rows"] is None
