"""replay.py (Tier 3, step 17): recorded outputs served offline only for the same input, and the Loop tab's Replay
mode (Crew Plan test_replay: replaying a recorded run reproduces the same screen rows). No network."""
import json
import time

import pytest
from streamlit.testing.v1 import AppTest

from softsignal import crew, loop, replay, ui_loop
from softsignal.agent_timer import AgentTimer, load_records, summarize
from softsignal.agents.base import FALLBACK, LIVE, REPLAY, input_hash
from softsignal.data import load_data
from softsignal.text_model import build_matrix
from softsignal.ui_results import PLACEHOLDER_CSV

RESULTS = replay.RESULTS


def rec(run, rnd, **agents):
    base = {"run": run, "round": rnd, "rule_decision": {}, "diff": {}, "applied": {"decision": {}, "source": "rule"}}
    return base | {k: {"status": s, "output": o, "fallback_reason": None, "input_hash": h} for k, (s, o, h) in agents.items()}


# --- Replayer: recorded LIVE outputs, served only for the same input ---------------------------------------------

def test_only_live_blocks_of_the_latest_run_are_kept():
    records = [rec("r1", 1, a1=(LIVE, {"x": 1}, "h1")),
               rec("r2", 1, a1=(LIVE, {"x": 2}, "h2"), a5=(FALLBACK, [], "h5")),
               rec("r2", 2, a1=(None, None, None))]
    r = replay.Replayer.from_records(records)
    assert r.run == "r2" and len(r) == 1 and r.lookup(1, "a1", "h2") == {"x": 2}
    assert r.lookup(1, "a1", "other-hash") is None and r.lookup(1, "a5", "h5") is None  # a fallback is not replayed
    assert replay.Replayer.from_records(records, run="r1").lookup(1, "a1", "h1") == {"x": 1}


def test_serve_marks_replay_checks_the_output_and_times_a_retrieval(tmp_path):
    payload = {"agent": "A9", "round": 3, "x": 1}
    h = input_hash(payload)
    r = replay.Replayer.from_records([rec("r", 3, a1=(LIVE, {"ok": True}, h))])
    timer = AgentTimer(tmp_path / "calls.jsonl", run="t")
    ok = lambda out, p: (None, [])  # noqa: E731
    got = replay.serve(r, "a1", 3, payload, ok, timer)
    assert (got.status, got.output, got.input_hash) == (REPLAY, {"ok": True}, h)
    (call,) = load_records(timer.path)
    assert (call["agent"], call["kind"], call["status"]) == ("A1", "retrieval", REPLAY)
    assert summarize([call]) == {"by_agent": {}, "by_kind": {}, "by_agent_kind": {}, "by_status": {}}  # not latency
    assert replay.serve(r, "a1", 3, {**payload, "x": 2}, ok, timer) is None  # a different input: never replayed
    assert replay.serve(r, "a1", 3, payload, lambda out, p: ("invalid_output", ["no"]), timer) is None
    assert replay.serve(None, "a1", 3, payload, ok) is None and replay.serve(replay.Replayer(), "a1", 3, payload, ok) is None


def test_read_records_skips_a_torn_tail_and_bad_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(rec("r", 0)) + "\n{bad json\n" + json.dumps({"run": 1}) + "\n" + '{"run": "r", "rou')
    assert [r["round"] for r in replay.read_records(path)] == [0]


@pytest.fixture(scope="module")
def split_and_tm(tmp_path_factory):
    train, test = load_data(on_param_mismatch="error")
    return train, test, build_matrix(train[loop.ID_COL], cache_dir=tmp_path_factory.mktemp("cache"))


def test_an_offline_crew_run_replays_a_recorded_live_output(split_and_tm, tmp_path):
    train, test, tm = split_and_tm

    def env(run):
        return loop.make_env(train, test, timer=AgentTimer(tmp_path / f"{run}.jsonl", run=run), tm=tm)

    _, first = crew.run_crew(env("first"), None, 2)
    recorded = [dict(r) for r in first]
    recorded[2] = {**recorded[2], "a1": {**recorded[2]["a1"], "status": LIVE}}  # pretend R2's A1 answered live
    replayer = replay.Replayer.from_records(recorded)
    _, again = crew.run_crew(env("again"), None, 2, replayer=replayer)
    assert again[2]["a1"]["status"] == REPLAY and again[2]["a1"]["output"] == recorded[2]["a1"]["output"]
    assert again[2]["a1"]["input_hash"] == recorded[2]["a1"]["input_hash"]  # the same input, so it may be replayed
    assert [r["a1"]["status"] for r in again[:2]] == [FALLBACK, FALLBACK]  # nothing recorded live: the fallback


# --- the Loop tab's Replay mode --------------------------------------------------------------------------------

def test_as_replay_marks_every_agent_that_ran_and_keeps_what_it_was():
    r = rec("r", 1, a1=(FALLBACK, {"drift": "not_real"}, "h"), a5=(LIVE, [], "h5"), a2=(None, None, None))
    out = replay.as_replay(r)
    assert (out["a1"]["status"], out["a1"]["recorded_status"]) == (REPLAY, FALLBACK)
    assert (out["a5"]["status"], out["a5"]["recorded_status"]) == (REPLAY, LIVE)
    assert out["a2"]["status"] is None and r["a1"]["status"] == FALLBACK  # not run stays not run; input untouched
    assert replay.as_replay(out)["a1"]["recorded_status"] == FALLBACK  # idempotent


@pytest.mark.parametrize("elapsed, shown", [(0, 1), (2.9, 1), (3.0, 2), (9.5, 4), (100, 8)])
def test_one_round_every_three_seconds(elapsed, shown):
    assert replay.revealed_rounds(1000.0, 1000.0 + elapsed, total=8) == shown


def test_load_recorded_is_the_latest_recorded_run_in_round_order():
    rounds, records = replay.load_recorded(RESULTS)
    assert list(rounds["round"]) == list(range(8)) and rounds["run"].nunique() == 1
    assert [r["round"] for r in records] == list(range(8)) and all(r["run"] == rounds["run"].iloc[0] for r in records)
    assert {r["a1"]["status"] for r in records} == {REPLAY} and {r["a5"]["status"] for r in records} == {REPLAY}


def render():
    from softsignal import ui_loop
    ui_loop.render_loop_tab()


@pytest.fixture
def recorded_only(tmp_path, monkeypatch):
    """A results folder with the committed recorded run and the ladder placeholder only."""
    for name in (replay.ROUNDS_RECORDED, replay.DECISIONS_RECORDED):
        (tmp_path / name).write_text((RESULTS / name).read_text())
    (tmp_path / "eval_placeholder.csv").write_text(PLACEHOLDER_CSV.read_text())
    monkeypatch.setattr(ui_loop, "RESULTS", tmp_path)
    monkeypatch.setattr(ui_loop.crew, "status", lambda: {"running": False, "run": None, "error": None})
    return tmp_path


def test_replay_reproduces_the_recorded_runs_screen_rows(recorded_only):
    normal = AppTest.from_function(render).run()
    replayed = AppTest.from_function(render)
    replayed.session_state["loop_replay"] = True
    replayed.session_state["replay_t0"] = time.time() - 1000  # every round revealed
    replayed.run()
    assert not normal.exception and not replayed.exception
    assert [t.value for t in replayed.text] == [t.value for t in normal.text]  # the same decision log, line for line
    assert {m.label: m.value for m in replayed.metric} == {m.label: m.value for m in normal.metric}
    badges = {m.value for m in replayed.markdown if "badge" in m.value}
    assert ":gray-badge[REPLAY]" in badges and not any("LIVE" in b or "FALLBACK" in b for b in badges)
    assert any("Replayed from the recording, where it was FALLBACK" in c.value for c in replayed.caption)


def test_replay_reveals_one_round_at_a_time(recorded_only):
    at = AppTest.from_function(render)
    at.session_state["loop_replay"] = True
    at.session_state["replay_t0"] = time.time()
    at.run()
    assert [t.value[:3] for t in at.text] == ["R0:"] and any(c.value.startswith("Replay: round 0 of 7") for c in at.caption)
    assert at.button[0].disabled  # no live run while replaying


def test_replay_is_off_without_a_recorded_run(tmp_path, monkeypatch):
    (tmp_path / "eval_placeholder.csv").write_text(PLACEHOLDER_CSV.read_text())
    monkeypatch.setattr(ui_loop, "RESULTS", tmp_path)
    at = AppTest.from_function(render).run()
    assert not at.exception and at.toggle[0].disabled
