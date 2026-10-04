"""crew.py (the loop with A1, appended round by round) and the Loop tab on its real output. No network."""
import threading

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from softsignal import crew, loop, ui_loop
from softsignal.agent_timer import AgentTimer
from softsignal.agents.base import FALLBACK, INSUFFICIENT, OFFLINE
from softsignal.data import load_data
from softsignal.metrics import ROUNDS_COLS
from softsignal.text_model import build_matrix
from softsignal.ui_loop import load_loop, valid_decision
from softsignal.ui_results import PLACEHOLDER_CSV

N_ROUNDS = 3  # R1-R3 hold (under 120 audit adults): no refit, so the run is quick


@pytest.fixture(scope="module")
def split_and_tm(tmp_path_factory):
    train, test = load_data(on_param_mismatch="error")
    return train, test, build_matrix(train[loop.ID_COL], cache_dir=tmp_path_factory.mktemp("cache"))


def env(split_and_tm, tmp_path, run):
    train, test, tm = split_and_tm
    return loop.make_env(train, test, timer=AgentTimer(tmp_path / "calls.jsonl", run=run), tm=tm)


@pytest.fixture(scope="module")
def crew_run(split_and_tm, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("results")
    (tmp / "eval_placeholder.csv").write_text(PLACEHOLDER_CSV.read_text())
    rounds, records = crew.run_crew(env(split_and_tm, tmp, "20261004T000000.000000Z-crew"), None, N_ROUNDS, write=True,
                                    rounds_path=tmp / "rounds.csv", decisions_path=tmp / "decisions.jsonl")
    return tmp, rounds, records


def test_crew_changes_nothing_the_rule_decides(split_and_tm, tmp_path, crew_run):
    _, rounds, records = crew_run
    plain_rounds, plain_records = loop.run_loop(env(split_and_tm, tmp_path, "20261004T000000.000000Z-crew"), N_ROUNDS)
    pd.testing.assert_frame_equal(rounds, plain_rounds)
    for got, want in zip(records, plain_records):
        agents = ("a1", "a5")  # the crew's own blocks; everything the rule decides must be identical
        assert {k: v for k, v in got.items() if k not in agents} == {k: v for k, v in want.items() if k not in agents}


def test_every_round_has_an_a1_block_and_a_valid_record(crew_run):
    _, rounds, records = crew_run
    assert list(rounds["round"]) == list(range(N_ROUNDS + 1)) and len(records) == N_ROUNDS + 1
    for r in records:
        a1 = r["a1"]
        assert valid_decision(r) and a1["status"] == FALLBACK and a1["input_hash"] and "ms" in a1
    assert [r["a1"]["fallback_reason"] for r in records] == [INSUFFICIENT, INSUFFICIENT] + [OFFLINE] * (N_ROUNDS - 1)
    assert all(r["a1"]["output"]["drift"] == "not_real" for r in records[2:])


def test_each_round_is_appended_as_it_lands(split_and_tm, tmp_path, monkeypatch):
    seen = []
    real = crew.write_run

    def spy(rounds, records, rounds_path, decisions_path):
        real(rounds, records, rounds_path, decisions_path)
        seen.append((len(rounds), len(pd.read_csv(rounds_path))))

    monkeypatch.setattr(crew, "write_run", spy)
    crew.run_crew(env(split_and_tm, tmp_path, "r-append"), None, 2, write=True, rounds_path=tmp_path / "rounds.csv",
                  decisions_path=tmp_path / "decisions.jsonl")
    assert seen == [(1, 1), (1, 2), (1, 3)]  # one row per call, the file growing round by round


def test_loop_tab_shows_the_real_run(crew_run, monkeypatch):
    tmp, _, _ = crew_run
    monkeypatch.setattr(ui_loop, "RESULTS", tmp)
    data = load_loop(tmp)
    assert not data.is_placeholder and len(data.rounds) == N_ROUNDS + 1 and data.n_skipped == 0

    def render():
        from softsignal import ui_loop
        ui_loop.render_loop_tab()

    at = AppTest.from_function(render).run()
    assert not at.exception and not at.error
    md = [m.value for m in at.markdown]
    assert ":orange-badge[FALLBACK]" in md and ":green-badge[LIVE]" not in md  # offline: never LIVE
    log = [t.value for t in at.text]
    assert log[0].startswith(f"R{N_ROUNDS}: applied hold from rule") and "A1 drift not_real" in log[0]
    assert any(c.value.startswith("Evidence: psi") for c in at.caption)


# --- background run (the Loop tab button) ----------------------------------------------------------

@pytest.fixture
def clean_crew(monkeypatch):
    monkeypatch.setattr(crew, "_current", crew.BackgroundRun())
    yield
    t = crew._current.thread
    if t is not None and t.ident is not None:  # started threads only (a fake may never start one)
        t.join(timeout=10)


def test_one_background_run_at_a_time(clean_crew, monkeypatch, tmp_path):
    go = threading.Event()
    monkeypatch.setattr(crew, "_background", lambda run, timer, results_dir, n: go.wait(5))
    first = crew.start_background(tmp_path)
    assert first and crew.status() == {"running": True, "run": first, "error": None}
    assert crew.start_background(tmp_path) is None  # refused while the first is running
    go.set()
    crew._current.thread.join(5)
    second = crew.start_background(tmp_path)
    assert second and second != first  # a new run id every run


def test_a_failed_background_run_reports_its_error(clean_crew, monkeypatch, tmp_path):
    def boom(**kw):
        raise RuntimeError("no data here")

    monkeypatch.setattr(crew, "load_data", boom)
    crew.start_background(tmp_path)
    crew._current.thread.join(5)
    assert crew.status() == {"running": False, "run": crew._current.run, "error": "RuntimeError: no data here"}


def test_run_button_starts_a_run_and_the_picker_follows_it(clean_crew, monkeypatch, crew_run):
    tmp, _, _ = crew_run
    monkeypatch.setattr(ui_loop, "RESULTS", tmp)
    started = []

    def fake_start(results_dir):
        started.append(results_dir)
        crew._current.run, crew._current.thread = "20261004T235959.000000Z-new", threading.Thread(target=lambda: None)
        return "20261004T235959.000000Z-new"

    monkeypatch.setattr(crew, "start_background", fake_start)
    monkeypatch.setattr(crew, "status", lambda: {"running": bool(started), "run": crew._current.run, "error": None})

    def render():
        from softsignal import ui_loop
        ui_loop.render_loop_tab()

    at = AppTest.from_function(render).run()
    assert at.button[0].label == "Run loop" and not at.button[0].disabled
    at.button[0].click().run()
    assert started == [tmp] and not at.exception
    assert at.selectbox[0].value == "20261004T235959.000000Z-new" and at.button[0].disabled
    assert "(running)" in at.selectbox[0].format_func(at.selectbox[0].value)


def test_a_failed_run_is_shown_in_the_tab(clean_crew, monkeypatch, crew_run):
    tmp, _, _ = crew_run
    monkeypatch.setattr(ui_loop, "RESULTS", tmp)
    monkeypatch.setattr(crew, "status", lambda: {"running": False, "run": "x", "error": "RuntimeError: boom"})

    def render():
        from softsignal import ui_loop
        ui_loop.render_loop_tab()

    other = AppTest.from_function(render).run()  # a session that did not start the run sees no banner
    assert not other.exception and not any("RuntimeError: boom" in e.value for e in other.error)
    at = AppTest.from_function(render)
    at.session_state["loop_started_run"] = "x"  # the session that pressed Run loop
    at.run()
    assert not at.exception and any("RuntimeError: boom" in e.value for e in at.error)


def test_a1_compares_each_batch_with_the_earlier_batches_only(split_and_tm, tmp_path, monkeypatch):
    payloads = []
    real = crew.run_a1
    monkeypatch.setattr(crew, "run_a1", lambda p, *a, **k: payloads.append(p) or real(p, *a, **k))
    crew.run_crew(env(split_and_tm, tmp_path, "r-ref"), None, N_ROUNDS, write=False)
    assert [(p["round"], p["n_reference"], p["n_batch"]) for p in payloads] == [
        (0, 0, 0), (1, 0, 300), (2, 300, 300), (3, 600, 300)]  # never the batch itself, never later batches
    assert [h["round"] for h in payloads[-1]["history"]] == [1, 2]  # earlier rounds' PSI only


def test_every_round_has_an_a5_block_checked_against_this_runs_rows(crew_run):
    tmp, rounds, records = crew_run
    for r in records:
        a5 = r["a5"]
        assert valid_decision(r) and a5["status"] == FALLBACK and a5["fallback_reason"] == OFFLINE
        (claim,) = a5["output"]  # no A2 wired yet: the round's headline only
        assert claim["claim"].startswith(f"Round {r['round']}:") and claim["verdict"] == "supported"
        assert claim["source"] == f"rounds.csv:R{r['round']}"
    on_disk = [__import__("json").loads(x) for x in (tmp / "decisions.jsonl").read_text().splitlines()]
    assert all(d["a5"]["status"] == FALLBACK for d in on_disk)  # A5 is in the record before it is written
