"""The model in training scored each round (loop.shadow_row / write_run's shadow file), report only. No network."""
from pathlib import Path

import pandas as pd

from softsignal.loop import SHADOW_COLS, shadow_path, write_run
from softsignal.metrics import ROUNDS_COLS
from softsignal.web import payload


def test_shadow_files_sit_next_to_their_rounds_file():
    assert shadow_path(Path("r/rounds.csv")) == Path("r/shadow.csv")
    assert shadow_path(Path("r/rounds_recorded.csv")) == Path("r/shadow_recorded.csv")
    assert shadow_path(Path("r/rounds_recorded.csv.new")) == Path("r/shadow_recorded.csv.new")


def row(run, rnd, **kw):
    return {c: None for c in ROUNDS_COLS} | {"run": run, "round": rnd, "mode": "SHADOW", "action": "hold",
                                            "applied_source": "rule", "rec": 0.8, "ft": 0.356} | kw


def shadow(run, rnd, rec):
    return {"run": run, "round": rnd, "candidate_round": 5, "t_verify": 0.41, "prec": 0.84, "rec": rec, "ft": 0.17,
            "mt": 1 - rec, "auc": 0.93}


def test_write_run_appends_shadow_rows_and_skips_none(tmp_path):
    rounds, decisions = tmp_path / "rounds.csv", tmp_path / "decisions.jsonl"
    write_run(pd.DataFrame([row("r", 4)], columns=ROUNDS_COLS), [{"run": "r", "round": 4}], rounds, decisions,
              shadow=[None])
    assert not (tmp_path / "shadow.csv").exists()  # no candidate before the first refit
    for rnd, rec in ((5, 0.86), (6, 0.88)):
        write_run(pd.DataFrame([row("r", rnd)], columns=ROUNDS_COLS), [{"run": "r", "round": rnd}], rounds, decisions,
                  shadow=[shadow("r", rnd, rec)])
    got = pd.read_csv(tmp_path / "shadow.csv")
    assert list(got.columns) == SHADOW_COLS and got["rec"].tolist() == [0.86, 0.88]


def test_decisions_never_carry_the_shadow_scores(tmp_path):
    rounds, decisions = tmp_path / "rounds.csv", tmp_path / "decisions.jsonl"
    write_run(pd.DataFrame([row("r", 5)], columns=ROUNDS_COLS), [{"run": "r", "round": 5}], rounds, decisions,
              shadow=[shadow("r", 5, 0.86)])
    assert "rec" not in decisions.read_text() and "0.86" not in decisions.read_text()


def test_the_console_shows_the_shadow_next_to_its_round(tmp_path):
    pd.DataFrame([row("r", 5), row("r", 6)], columns=ROUNDS_COLS).to_csv(tmp_path / "rounds.csv", index=False)
    pd.DataFrame([shadow("r", 6, 0.88)], columns=SHADOW_COLS).to_csv(tmp_path / "shadow.csv", index=False)
    (live,) = [r for r in payload.load_runs(tmp_path) if r["source"] == "live"]
    by_round = {v["round"]: v for v in live["rounds"]}
    assert by_round[5]["shadow"] is None and by_round[6]["shadow"]["rec"] == 0.88
    assert by_round[6]["rec"] == 0.8  # the live model's own numbers are untouched


def test_a2_is_told_to_promote_when_the_gate_allows_it():
    from softsignal.agents.a2_controller import SYSTEM

    assert "promote_allowed is true" in SYSTEM and "name that problem" in SYSTEM
