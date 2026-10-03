import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from softsignal import ui_loop
from softsignal.metrics import ROUNDS_COLS
from softsignal.ui_loop import (
    BANNER, DECISIONS_PLACEHOLDER, ROUNDS_PLACEHOLDER, diff_table, load_loop, log_line, read_complete_lines,
    valid_decision,
)
from softsignal.ui_results import FOOTER, PLACEHOLDER_CSV

RESULTS = Path(__file__).resolve().parents[1] / "results"
APP = str(Path(__file__).resolve().parents[1] / "softsignal" / "app.py")


def render():  # AppTest.from_function runs this in the script thread
    from softsignal import ui_loop

    ui_loop.render_loop_tab()


def copy_placeholders(folder: Path, loop=True, ladder=True) -> None:
    if loop:
        for name in (ROUNDS_PLACEHOLDER, DECISIONS_PLACEHOLDER):
            (folder / name).write_text((RESULTS / name).read_text())
    if ladder:
        (folder / "eval_placeholder.csv").write_text(PLACEHOLDER_CSV.read_text())


@pytest.fixture
def placeholder_only(tmp_path, monkeypatch):
    """Point the tab at a folder holding only the placeholders, so tests do not depend on real results files."""
    copy_placeholders(tmp_path)
    monkeypatch.setattr(ui_loop, "RESULTS", tmp_path)
    return tmp_path


@pytest.fixture
def real_dir(tmp_path, monkeypatch):
    """A folder with the ladder placeholder only; each test writes its own real rounds.csv / decisions.jsonl."""
    copy_placeholders(tmp_path, loop=False)
    monkeypatch.setattr(ui_loop, "RESULTS", tmp_path)
    return tmp_path


def row(run="20261003T150000Z", rnd=0, **kw) -> dict:
    base = dict(run=run, round=rnd, mode="SHADOW", action="hold", applied_source="rule", cap=0.15, t_soft="",
                t_verify=0.5, n_verify=0, n_labels=0, n_audit_adults=0, audit_ft="", psi="", prec=0.7, rec=0.8,
                ft=0.3, mt=0.2, auc=0.85, refit_s="")
    return base | kw


def write_rounds(folder: Path, rows, tail="\n") -> None:
    body = "\n".join(",".join(str(r[c]) for c in ROUNDS_COLS) for r in rows)
    (folder / "rounds.csv").write_text(",".join(ROUNDS_COLS) + "\n" + body + tail)


def agent(status="LIVE", output=None, reason=None) -> dict:
    return {"status": status, "output": output, "fallback_reason": reason}


def decision(run="20261003T150000Z", rnd=0, **kw) -> dict:
    a2 = {"blend_w": None, "cutoff": 0.5, "cap": 0.15, "action": "hold", "reason": "r", "cites": []}
    base = {"run": run, "round": rnd, "a1": agent(output={"drift": "not_real", "evidence": [], "reason": "ok"}),
            "a2": agent(output=a2), "a3": agent(output="insufficient_data"), "a4": agent(output={"batch_reason": "b"}),
            "a5": agent(output=[]), "rule_decision": {"cutoff": 0.5, "cap": 0.15, "action": "hold"},
            "applied": {"decision": {"cutoff": 0.5, "cap": 0.15, "action": "hold"}, "source": "A2"}}
    return base | kw


def write_decisions(folder: Path, recs, tail="") -> None:
    (folder / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs) + tail)


def markdowns(at) -> list[str]:
    return [m.value for m in at.markdown]


# placeholders

def test_rounds_placeholder_has_frozen_schema_and_projected_run():
    df = pd.read_csv(RESULTS / ROUNDS_PLACEHOLDER)
    assert list(df.columns) == ROUNDS_COLS
    assert set(df["run"]) == {"placeholder"} and df["round"].tolist() == list(range(8))


def test_rounds_placeholder_matches_eval_placeholder_loop_rows():
    rounds = pd.read_csv(RESULTS / ROUNDS_PLACEHOLDER).set_index("round")
    ladder = pd.read_csv(PLACEHOLDER_CSV).set_index("stage")
    for r in (0, 7):
        assert rounds.loc[r, ["rec", "ft"]].tolist() == pytest.approx(ladder.loc[f"9 loop R{r}", ["rec", "ft"]].tolist())
    assert rounds.loc[1, ["rec", "ft"]].tolist() == rounds.loc[0, ["rec", "ft"]].tolist()  # R1 holds


def test_rounds_placeholder_follows_the_spec_path():
    df = pd.read_csv(RESULTS / ROUNDS_PLACEHOLDER).set_index("round")
    assert df["n_labels"].tolist() == [120 * r for r in range(8)]
    assert df["n_audit_adults"].tolist() == [30 * r for r in range(8)]
    assert df.loc[[1, 2, 3], "action"].eq("hold").all() and df.loc[4, "action"] == "re-tune"
    assert df.loc[6, "action"] == "promote" and df.loc[5, "mode"] == "SHADOW" and df.loc[6, "mode"] == "ACTIVE"
    assert df["refit_s"].isna().all()  # no latency until measured
    assert df["rec"].is_monotonic_increasing and df["ft"].is_monotonic_decreasing


def test_every_decisions_placeholder_line_is_valid():
    lines = (RESULTS / DECISIONS_PLACEHOLDER).read_text().splitlines()
    recs = [json.loads(ln) for ln in lines]
    assert len(recs) == 8 and all(valid_decision(r) for r in recs)
    statuses = [r[a]["status"] for r in recs for a in ui_loop.AGENTS if r[a]]
    assert statuses.count("FALLBACK") == 1
    assert sum(bool(r["diff"]) for r in recs) == 1
    assert recs[0]["a1"]["output"]["drift"] == "insufficient_data" and recs[0]["a3"]["output"] == "insufficient_data"


# loader

def test_placeholders_used_when_no_real_file(tmp_path):
    copy_placeholders(tmp_path)
    data = load_loop(tmp_path)
    assert data.is_placeholder and len(data.rounds) == 8 and len(data.decisions) == 8 and data.n_skipped == 0


def test_real_file_wins_and_a_missing_one_counts_as_empty(tmp_path):
    copy_placeholders(tmp_path)
    write_decisions(tmp_path, [decision()])
    data = load_loop(tmp_path)
    assert not data.is_placeholder and data.rounds.empty and len(data.decisions) == 1


def test_no_files_at_all_is_an_empty_state(tmp_path):
    data = load_loop(tmp_path)
    assert not data.is_placeholder and data.rounds.empty and data.decisions == []


@pytest.mark.parametrize("content, msg", [
    ("", "is empty"),
    ("run,round\nx,0\n", "missing columns"),
    (",".join(ROUNDS_COLS) + "\n" + ",".join(str(row(rec="high")[c]) for c in ROUNDS_COLS) + "\n", "rec has non-numeric"),
    (",".join(ROUNDS_COLS) + "\n" + ",".join(str(row(rec="")[c]) for c in ROUNDS_COLS) + "\n", "rec has blank"),
    (",".join(ROUNDS_COLS) + "\n" + ",".join(str(row(rnd=-1)[c]) for c in ROUNDS_COLS) + "\n", "round must be whole"),
    (",".join(ROUNDS_COLS) + "\n" + ",".join(str(row(rnd=1.5)[c]) for c in ROUNDS_COLS) + "\n", "round must be whole"),
    (",".join(ROUNDS_COLS) + "\n" + ",".join(str(row(mode="LIVE")[c]) for c in ROUNDS_COLS) + "\n", "mode must be"),
    (",".join(ROUNDS_COLS) + "\n" + ",".join(str(row(applied_source="x")[c]) for c in ROUNDS_COLS) + "\n", "applied_source"),
    (",".join(ROUNDS_COLS) + "\n" + ",".join(str(row(ft=34)[c]) for c in ROUNDS_COLS) + "\n", "ft has rates outside"),
])
def test_bad_rounds_file_raises_clear_error(tmp_path, content, msg):
    (tmp_path / "rounds.csv").write_text(content)
    with pytest.raises(ValueError, match=msg):
        load_loop(tmp_path)


def test_blanks_allowed_only_in_optional_columns(tmp_path):
    write_rounds(tmp_path, [row()])
    df = load_loop(tmp_path).rounds
    assert df[["t_soft", "audit_ft", "psi", "refit_s"]].isna().all().all() and df["round"].tolist() == [0]


@pytest.mark.parametrize("name", ["rounds.csv", "decisions.jsonl"])
def test_folder_instead_of_file_raises_clear_error(tmp_path, name):
    (tmp_path / name).mkdir()
    with pytest.raises(ValueError, match=f"{name} cannot be read"):
        load_loop(tmp_path)


def test_half_written_last_line_is_skipped_in_both_files(tmp_path):
    write_rounds(tmp_path, [row(rnd=0), row(rnd=1)], tail="\n" + "20261003T150000Z,2,SHA")
    write_decisions(tmp_path, [decision(rnd=0)], tail='{"run": "20261003T150000Z", "rou')
    data = load_loop(tmp_path)
    assert data.rounds["round"].tolist() == [0, 1]
    assert [d["round"] for d in data.decisions] == [0] and data.n_skipped == 0


def test_read_complete_lines_drops_only_an_unterminated_tail(tmp_path):
    p = tmp_path / "f"
    p.write_text("a\nb\nc")
    assert read_complete_lines(p) == ["a", "b"]
    p.write_text("a\nb\n")
    assert read_complete_lines(p) == ["a", "b"]


def test_bad_json_and_invalid_records_are_skipped_and_counted(tmp_path):
    bad = [decision(rnd=1) | {"rec": 0.9},  # test metrics never belong in decisions.jsonl
           {k: v for k, v in decision(rnd=2).items() if k != "a4"},
           decision(rnd=3, a2={"status": "MAYBE", "output": None, "fallback_reason": None}),
           decision(rnd=4) | {"applied": {"decision": {}, "source": "A9"}}]
    write_decisions(tmp_path, [decision(rnd=0)] + bad, tail="not json\n")
    data = load_loop(tmp_path)
    assert [d["round"] for d in data.decisions] == [0] and data.n_skipped == 5


def test_duplicate_run_round_keeps_last_in_both_files(tmp_path):
    write_rounds(tmp_path, [row(rnd=0, rec=0.5), row(rnd=0, rec=0.6)])
    write_decisions(tmp_path, [decision(rnd=0, a4=None), decision(rnd=0)])  # A4 landed later
    data = load_loop(tmp_path)
    assert data.rounds["rec"].tolist() == [0.6] and any("repeated" in w for w in data.warnings)
    assert len(data.decisions) == 1 and data.decisions[0]["a4"] is not None and data.decisions[0]["diff"] == {}


def test_runs_newest_first_and_filtered(tmp_path):
    old, new = "20261003T140000Z", "20261003T150000Z"
    write_rounds(tmp_path, [row(run=old, rnd=0), row(run=new, rnd=0), row(run=new, rnd=1)])
    write_decisions(tmp_path, [decision(run=old, rnd=0), decision(run=new, rnd=1)])
    data = load_loop(tmp_path)
    assert ui_loop.runs(data) == [new, old]
    rounds, decs = ui_loop.for_run(data, old)
    assert rounds["round"].tolist() == [0] and [d["run"] for d in decs] == [old]


# pure helpers

def test_diff_table_marks_a2_vs_rule_changes():
    d = decision(diff={"cutoff": [0.55, 0.58]})
    d["a2"]["output"]["cutoff"] = 0.58
    d["applied"]["decision"]["cutoff"] = 0.58
    t = diff_table(d).set_index("field")
    assert list(t.index) == ["blend_w", "cutoff", "cap", "action"]
    assert list(t.columns) == ["rule", "A2", "applied", "changed"]
    assert t.loc["cutoff"].tolist() == ["0.50", "0.58", "0.58", "YES"] and t.loc["cap", "changed"] == ""


def test_log_line_mentions_action_source_diff_fallback_and_metrics():
    d = decision(rnd=4, diff={"cutoff": [0.55, 0.58]}, a3=agent("FALLBACK", None, "timeout"))
    d["applied"]["decision"]["action"] = "re-tune"
    line = log_line(d, pd.Series(row(rnd=4, rec=0.85, ft=0.2)))
    assert line.startswith("R4: applied re-tune from A2")
    assert "cutoff 0.55 -> 0.58" in line and "A3 (timeout)" in line and "recall 85%" in line
    assert "recall" not in log_line(d, None)


# app

def test_app_renders_both_tabs_without_exception(placeholder_only, monkeypatch):
    from softsignal import ui_results

    monkeypatch.setattr(ui_results, "RESULTS", placeholder_only)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Results", "Loop"]
    assert any(BANNER in w.value for w in at.warning)


def test_placeholder_tab_shows_banner_footer_and_projected_tiles(placeholder_only):
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert any("PLACEHOLDER, projected, not measured" in w.value for w in at.warning)
    assert any(FOOTER in c.value for c in at.caption)
    assert ":gray-badge[PLACEHOLDER]" in markdowns(at) and ":blue-badge[ACTIVE]" in markdowns(at)
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Round"] == "7" and tiles["Labels learned (projected)"] == "840" and tiles["Fallbacks this run"] == "1"


def chart_marks(at):
    spec = json.loads(at.get("vega_lite_chart")[0].proto.spec)
    return [layer["mark"]["type"] for layer in spec["layer"]]


def test_chart_has_band_baselines_and_lines(placeholder_only):
    at = AppTest.from_function(render).run()
    assert chart_marks(at) == ["area", "rule", "line"]


def test_chart_skips_baselines_when_ladder_is_unreadable(placeholder_only):
    (placeholder_only / "eval_placeholder.csv").unlink()
    at = AppTest.from_function(render).run()
    assert not at.exception and chart_marks(at) == ["area", "line"]


def test_placeholder_shows_fallback_badge_and_a2_diff_row(placeholder_only):
    at = AppTest.from_function(render).run()
    md = markdowns(at)
    assert ":green-badge[LIVE]" in md and ":orange-badge[FALLBACK]" in md
    assert any("Fallback reason: timeout" in c.value for c in at.caption)
    diffs = [df.value.set_index("field") for df in at.dataframe]
    assert any(t.loc["cutoff", "changed"] == "YES" and t.loc["cutoff", "A2"] == "0.58" for t in diffs)


def test_replay_badges_and_header(real_dir):
    replay = agent("REPLAY", {"drift": "not_real", "evidence": [], "reason": "ok"})
    write_decisions(real_dir, [decision(rnd=1, a1=replay)])
    at = AppTest.from_function(render).run()
    assert not at.exception
    md = markdowns(at)
    assert md.count(":gray-badge[REPLAY]") == 2  # the header and the A1 card
    assert not any(BANNER in w.value for w in at.warning)


def test_decision_log_is_newest_first(placeholder_only):
    at = AppTest.from_function(render).run()
    log = [t.value for t in at.text]
    assert [ln.split(":")[0] for ln in log] == [f"R{r}" for r in range(7, -1, -1)]


def test_empty_state_without_any_file(tmp_path, monkeypatch):
    (tmp_path / "eval_placeholder.csv").write_text(PLACEHOLDER_CSV.read_text())
    monkeypatch.setattr(ui_loop, "RESULTS", tmp_path)
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert any("No loop run yet" in i.value for i in at.info)
    assert chart_marks(at) == ["area", "rule"]
    assert sum("Waiting for a run" in c.value for c in at.caption) == 5
    assert any(FOOTER in c.value for c in at.caption)


def test_decisions_without_rounds_show_cards_only(real_dir):
    write_decisions(real_dir, [decision(rnd=1, a4=None)])
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert any(c.value == "working..." for c in at.caption)
    assert chart_marks(at) == ["area", "rule"] and at.metric[0].value == "-"


def test_error_path_shows_error_and_footer(real_dir):
    (real_dir / "rounds.csv").write_text("")
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert any("Cannot show loop" in e.value for e in at.error)
    assert any(FOOTER in c.value for c in at.caption)


def test_skipped_lines_are_reported(real_dir):
    write_decisions(real_dir, [decision()], tail="{oops\n")
    at = AppTest.from_function(render).run()
    assert any("1 lines skipped" in c.value for c in at.caption)


def test_live_crew_and_replay_are_disabled_stubs(placeholder_only):
    at = AppTest.from_function(render).run()
    assert at.button[0].label == "Run live crew" and at.button[0].disabled
    assert at.toggle[0].label == "Replay" and at.toggle[0].disabled
