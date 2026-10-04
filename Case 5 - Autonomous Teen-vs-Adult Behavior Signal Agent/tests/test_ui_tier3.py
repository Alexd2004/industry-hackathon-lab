"""Step 18, sub-step 4: the Tier 3 panel in the Results tab (files in, no model, no loop run)."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from softsignal import tier3_report as t3
from softsignal import ui_results
from softsignal.metrics import EVAL_COLS
from softsignal.ui_results import load_tier3, tier3_verdict


def render():
    from softsignal import ui_results

    ui_results.render_results_tab()


def quality(rule_rec=0.88, live_rec=0.80) -> pd.DataFrame:
    base = dict.fromkeys(EVAL_COLS, 0.5) | {"eval_set": "test"}
    return pd.DataFrame([base | {"stage": "loop_rule_R7", "rec": rule_rec, "ft": 0.167},
                         base | {"stage": "loop_crew_live_R7", "rec": live_rec, "ft": 0.356}], columns=EVAL_COLS)


def latency(**round_time) -> pd.DataFrame:
    rt = {"row": "round_time", "n": 7, "p50_ms": 6727.1, "p95_ms": 15717.9, "tokens_in": None, "tokens_out": None,
          "n_model_fallback": None, "runs": "run-a", "note": ""} | round_time
    a3 = {"row": "A3", "n": 0, "p50_ms": None, "p95_ms": None, "tokens_in": None, "tokens_out": None,
          "n_model_fallback": 1, "runs": "run-a", "note": "no LIVE model call"}
    return pd.DataFrame([a3, rt], columns=t3.LATENCY_COLS)


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_results, "RESULTS", tmp_path)
    (tmp_path / "eval_placeholder.csv").write_text(ui_results.PLACEHOLDER_CSV.read_text())
    return tmp_path


def texts(at) -> str:
    return " ".join(e.value for e in [*at.info, *at.error, *at.caption, *at.subheader])


def test_load_tier3_returns_none_for_missing_files(tmp_path):
    assert load_tier3(tmp_path) == (None, None)


def test_load_tier3_reads_both_files_and_keeps_blanks(tmp_path):
    quality().to_csv(tmp_path / "eval_tier3.csv", index=False)
    latency().to_csv(tmp_path / "tier3_latency.csv", index=False)
    q, lat = load_tier3(tmp_path)
    assert len(q) == 2 and pd.isna(lat.loc[lat["row"] == "A3", "p95_ms"]).all()


@pytest.mark.parametrize("bad, match", [
    (quality().drop(columns=["auc"]), "missing columns"),
    (quality().assign(rec="x"), "numbers"),
])
def test_load_tier3_rejects_a_bad_quality_file(tmp_path, bad, match):
    bad.to_csv(tmp_path / "eval_tier3.csv", index=False)
    with pytest.raises(ValueError, match=match):
        load_tier3(tmp_path)


@pytest.mark.parametrize("bad, match", [
    (latency().drop(columns=["note"]), "missing columns"),
    (latency(p95_ms=None), "round_time row"),
    (latency().iloc[:1], "round_time row"),
    (latency(p50_ms="fast"), "numbers or blank"),
    (latency(p95_ms=float("inf")), "numbers or blank"),
])
def test_load_tier3_rejects_a_bad_latency_file(tmp_path, bad, match):
    bad.to_csv(tmp_path / "tier3_latency.csv", index=False)
    with pytest.raises(ValueError, match=match):
        load_tier3(tmp_path)


@pytest.mark.parametrize("live_rec, word", [(0.80, "lower than"), (0.95, "higher than"), (0.88, "equal to")])
def test_verdict_says_lower_higher_or_equal_and_never_wins(live_rec, word):
    v = tier3_verdict(quality(live_rec=live_rec))
    assert word in v and "once, not how often" in v and "win" not in v.lower()


def test_verdict_needs_both_rows():
    assert tier3_verdict(quality().iloc[1:]) is None and tier3_verdict(quality().iloc[:1]) is None


def test_tab_shows_both_hints_when_no_tier3_files(folder):
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert ui_results.NO_TIER3 in texts(at) and ui_results.NO_LATENCY in texts(at)


def test_tab_shows_rows_verdict_and_budget_line(folder):
    quality().to_csv(folder / "eval_tier3.csv", index=False)
    latency().to_csv(folder / "tier3_latency.csv", index=False)
    at = AppTest.from_function(render).run()
    assert not at.exception
    t = texts(at)
    assert "lower than" in t and "n = 7 rounds" in t and "over the 15 s budget" in t and "run-a" in t


def test_tab_shows_an_error_for_a_bad_tier3_file_and_keeps_the_rest(folder):
    (folder / "eval_tier3.csv").write_text("stage\nx\n")
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert "Cannot show the Tier 3 rows" in texts(at) and FOOTER_TEXT in " ".join(c.value for c in at.caption)


FOOTER_TEXT = ui_results.FOOTER


def test_committed_files_render():
    at = AppTest.from_function(render).run()
    assert not at.exception and "n = 7 rounds" in texts(at)
