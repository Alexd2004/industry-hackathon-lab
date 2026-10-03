from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from softsignal import ui_results
from softsignal.metrics import EVAL_COLS
from softsignal.ui_results import BANNER, FOOTER, PLACEHOLDER_CSV, CI_NOTE, FT_CI, REC_CI, best_under_cap, headline_rows, load_ladder

APP = str(Path(__file__).resolve().parents[1] / "softsignal" / "app.py")


def render():  # AppTest.from_function runs this in the script thread
    from softsignal import ui_results

    ui_results.render_results_tab()


@pytest.fixture
def placeholder_only(tmp_path, monkeypatch):
    """Point the app at a folder holding only the placeholder, so tests do not depend on results/eval.csv."""
    (tmp_path / "eval_placeholder.csv").write_text(PLACEHOLDER_CSV.read_text())
    monkeypatch.setattr(ui_results, "RESULTS", tmp_path)


def test_placeholder_has_frozen_schema_and_projected_label():
    df = pd.read_csv(PLACEHOLDER_CSV)
    assert list(df.columns) == EVAL_COLS
    assert set(df["eval_set"]) == {"projected"}


def test_placeholder_used_when_no_real_file(tmp_path):
    (tmp_path / "eval_placeholder.csv").write_text(PLACEHOLDER_CSV.read_text())
    ladder, is_placeholder = load_ladder(tmp_path)
    assert is_placeholder and len(ladder) > 0


def test_real_file_wins_over_placeholder(tmp_path):
    (tmp_path / "eval_placeholder.csv").write_text(PLACEHOLDER_CSV.read_text())
    pd.DataFrame([dict.fromkeys(EVAL_COLS, 0.5) | {"stage": "x", "eval_set": "test"}]).to_csv(tmp_path / "eval.csv", index=False)
    ladder, is_placeholder = load_ladder(tmp_path)
    assert not is_placeholder and list(ladder["stage"]) == ["x"]


def test_missing_columns_raise(tmp_path):
    pd.DataFrame({"stage": ["x"]}).to_csv(tmp_path / "eval.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_ladder(tmp_path)


def test_best_under_cap_picks_highest_recall_within_cap():
    df = pd.DataFrame({"stage": ["a", "b", "c"], "rec": [0.9, 0.7, 0.8], "ft": [0.30, 0.10, 0.15]})
    assert best_under_cap(df, 0.15)["stage"] == "c"
    assert best_under_cap(df, 0.05) is None


def test_app_shows_both_tabs_banner_and_footer(placeholder_only):
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Results", "Loop"]
    assert any(BANNER in w.value for w in at.warning)
    assert any(FOOTER in c.value for c in at.caption)


def csv(*rows):
    return ",".join(EVAL_COLS) + chr(10) + "".join(r + chr(10) for r in rows)


@pytest.mark.parametrize("content, msg", [
    ("", "is empty"),
    (csv("x,test,0.5,0.7,high,0.3,0.6,0.8"), "column ft has non-numeric"),
    (csv("x,test,0.5,70%,0.1,0.3,0.6,0.8"), "column rec has non-numeric"),
])
def test_bad_real_file_raises_clear_error(tmp_path, content, msg):
    (tmp_path / "eval.csv").write_text(content)
    with pytest.raises(ValueError, match=msg):
        load_ladder(tmp_path)


def test_blank_numbers_are_allowed(tmp_path):
    (tmp_path / "eval.csv").write_text(csv("x,test,,0.7,0.1,,,"))
    ladder, _ = load_ladder(tmp_path)
    assert ladder["rec"].tolist() == [0.7] and ladder["prec"].isna().all()


def test_tab_shows_error_not_traceback_on_bad_file(tmp_path, monkeypatch):
    (tmp_path / "eval.csv").write_text("")
    monkeypatch.setattr(ui_results, "RESULTS", tmp_path)
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert any("Cannot show results" in e.value for e in at.error)


def test_headline_rows_placeholder_keeps_all_real_keeps_test_only():
    df = pd.DataFrame({"eval_set": ["test", "train", "projected", "test"], "rec": [1, 2, 3, 4]})
    assert len(headline_rows(df, True)) == 4
    assert headline_rows(df, False)["rec"].tolist() == [1, 4]


def test_real_file_tile_ignores_non_test_rows(tmp_path, monkeypatch):
    (tmp_path / "eval.csv").write_text(csv("good,test,,0.60,0.10,,,", "cv,cv,,0.95,0.05,,,", "proj,projected,,0.99,0.01,,,"))
    monkeypatch.setattr(ui_results, "RESULTS", tmp_path)
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert at.metric[0].value == "60.0%"
    assert any("marked projected" in w.value for w in at.warning)
    assert not any(BANNER in w.value for w in at.warning)


def test_placeholder_labels_tile_as_projected(placeholder_only):
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert "(projected)" in at.metric[0].label


def test_ci_note_matches_constants():
    assert f"{REC_CI * 100:.1f}" in CI_NOTE and f"{FT_CI * 100:.1f}" in CI_NOTE


def test_cap_band_shown_around_cap(placeholder_only):
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert any("Band: 11.7% to 18.3%" in c.value for c in at.caption)
    assert any("Point estimate" in c.value for c in at.caption)


def test_placeholder_regulator_rows_match_winning_plan_pdf():
    # Case5-Winning-Plan.pdf ladder table, "Regulator framing" rows (frozen 600 in the PDF)
    df = pd.read_csv(PLACEHOLDER_CSV).set_index("stage")
    for stage, want in {
        "10 regulator missed-teen <= 10%": (0.861, 0.913, 0.160, 0.087, 0.955),
        "10 regulator missed-teen <= 3%": (0.738, 0.974, 0.375, 0.026, 0.955),
    }.items():
        got = df.loc[stage, ["prec", "rec", "ft", "mt", "auc"]].tolist()
        assert got == pytest.approx(want)
