from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from softsignal import ui_results
from softsignal.metrics import EVAL_COLS
from softsignal.ui_results import BANNER, FOOTER, PLACEHOLDER_CSV, best_under_cap, load_ladder

APP = str(Path(__file__).resolve().parents[1] / "softsignal" / "app.py")


def render():  # AppTest.from_function runs this in the script thread
    from softsignal import ui_results

    ui_results.render_results_tab()


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


def test_app_shows_both_tabs_banner_and_footer():
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
