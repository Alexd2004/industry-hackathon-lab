import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from softsignal.metrics import EVAL_COLS
from softsignal.ui_results import BANNER, FOOTER, PLACEHOLDER_CSV, best_under_cap, load_ladder

APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "softsignal" / "app.py")


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
