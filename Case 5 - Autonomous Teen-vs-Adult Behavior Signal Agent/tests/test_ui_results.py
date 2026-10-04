import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from softsignal import ui_results
from softsignal.metrics import EVAL_COLS
from softsignal.ui_results import (
    BANNER, CI_NOTE, FOOTER, FT_CI, PLACEHOLDER_CSV, REC_CI, best_under_cap, headline_rows, load_ladder,
)

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


def test_cap_band_shown_around_cap(placeholder_only):
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert any("Band: 11.7% to 18.3%" in c.value for c in at.caption)
    assert any("Point estimate" in c.value for c in at.caption)


def test_placeholder_regulator_rows_match_winning_plan_pdf():
    # Case5-Winning-Plan.pdf ladder table, "Regulator framing" rows (frozen 600 in the PDF)
    df = pd.read_csv(PLACEHOLDER_CSV).set_index("stage")
    for stage, want in {
        "10 regulator missed-teen <= 10% (frozen 600)": (0.861, 0.913, 0.160, 0.087, 0.955),
        "10 regulator missed-teen <= 3% (frozen 600)": (0.738, 0.974, 0.375, 0.026, 0.955),
    }.items():
        got = df.loc[stage, ["prec", "rec", "ft", "mt", "auc"]].tolist()
        assert got == pytest.approx(want)


def test_missing_placeholder_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="eval_placeholder.csv cannot be read"):
        load_ladder(tmp_path)


def test_unreadable_eval_csv_raises_clear_error(tmp_path):
    (tmp_path / "eval.csv").mkdir()  # a folder where the file should be
    with pytest.raises(ValueError, match="eval.csv cannot be read"):
        load_ladder(tmp_path)


def test_tab_shows_error_when_eval_csv_is_a_folder(tmp_path, monkeypatch):
    (tmp_path / "eval.csv").mkdir()
    monkeypatch.setattr(ui_results, "RESULTS", tmp_path)
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert any("Cannot show results" in e.value for e in at.error)


def test_placeholder_stage_names_state_cap_and_test_set():
    # PDF: the 6% in the plan is the measured false-teen at a 5% cap; regulator rows use a frozen 600 set
    stages = pd.read_csv(PLACEHOLDER_CSV)["stage"].tolist()
    assert "8 SoftSignal at 5% cap" in stages and not any("6% cap" in s for s in stages)
    assert sum("(frozen 600)" in s for s in stages) == 2 and all("regulator" in s for s in stages if "frozen 600" in s)


def test_ci_note_names_its_basis_and_scope():
    assert "450 adults" in CI_NOTE and "450 teens" in CI_NOTE and "frozen 600" in CI_NOTE


def chart_marks(at):
    spec = json.loads(at.get("vega_lite_chart")[0].proto.spec)
    return [layer["mark"]["type"] for layer in spec["layer"]]


def test_banner_text_is_the_required_literal(placeholder_only):
    at = AppTest.from_function(render).run()
    assert any("PLACEHOLDER, projected, not measured" in w.value for w in at.warning)


def test_cap_is_drawn_as_a_band_with_a_centre_line(placeholder_only):
    at = AppTest.from_function(render).run()
    assert chart_marks(at) == ["rect", "circle", "rule"]


def test_ci_widths_are_the_plans_numbers():
    assert (REC_CI, FT_CI) == (0.025, 0.033)
    assert "2.5 pts" in CI_NOTE and "3.3 pts" in CI_NOTE


def test_placeholder_tile_shows_the_stack_row(placeholder_only):
    at = AppTest.from_function(render).run()
    assert at.metric[0].value == "92.0%"  # row 7 at 15% cap, the best row within the cap
    assert at.metric[1].value == "15.0%"


def test_cap_slider_is_live_from_8_to_30_percent_at_the_policy_cap(placeholder_only):
    at = AppTest.from_function(render).run()
    s = at.slider[0]
    assert not s.disabled and (s.min, s.max, s.value) == (8, 30, 15)  # policy.yaml cap_false_teen 0.15
    at.slider[0].set_value(10).run()
    assert not at.exception and at.metric[2].value == "10.0%"  # the ladder tile follows the slider


def test_error_path_still_shows_footer(tmp_path, monkeypatch):
    (tmp_path / "eval.csv").write_text("")
    monkeypatch.setattr(ui_results, "RESULTS", tmp_path)
    at = AppTest.from_function(render).run()
    assert any(FOOTER in c.value for c in at.caption)


PLACEHOLDER_TABLE = [  # (stage, rec, ft), Combined Plan section 7 rounded, regulator rows exact from the PDF
    ("1 keyword baseline", 0.50, 0.32),
    ("2 starter blend w=0.45 cut=0.50", 0.82, 0.34),
    ("4 tune() under 15% cap", 0.72, 0.15),
    ("5 tabular LR 16 columns", 0.82, 0.15),
    ("6 TF-IDF text only", 0.80, 0.17),
    ("7 SoftSignal stack at 15% cap", 0.92, 0.15),
    ("8 SoftSignal at 10% cap", 0.88, 0.10),
    ("8 SoftSignal at 5% cap", 0.74, 0.06),
    ("9 loop R0", 0.82, 0.34),
    ("9 loop R7", 0.87, 0.14),
    ("10 regulator missed-teen <= 10% (frozen 600)", 0.913, 0.160),
    ("10 regulator missed-teen <= 3% (frozen 600)", 0.974, 0.375),
]


def test_placeholder_matches_the_expected_table():
    df = pd.read_csv(PLACEHOLDER_CSV)
    got = list(zip(df["stage"], df["rec"], df["ft"]))
    assert [g[0] for g in got] == [w[0] for w in PLACEHOLDER_TABLE]
    assert [g[1:] for g in got] == pytest.approx([w[1:] for w in PLACEHOLDER_TABLE])


# --- policy panel (ranked.csv, policy_grid.csv, contrib.csv) ------------------------------------------

REAL = Path(ui_results.RESULTS)
POLICY_FILES = (ui_results.RANKED_FILE, ui_results.GRID_FILE, ui_results.CONTRIB_FILE)
needs_files = pytest.mark.skipif(not all((REAL / f).exists() for f in POLICY_FILES),
                                 reason="run python -m softsignal.explain to write the policy files")


@pytest.fixture
def policy_dir(tmp_path, monkeypatch):
    """A results folder with the placeholder ladder and copies of the real policy files."""
    (tmp_path / "eval_placeholder.csv").write_text(PLACEHOLDER_CSV.read_text())
    for f in POLICY_FILES:
        (tmp_path / f).write_bytes((REAL / f).read_bytes())
    monkeypatch.setattr(ui_results, "RESULTS", tmp_path)
    return tmp_path


def metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


def test_policy_panel_without_files_shows_how_to_make_them(placeholder_only):
    at = AppTest.from_function(render).run()
    assert not at.exception
    assert any(ui_results.NO_FILES in i.value for i in at.info)
    assert any(FOOTER in c.value for c in at.caption)


@needs_files
def test_policy_files_carry_no_label():
    for f in POLICY_FILES:
        cols = set(pd.read_csv(REAL / f, nrows=1).columns)
        assert not cols & {"label_teen", "is_teen", "age", "gender", "job"}, f


@needs_files
def test_every_grid_cap_rebands_to_the_grid_counts():
    files = ui_results.load_policy_files(REAL)
    assert len(files.grid) == 23
    for _, row in files.grid.iterrows():
        assert ui_results.grid_mismatches(ui_results.reband(files.ranked, row), row) == [], row["cap"]


@needs_files
def test_tiles_come_from_the_grid_row_and_follow_the_slider(policy_dir):
    grid = ui_results.load_policy_files(policy_dir).grid
    at = AppTest.from_function(render).run()
    assert not at.exception and not [w for w in at.warning if "disagree" in w.value]
    for cap in (15, 10):
        if cap != 15:
            at.slider[0].set_value(cap).run()
        row = ui_results.grid_row(grid, cap / 100)
        m = metrics(at)
        assert m["Flagged for verification"] == f"{int(row['n_flagged'])} ({row['flagged_share']:.0%})"
        assert m["Sent to verification now"] == str(int(row["n_verify"]))
        assert m["Teen-safe defaults (soft band)"] == str(int(row["n_soft"]))
        assert m["False-teen, flagged"] == f"{row['ft_flagged']:.1%}"


@needs_files
def test_around_the_cutoff_view_shows_the_band_edge(policy_dir):
    at = AppTest.from_function(render).run()
    at.slider[0].set_value(10).run()
    at.radio[0].set_value(ui_results.VIEWS[1]).run()
    shown = at.dataframe[1].value
    assert len(shown) == ui_results.N_SHOWN and {"verify", "soft"} <= set(shown["band"])
    assert set(shown.loc[shown["band"] == "verify", "review"]) <= {ui_results.SENT, ui_results.QUEUED}


@needs_files
def test_detail_card_is_exact_and_label_free(policy_dir):
    at = AppTest.from_function(render).run()
    at.selectbox[0].set_value(3).run()
    assert not at.exception
    assert any("(exact" in c.value for c in at.caption)
    for df in at.dataframe:
        assert "label_teen" not in df.value.columns


@needs_files
def test_mismatched_files_warn(policy_dir):
    grid = pd.read_csv(policy_dir / ui_results.GRID_FILE)
    grid.loc[grid["cap"] == 0.15, "n_flagged"] += 1
    grid.to_csv(policy_dir / ui_results.GRID_FILE, index=False)
    at = AppTest.from_function(render).run()
    assert not at.exception and any("disagree" in w.value for w in at.warning)


@needs_files
def test_missing_contrib_turns_the_card_off(policy_dir):
    (policy_dir / ui_results.CONTRIB_FILE).unlink()
    at = AppTest.from_function(render).run()
    assert not at.exception and any(ui_results.CONTRIB_FILE in i.value for i in at.info)


@needs_files
def test_grid_without_the_cap_warns(policy_dir):
    grid = pd.read_csv(policy_dir / ui_results.GRID_FILE)
    grid[grid["cap"] != 0.15].to_csv(policy_dir / ui_results.GRID_FILE, index=False)
    at = AppTest.from_function(render).run()
    assert not at.exception and any("no row for a 15% cap" in w.value for w in at.warning)


@needs_files
def test_broken_ranked_file_shows_an_error_not_a_traceback(policy_dir):
    pd.read_csv(policy_dir / ui_results.RANKED_FILE).drop(columns="score").to_csv(
        policy_dir / ui_results.RANKED_FILE, index=False)
    at = AppTest.from_function(render).run()
    assert not at.exception and any("Cannot show the ranked list" in e.value for e in at.error)
    assert any(FOOTER in c.value for c in at.caption)


EDGE = 0.965007418532936  # a real score whose 1-ulp-higher cutoff the default CSV parser reads back as equal


def tiny_files(tmp_path, t_budget):
    """3 accounts and a one-row grid; scores and cutoffs written exactly as explain.py writes them."""
    from softsignal.explain import _write_csv
    from softsignal.metrics import POLICY_GRID_COLS, RANKED_COLS

    scores = [0.99, EDGE, 0.1]
    ranked = pd.DataFrame({c: [""] * 3 for c in RANKED_COLS})
    ranked["rank"], ranked["blogger_id"], ranked["score"] = [1, 2, 3], ["A", "B", "C"], scores
    ranked["band"], ranked["action"], ranked["reason"] = "none", "No action", "x"
    ranked[["v1", "v2", "v3"]] = 0.0
    row = dict.fromkeys(POLICY_GRID_COLS, 0.0) | {
        "cap": 0.15, "t_verify": 0.5, "t_soft": 0.5, "t_budget": t_budget, "flags": "", "n": 3, "n_flagged": 2,
        "n_verify": 1, "n_soft": 0, "n_none": 1, "budget_binding": True}
    _write_csv(ranked, tmp_path / "ranked.csv")
    _write_csv(pd.DataFrame([row], columns=POLICY_GRID_COLS), tmp_path / "policy_grid.csv")
    return ui_results.load_policy_files(tmp_path)


def test_cutoff_one_ulp_above_a_score_survives_the_csv(tmp_path):
    # regression: the default CSV float parser read t_budget back equal to the score just below it
    files = tiny_files(tmp_path, float(np.nextafter(EDGE, 1.0)))
    banded = ui_results.reband(files.ranked, files.grid.iloc[0])
    assert banded["queue"].tolist() == [ui_results.SENT, ui_results.QUEUED, ""]
    assert ui_results.grid_mismatches(banded, files.grid.iloc[0]) == []


def test_words_show_only_for_flagged_accounts():
    rows = pd.DataFrame({"rank": [1, 2], "blogger_id": ["A", "B"], "score": [0.9, 0.1], "band": ["verify", "none"],
                         "action": ["Request verification", "No action"], "queue": [ui_results.SENT, ""],
                         "c1": ["x +1.00", "y -1.00"], "c2": ["", ""], "c3": ["", ""], "words": ["lol", "lol"]})
    assert ui_results.list_table(rows)["teen-leaning words"].tolist() == ["lol", ""]


def test_malformed_policy_yaml_does_not_break_the_tab(placeholder_only, tmp_path, monkeypatch):
    bad = tmp_path / "bad_policy.yaml"
    bad.write_text("cap_false_teen: [0.15\n")
    monkeypatch.setattr(ui_results, "POLICY_PATH", bad)
    at = AppTest.from_function(render).run()
    assert not at.exception and at.slider[0].value == 15  # the default cap
    assert any("policy.yaml cannot be used" in w.value for w in at.warning)
    assert any(FOOTER in c.value for c in at.caption)


def test_slider_range_comes_from_the_grid(tmp_path):
    from softsignal.explain import SLIDER_CAPS

    assert ui_results.slider_bounds(None) == (round(min(SLIDER_CAPS) * 100), round(max(SLIDER_CAPS) * 100)) == (8, 30)
    files = tiny_files(tmp_path, 0.95)
    files.grid = pd.concat([files.grid.assign(cap=c) for c in (0.10, 0.15, 0.20)], ignore_index=True)
    assert ui_results.slider_bounds(files) == (10, 20)
