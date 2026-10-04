"""Step 18, sub-step 1: the Tier 3 report reader (fake rounds frames, no model, no loop run)."""
import pandas as pd
import pytest

from softsignal import tier3_report as t3
from softsignal.metrics import ROUNDS_COLS

R0 = {"prec": 0.69, "rec": 0.80, "ft": 0.36, "mt": 0.20, "auc": 0.82}


def make_run(run: str, sources: list[str], **r0) -> pd.DataFrame:
    rows = []
    for rnd, src in enumerate(sources):
        row = dict.fromkeys(ROUNDS_COLS) | {"run": run, "round": rnd, "mode": "SHADOW", "action": "hold",
                                            "applied_source": src, "diff_count": 0, "cap": 0.15}
        rows.append(row | (R0 | r0))
    return pd.DataFrame(rows, columns=ROUNDS_COLS)


RULE = ["starter", "rule", "rule"]
CREW = ["starter", "A2", "rule"]


def test_load_rounds_reads_the_committed_recording():
    df = t3.load_rounds(t3.ROUNDS_CSV.with_name("rounds_recorded.csv"))
    assert list(df.columns) == ROUNDS_COLS and len(t3.run_ids(df)) == 1


def test_load_rounds_rejects_a_different_header(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("run,round\nx,0\n")
    with pytest.raises(ValueError, match="frozen ROUNDS_COLS"):
        t3.load_rounds(p)


def test_get_run_sorts_by_round_and_returns_only_that_run():
    both = pd.concat([make_run("a", CREW), make_run("b", RULE)]).sample(frac=1, random_state=0)
    run = t3.get_run(both, "a")
    assert list(run["round"]) == [0, 1, 2] and set(run["run"]) == {"a"}


def test_get_run_rejects_an_unknown_run():
    with pytest.raises(ValueError, match="not in the rounds file"):
        t3.get_run(make_run("a", CREW), "zzz")


@pytest.mark.parametrize("drop", [[1], [0], [0, 2]])
def test_get_run_rejects_a_gap_or_a_missing_r0(drop):
    df = make_run("a", CREW)
    with pytest.raises(ValueError, match="incomplete"):
        t3.get_run(df[~df["round"].isin(drop)], "a")


def test_get_run_rejects_a_repeated_round():
    df = make_run("a", CREW)
    with pytest.raises(ValueError, match="incomplete"):
        t3.get_run(pd.concat([df, df.iloc[[1]]]), "a")


def test_final_row_is_the_last_round_and_sources_follow_round_order():
    run = make_run("a", CREW)
    assert t3.final_row(run)["round"] == 2 and t3.sources(run) == CREW


def test_matching_runs_are_comparable():
    rule, crew = make_run("r", RULE), make_run("c", CREW)
    assert t3.comparison_errors(rule, crew) == []
    t3.check_comparable(rule, crew)


def test_runs_with_different_rounds_are_not_comparable():
    errors = t3.comparison_errors(make_run("r", RULE), make_run("c", CREW + ["rule"]))
    assert len(errors) == 1 and "rounds differ" in errors[0]


def test_a_rule_run_with_an_a2_decision_is_not_a_rule_run():
    errors = t3.comparison_errors(make_run("r", CREW), make_run("c", CREW))
    assert len(errors) == 1 and "not a rule-only run" in errors[0]


def test_different_r0_means_a_different_seed_or_split():
    errors = t3.comparison_errors(make_run("r", RULE), make_run("c", CREW, rec=0.5))
    assert len(errors) == 1 and "R0 differs" in errors[0]


def test_float_noise_in_r0_is_not_a_difference():
    assert t3.comparison_errors(make_run("r", RULE), make_run("c", CREW, rec=0.80 + 1e-12)) == []


def test_check_comparable_lists_every_reason():
    with pytest.raises(ValueError) as e:
        t3.check_comparable(make_run("r", CREW), make_run("c", CREW + ["rule"]))
    assert "rounds differ" in str(e.value) and "not a rule-only run" in str(e.value)


def test_calls_for_run_keeps_only_that_run():
    records = [{"run": "a", "n": 1}, {"run": "b", "n": 2}, {"n": 3}]
    assert t3.calls_for_run(records, "a") == [{"run": "a", "n": 1}]
