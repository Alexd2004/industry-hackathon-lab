"""Step 18, sub-steps 1 to 3: the Tier 3 report reader, quality rows and latency table (fake rounds frames, no model, no loop run)."""
import pandas as pd
import pytest

from softsignal import tier3_report as t3
from softsignal.agents.base import FALLBACK, LIVE, REPLAY
from softsignal.metrics import EVAL_COLS, ROUNDS_COLS, f1

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


def test_quality_row_copies_the_final_round_and_derives_f1():
    run = make_run("a", CREW)
    run.loc[2, ["prec", "rec", "ft", "mt", "auc"]] = [0.8, 0.9, 0.2, 0.1, 0.95]
    row = t3.quality_row("s", run)
    assert row == {"stage": "s", "eval_set": "test", "prec": 0.8, "rec": 0.9, "ft": 0.2, "mt": 0.1,
                   "f1": f1(0.8, 0.9), "auc": 0.95}


def test_quality_rows_names_follow_the_final_round_and_use_the_frozen_columns():
    rows = t3.quality_rows(make_run("r", RULE), make_run("c", CREW), make_run("p", CREW))
    assert list(rows.columns) == EVAL_COLS
    assert list(rows["stage"]) == ["loop_rule_R2", "loop_crew_live_R2", "loop_crew_replay_R2"]
    assert set(rows["eval_set"]) == {"test"}


def test_quality_rows_without_a_replay_has_two_rows():
    assert len(t3.quality_rows(make_run("r", RULE), make_run("c", CREW))) == 2


def test_quality_rows_refuse_a_crew_run_that_is_not_comparable():
    with pytest.raises(ValueError, match="R0 differs"):
        t3.quality_rows(make_run("r", RULE), make_run("c", CREW, rec=0.5))
    with pytest.raises(ValueError, match="rounds differ"):
        t3.quality_rows(make_run("r", RULE), make_run("c", CREW), make_run("p", CREW + ["rule"]))


def test_write_eval_tier3_round_trips_and_rejects_other_columns(tmp_path):
    rows = t3.quality_rows(make_run("r", RULE), make_run("c", CREW))
    p = tmp_path / "e.csv"
    t3.write_eval_tier3(rows, p)
    assert list(pd.read_csv(p).columns) == EVAL_COLS and len(pd.read_csv(p)) == 2
    with pytest.raises(ValueError, match="EVAL_COLS"):
        t3.write_eval_tier3(rows.drop(columns=["auc"]), p)


def test_committed_files_agree_with_each_other():
    rule, crew = t3.load_rounds(t3.ROUNDS_RULE), t3.load_rounds(t3.ROUNDS_RECORDED)
    rule_rows, crew_rows = t3.get_run(rule, t3.run_ids(rule)[-1]), t3.get_run(crew, t3.run_ids(crew)[-1])
    t3.check_comparable(rule_rows, crew_rows)
    committed = pd.read_csv(t3.EVAL_TIER3)
    assert list(committed.columns) == EVAL_COLS
    live = committed[committed["stage"].str.startswith("loop_crew_live")].iloc[0]
    assert live["rec"] == pytest.approx(float(t3.final_row(crew_rows)["rec"]))


def call(run="r", rnd=1, agent="A1", step="drift", kind="model", status=LIVE, ms=1000.0, tin=10, tout=5,
         start="2026-10-04T00:00:00.000+00:00", end="2026-10-04T00:00:01.000+00:00"):
    return {"run": run, "round": rnd, "agent": agent, "step": step, "kind": kind, "status": status, "start": start,
            "end": end, "ms": ms, "tokens_in": tin, "tokens_out": tout, "error": None}


def row(table, name):
    return table[table["row"] == name].iloc[0]


def test_live_runs_need_a_live_model_call():
    records = [call(run="a"), call(run="b", status=FALLBACK), call(run="c", kind="retrieval", status=REPLAY),
               call(run="d", kind="tool", status=LIVE)]
    assert t3.live_runs(records) == ["a"]


def test_latency_without_a_live_run_is_an_error():
    with pytest.raises(ValueError, match="no LIVE model calls"):
        t3.latency_table([call(status=REPLAY, kind="retrieval")])


def test_agent_rows_count_live_model_calls_only():
    records = [call(ms=100.0), call(rnd=2, ms=300.0, tin=20, tout=1), call(rnd=3, ms=9999.0, status=FALLBACK),
               call(rnd=4, ms=7.0, kind="retrieval", status=REPLAY), call(rnd=5, ms=1.0, kind="tool", status=LIVE)]
    a1 = row(t3.latency_table(records), "A1")
    assert (a1["n"], a1["p50_ms"], a1["p95_ms"], a1["tokens_in"], a1["tokens_out"], a1["n_model_fallback"]) == (
        2, 100.0, 300.0, 30, 6, 1)


def test_an_agent_with_no_live_call_has_empty_numbers_and_a_note():
    table = t3.latency_table([call(), call(agent="A3", status=FALLBACK)])
    a3, a4 = row(table, "A3"), row(table, "A4")
    assert a3["n"] == 0 and pd.isna(a3["p95_ms"]) and a3["n_model_fallback"] == 1 and "fell back" in a3["note"]
    assert a4["n"] == 0 and a4["n_model_fallback"] == 0 and "not run in the crew" in a4["note"]


def test_a_run_without_live_calls_is_left_out_by_default():
    table = t3.latency_table([call(run="a", ms=100.0), call(run="b", ms=5000.0, status=FALLBACK)])
    assert row(table, "A1")["n"] == 1 and row(table, "A1")["runs"] == "a"


def test_tokens_are_blank_when_no_live_call_reports_them():
    a1 = row(t3.latency_table([call(tin=None, tout=None)]), "A1")
    assert pd.isna(a1["tokens_in"]) and pd.isna(a1["tokens_out"])


def test_round_rows_follow_the_plan_formula_with_run_round_tool_time():
    # A1 1 s, A3 3 s (parallel, so max = 3 s), A2 2 s, run_round tool 0.5 s, A5 1.5 s: 3 + 2 + 0.5 + 1.5 = 7 s
    t = "2026-10-04T00:00:0{}.000+00:00"
    records = [
        call(agent="A1", ms=1000.0, start=t.format(0), end=t.format(1)),
        call(agent="A3", step="errors", ms=3000.0, start=t.format(0), end=t.format(3)),
        call(agent="A2", step="decide", ms=2000.0, start=t.format(3), end=t.format(5)),
        call(agent="loop", step="run_round", kind="tool", status=None, ms=500.0, tin=None, tout=None,
             start="2026-10-04T00:00:05.000+00:00", end="2026-10-04T00:00:05.500+00:00"),
        call(agent="A5", step="audit", ms=1500.0, start="2026-10-04T00:00:05.500+00:00",
             end="2026-10-04T00:00:07.000+00:00"),
    ]
    table = t3.latency_table(records)
    assert row(table, "round_time")["p50_ms"] == 7000.0 and row(table, "round_time")["n"] == 1
    assert row(table, "run_round_tool")["p50_ms"] == 500.0
    assert row(table, "round_span")["p50_ms"] == 7000.0


def test_round_rows_skip_rounds_without_a_run_round_call():
    records = [call(rnd=0, agent="A5", step="audit", ms=50.0)]
    records.append(call(rnd=1, agent="loop", step="run_round", kind="tool", status=None, ms=10.0, tin=None, tout=None))
    records.append(call(rnd=1))
    assert row(t3.latency_table(records), "round_time")["n"] == 1


def test_a_decision_agent_inside_run_round_is_not_counted_twice():
    # run_round holds A1 (1 s) inside its 1.2 s span: the tool part is 0.2 s and round_time 1.2 s
    outer = call(agent="loop", step="run_round", kind="tool", status=None, ms=1200.0, tin=None, tout=None,
                 start="2026-10-04T00:00:00.000+00:00", end="2026-10-04T00:00:01.200+00:00")
    table = t3.latency_table([outer, call(ms=1000.0)])
    assert row(table, "run_round_tool")["p50_ms"] == pytest.approx(200.0)
    assert row(table, "round_time")["p50_ms"] == pytest.approx(1200.0)


def test_several_runs_pool_their_calls_and_rounds():
    def run_calls(run, ms):
        return [call(run=run, ms=ms), call(run=run, agent="loop", step="run_round", kind="tool", status=None,
                                           ms=5.0, tin=None, tout=None)]
    table = t3.latency_table(run_calls("a", 100.0) + run_calls("b", 300.0))
    assert row(table, "A1")["n"] == 2 and row(table, "round_time")["n"] == 2
    assert row(table, "A1")["runs"] == "a b"


def test_write_latency_round_trips_and_rejects_other_columns(tmp_path):
    table = t3.latency_table([call(), call(agent="loop", step="run_round", kind="tool", status=None, ms=5.0,
                                           tin=None, tout=None)])
    p = tmp_path / "l.csv"
    t3.write_latency(table, p)
    assert list(pd.read_csv(p).columns) == t3.LATENCY_COLS
    with pytest.raises(ValueError, match="LATENCY_COLS"):
        t3.write_latency(table.drop(columns=["note"]), p)


@pytest.mark.parametrize("p95_ms, word", [(15000.0, "within"), (15001.0, "over")])
def test_budget_line_compares_p95_with_the_budget_and_shows_n(p95_ms, word):
    table = pd.DataFrame([{"row": "round_time", "n": 7, "p50_ms": 1000.0, "p95_ms": p95_ms}])
    line = t3.budget_line(table)
    assert word in line and "n = 7 rounds" in line and "judgement" in line


def test_committed_latency_file_has_the_fixed_columns_and_rows():
    committed = pd.read_csv(t3.LATENCY_TIER3)
    assert list(committed.columns) == t3.LATENCY_COLS
    assert list(committed["row"]) == [*t3.AGENTS, "run_round_tool", "round_time", "round_span"]
