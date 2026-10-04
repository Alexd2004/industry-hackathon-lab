"""recorded_check: the safety scan of the committed recording. No network."""
import json

import pytest

from softsignal import recorded_check as rc

TEST_IDS = frozenset({"B1015252", "B1045831"})


def jsonl(tmp_path, *records):
    p = tmp_path / "decisions_recorded.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return p


def test_clean_records_pass(tmp_path):
    p = jsonl(tmp_path, {"run": "r1", "round": 0, "a1": {"status": "LIVE", "output": {"reason": "PSI 0.07 is low"}}})
    assert rc.scan_file(p, TEST_IDS) == []


@pytest.mark.parametrize("text, kind", [
    ("key sk-ant-api03-abcdefghijklmnop1234", "secret"),
    ("ANTHROPIC_API_KEY=abc", "secret"),
    ("header x-api-key was set", "secret"),
    ("account B1015252 was missed", "test_id"),
    ("the age of the account", "forbidden"),
    ("job and gender", "forbidden"),
])
def test_each_kind_of_problem_in_a_value_is_found(tmp_path, text, kind):
    p = jsonl(tmp_path, {"run": "r1", "round": 0, "a1": {"output": {"reason": text}}})
    assert any(f": {kind}:" in problem for problem in rc.scan_file(p, TEST_IDS))


def test_a_label_key_is_found_at_any_depth_but_the_word_average_is_not_forbidden(tmp_path):
    p = jsonl(tmp_path, {"run": "r1", "round": 0, "a3": {"output": {"rows": [{"label_teen": 1}]}},
                         "a4": {"output": {"reason": "average word length, message, page"}}})
    problems = rc.scan_file(p, TEST_IDS)
    assert problems == ["decisions_recorded.jsonl: label_key: label_teen"]


def test_csv_header_and_values_are_scanned(tmp_path):
    p = tmp_path / "rounds_recorded.csv"
    p.write_text("run,round,label_teen\nr1,0,B1045831\n", encoding="utf-8")
    problems = rc.scan_file(p, TEST_IDS)
    assert "rounds_recorded.csv: label_key: label_teen" in problems
    assert "rounds_recorded.csv: test_id: B1045831" in problems


def test_a_missing_file_and_bad_json_are_problems(tmp_path):
    assert rc.scan_file(tmp_path / "nope.csv", TEST_IDS) == ["nope.csv: missing"]
    p = tmp_path / "decisions_recorded.jsonl"
    p.write_text("{not json\n", encoding="utf-8")
    assert rc.scan_file(p, TEST_IDS) == ["decisions_recorded.jsonl: not valid JSON lines"]


def test_the_committed_recording_is_clean():
    assert rc.scan_recorded() == []
