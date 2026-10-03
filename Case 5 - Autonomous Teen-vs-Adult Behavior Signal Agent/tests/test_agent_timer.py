"""Tests for softsignal.agent_timer. Runs under pytest or `python -m unittest`."""
import json
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from softsignal.agent_timer import (  # noqa: E402
    AgentTimer, load_records, percentile, round_agent_summary, round_time_ms, summarize,
)


def fake_response(tin: int, tout: int) -> SimpleNamespace:
    return SimpleNamespace(usage=SimpleNamespace(input_tokens=tin, output_tokens=tout))


def rec(agent, step, kind, ms, rnd=1, tin=None, tout=None, status=None, error=None, run="r"):
    return {"run": run, "round": rnd, "agent": agent, "step": step, "kind": kind,
            "status": status, "start": "", "end": "", "ms": ms,
            "tokens_in": tin, "tokens_out": tout, "error": error}


class TimerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sub" / "calls.jsonl"
        self.timer = AgentTimer(self.path, run="test")
        self.timer.round = 2

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_context_manager_logs_model_call(self) -> None:
        with self.timer.call("A2", "decide", status="LIVE") as c:
            c.usage(fake_response(120, 30))
        [r] = load_records(self.path)
        self.assertEqual(set(r), {"run", "round", "agent", "step", "kind", "status", "start",
                                  "end", "ms", "tokens_in", "tokens_out", "error"})
        self.assertEqual((r["agent"], r["step"], r["kind"], r["status"]), ("A2", "decide", "model", "LIVE"))
        self.assertEqual((r["tokens_in"], r["tokens_out"]), (120, 30))
        self.assertEqual((r["run"], r["round"]), ("test", 2))
        self.assertIsNone(r["error"])
        self.assertGreaterEqual(r["ms"], 0)
        self.assertLessEqual(datetime.fromisoformat(r["start"]), datetime.fromisoformat(r["end"]))

    def test_tool_call_tokens_are_null(self) -> None:
        with self.timer.call("A1", "psi", kind="tool"):
            pass
        [r] = load_records(self.path)
        self.assertIsNone(r["tokens_in"])
        self.assertIsNone(r["tokens_out"])
        self.assertIsNone(r["status"])

    def test_decorator_reads_usage_for_model(self) -> None:
        @self.timer.timed("A4", "triage", kind="model")
        def ask() -> SimpleNamespace:
            return fake_response(50, 10)

        @self.timer.timed("A2", "run_round")
        def tool(x: int) -> int:
            return x + 1

        self.assertEqual(tool(1), 2)
        ask()
        tool_r, model_r = load_records(self.path)
        self.assertEqual(tool_r["kind"], "tool")
        self.assertIsNone(tool_r["tokens_in"])
        self.assertEqual((model_r["tokens_in"], model_r["tokens_out"]), (50, 10))

    def test_exception_is_logged_and_reraised(self) -> None:
        with self.assertRaises(TimeoutError):
            with self.timer.call("A3", "analyse") as c:
                c.status = "FALLBACK"
                raise TimeoutError("4 s")
        [r] = load_records(self.path)
        self.assertEqual(r["error"], "TimeoutError: 4 s")
        self.assertEqual(r["status"], "FALLBACK")

    def test_append_only_one_object_per_line(self) -> None:
        for i in range(3):
            with self.timer.call("A1", f"s{i}", kind="retrieval"):
                pass
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual([json.loads(x)["step"] for x in lines], ["s0", "s1", "s2"])
        # a second timer on the same file appends, not truncates
        with AgentTimer(self.path).call("A5", "audit"):
            pass
        self.assertEqual(len(self.path.read_text().splitlines()), 4)

    def test_rejects_bad_kind_and_status(self) -> None:
        with self.assertRaises(ValueError):
            with self.timer.call("A1", "x", kind="llm"):
                pass
        with self.assertRaises(ValueError):
            with self.timer.call("A1", "x", status="OK"):
                pass

    def test_ms_uses_perf_counter_not_wall_clock(self) -> None:
        # a wall-clock jump (NTP, DST) during the call must not change ms
        with mock.patch("softsignal.agent_timer.time.time", side_effect=[1000.0, -5.0]):
            with self.timer.call("A2", "decide"):
                time.sleep(0.02)
        [r] = load_records(self.path)
        self.assertGreaterEqual(r["ms"], 15)
        self.assertLess(r["ms"], 1000)

    def test_invalid_status_set_in_block_is_logged_then_rejected(self) -> None:
        with self.assertRaises(ValueError):
            with self.timer.call("A2", "decide") as c:
                c.status = "OK"
        self.assertEqual(len(load_records(self.path)), 1)

    def test_default_run_ids_differ(self) -> None:
        a, b = AgentTimer(self.path), AgentTimer(self.path)
        self.assertIsNotNone(a.run)
        self.assertNotEqual(a.run, b.run)

    def test_threads_write_whole_lines(self) -> None:
        def work(name: str) -> None:
            for _ in range(50):
                with self.timer.call(name, "ask"):
                    pass

        threads = [threading.Thread(target=work, args=(f"A{i}",)) for i in range(1, 6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 250)
        for line in lines:
            json.loads(line)

    def test_half_written_last_line_is_skipped(self) -> None:
        with self.timer.call("A1", "ask"):
            pass
        with self.path.open("a") as f:
            f.write('{"run": "test", "rou')
        self.assertEqual(len(load_records(self.path)), 1)

    def test_missing_log_is_empty(self) -> None:
        self.assertEqual(load_records(Path(self.tmp.name) / "nope.jsonl"), [])


class SummaryTest(unittest.TestCase):
    def test_percentile_nearest_rank(self) -> None:
        xs = [15, 20, 35, 40, 50]
        self.assertEqual(percentile(xs, 50), 35)
        self.assertEqual(percentile(xs, 95), 50)
        self.assertEqual(percentile(list(range(1, 101)), 95), 95)
        self.assertEqual(percentile([7], 50), 7)
        self.assertEqual(percentile([7], 95), 7)
        self.assertIsNone(percentile([], 50))

    def test_percentile_edges(self) -> None:
        self.assertEqual(percentile([20, 10], 50), 10)
        self.assertEqual(percentile([20, 10], 95), 20)
        self.assertEqual(percentile([3, 1, 2], 0), 1)
        self.assertEqual(percentile([3, 1, 2], 100), 3)
        # float trap: 7 / 100 * 100 = 7.000000000000001 would give rank 8
        self.assertEqual(percentile(list(range(1, 101)), 7), 7)
        for bad in (-1, 101):
            with self.assertRaises(ValueError):
                percentile([1, 2], bad)

    def test_replay_excluded_from_latency_by_default(self) -> None:
        records = [
            rec("A2", "ask", "model", 900, status="LIVE"),
            rec("A2", "ask", "model", 1, status="REPLAY"),
        ]
        self.assertEqual(summarize(records)["by_agent"]["A2"]["n"], 1)
        self.assertEqual(summarize(records)["by_agent"]["A2"]["p50_ms"], 900)
        self.assertEqual(summarize(records, exclude_status=())["by_agent"]["A2"]["n"], 2)

    def test_summarize_groups_and_tokens(self) -> None:
        records = [
            rec("A1", "ask", "model", 10, tin=100, tout=20),
            rec("A1", "ask", "model", 30, tin=110, tout=25),
            rec("A1", "psi", "tool", 5),
            rec("A2", "read", "retrieval", 1),
        ]
        s = summarize(records)
        self.assertEqual(s["by_agent"]["A1"]["n"], 3)
        self.assertEqual(s["by_agent"]["A1"]["tokens_in"], 210)
        self.assertEqual(s["by_agent_kind"]["A1/model"]["p50_ms"], 10)
        self.assertEqual(s["by_agent_kind"]["A1/model"]["p95_ms"], 30)
        self.assertIsNone(s["by_kind"]["tool"]["tokens_in"])
        self.assertIsNone(s["by_agent"]["A2"]["tokens_out"])

    def test_round_summary_and_round_time(self) -> None:
        records = [
            rec("A1", "ask", "model", 100, tin=10, tout=2, status="LIVE"),
            rec("A3", "ask", "model", 300, status="FALLBACK", error="TimeoutError: x"),
            rec("A2", "ask", "model", 200, tin=5, tout=1, status="REPLAY"),
            rec("A2", "run_round", "tool", 50),
            rec("A4", "ask", "model", 40, status="LIVE"),
            rec("A5", "ask", "model", 60, status="LIVE"),
            rec("A1", "ask", "model", 999, rnd=2),
            rec("A1", "ask", "model", 999, run="other"),
        ]
        agents = round_agent_summary(records, 1, run="r")
        self.assertEqual(agents["A2"]["ms"], 200)  # run_round excluded
        self.assertEqual(agents["A2"]["status"], "REPLAY")
        self.assertEqual(agents["A3"]["status"], "FALLBACK")
        self.assertEqual(agents["A3"]["errors"], ["ask: TimeoutError: x"])
        self.assertIsNone(agents["A3"]["tokens_in"])
        self.assertEqual(agents["A1"]["tokens_in"], 10)
        # max(100, 300) + 200 + 50 + max(40, 60)
        self.assertEqual(round_time_ms(records, 1, run="r"), 610)
        # round 0 with only A2 running: missing agents count as 0
        self.assertEqual(round_time_ms([rec("A2", "ask", "model", 7, rnd=0)], 0), 7)

    def test_round0_without_a1_a3_model_calls(self) -> None:
        # R0: A1/A3 return insufficient_data with no call; A2 uses the starter rule
        records = [rec("A2", "run_round", "tool", 50, rnd=0),
                   rec("A4", "ask", "model", 30, rnd=0), rec("A5", "ask", "model", 20, rnd=0)]
        self.assertEqual(round_time_ms(records, 0), 80)
        self.assertNotIn("A1", round_agent_summary(records, 0))

    def test_timed_out_call_counts_toward_round_time(self) -> None:
        records = [
            rec("A1", "ask", "model", 4000, status="FALLBACK", error="APITimeoutError: x"),
            rec("A1", "psi_fallback", "tool", 5),
            rec("A3", "ask", "model", 100, status="LIVE"),
            rec("A2", "run_round", "tool", 10),
        ]
        a1 = round_agent_summary(records, 1)["A1"]
        self.assertEqual((a1["ms"], a1["model_ms"], a1["tool_ms"]), (4005, 4000, 5))
        self.assertEqual(round_time_ms(records, 1), 4015)

    def test_mixed_runs_need_run_filter(self) -> None:
        records = [rec("A1", "ask", "model", 10, run="a"), rec("A1", "ask", "model", 20, run="b")]
        with self.assertRaises(ValueError):
            round_time_ms(records, 1)
        self.assertEqual(round_time_ms(records, 1, run="b"), 20)


if __name__ == "__main__":
    unittest.main()
