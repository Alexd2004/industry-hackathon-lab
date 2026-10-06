"""Web console server: the static page plus a small JSON API over the results files and the real runs.

Run: python -m softsignal.web [--port 8000] [--host 127.0.0.1]   then open http://127.0.0.1:8000

Standard library only (http.server), no network needed: agents run live only with ANTHROPIC_API_KEY (or .env),
otherwise they replay the committed recording or fall back, exactly as in crew.py. One job at a time:

    POST /api/run {"mode": "crew"}   crew.run_crew, A2's decision applied, each round appended to rounds.csv
    POST /api/run {"mode": "rule"}   loop.run_loop, the rule decides, no agents, each round appended
    POST /api/pipeline               agent_starter.py (Tier 1), python -m softsignal.stack, python -m softsignal.explain
                                     (refits the models and rewrites eval_tier1.csv, ranked.csv, contrib.csv,
                                     policy_grid.csv; the cached TF-IDF matrix is reused)

    GET /api/runs     every run (payload.load_runs) and the job status; the page polls this while a job runs
    GET /api/static   ranked list, cap grid, login heatmap, signal weights, test counts, latency
    GET /api/ladder   the results ladder (first call computes the keyword baseline and tabular LR, a few seconds)
    GET /api/job      the job status and the last lines of its log
    GET /api/list?run=<source:run id>   that run's own likely-teen list (web/run_list.py), or rows null if it has none
    POST /api/review  {"account", "verdict": agree|disagree|null}: a reviewer's call, kept in results/reviews.json
                      (gitignored; never read by a model or the loop)
"""
import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from softsignal.data import ROOT
from softsignal.web import payload

STATIC = Path(__file__).resolve().parent / "static"
REVIEWS = payload.RESULTS / "reviews.json"
VERDICTS = ("agree", "disagree", None)
PIPELINE = (
    ("Tier 1: starter, tune() and changed weight", [sys.executable, "agent_starter.py"]),
    ("Stack: nested out-of-fold refit", [sys.executable, "-m", "softsignal.stack"]),
    ("Explain: ranked list, contributions, cap grid", [sys.executable, "-m", "softsignal.explain"]),
)


@dataclass
class Job:
    kind: str | None = None  # "crew", "rule" or "pipeline"
    run: str | None = None  # the loop run id (AgentTimer run), None for the pipeline
    step: str | None = None
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    thread: threading.Thread | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=200))

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def view(self) -> dict:
        return {"kind": self.kind, "run": self.run, "step": self.step, "error": self.error, "running": self.running,
                "started": self.started, "finished": self.finished, "log": list(self.log)[-40:]}


_job = Job()
_job_lock = threading.Lock()
_reviews_lock = threading.Lock()


def _start(kind: str, target) -> Job | None:
    """Start target(job) in a daemon thread; None while another job is still going."""
    global _job
    with _job_lock:
        if _job.running:
            return None
        job = Job(kind=kind, started=time.time())
        job.thread = threading.Thread(target=_guard, args=(job, target), daemon=True, name=f"softsignal-{kind}")
        _job = job
        job.thread.start()
        return job


def _guard(job: Job, target) -> None:
    try:
        target(job)
    except Exception as e:  # noqa: BLE001 - shown on the page instead of dying silently in a thread
        job.error = f"{type(e).__name__}: {e}"[:500]
        job.log.append(job.error)
    finally:
        job.finished = time.time()


def _loop_job(mode: str):
    def target(job: Job) -> None:
        from softsignal.agent_timer import DEFAULT_LOG, AgentTimer
        from softsignal.data import load_data
        from softsignal.loop import (DECISIONS_JSONL, ROUNDS_CSV, check_rounds_header, make_env, new_state, run_loop,
                                     write_run)
        from softsignal.metrics import ROUNDS_COLS
        from softsignal.web.run_list import final_list, live_path, write_list

        check_rounds_header(ROUNDS_CSV, ROUNDS_COLS)  # fail now, not after every refit
        timer = AgentTimer(DEFAULT_LOG)
        job.run = timer.run
        job.step = "loading data and the TF-IDF cache"
        job.log.append(f"run {timer.run} ({mode})")
        train, test = load_data(on_param_mismatch="error")
        env = make_env(train, test, timer=timer)
        state = new_state(env.policy)  # read after the run: its final live rule scores the run's own list
        job.step = "round 0"

        def landed(row: dict) -> None:
            rnd = int(row["round"])
            job.step = f"round {rnd} done"
            job.log.append(f"R{rnd}: {row['action']} ({row['applied_source']}), "
                           f"recall {row['rec']:.3f}, false-teen {row['ft']:.3f}")

        if mode == "crew":
            from softsignal import crew
            from softsignal.agents.base import make_client
            from softsignal.replay import Replayer

            client = make_client()
            job.log.append("agents live" if client else "agents offline: recorded replay, else fallback")
            replayer = None if client else Replayer.from_file(crew.DECISIONS_RECORDED)
            job.step = "crew rounds (each lands in rounds.csv as it ends)"
            # run_crew appends each round itself (write=True); the page reads the rounds from the file as they land
            rounds, _ = crew.run_crew(env, client, None, True, ROUNDS_CSV, DECISIONS_JSONL, state=state,
                                      replayer=replayer, apply_a2=True)
            for row in rounds.to_dict("records"):
                landed(row)
        else:
            import pandas as pd

            def on_round(result) -> None:
                write_run(pd.DataFrame([result.row], columns=ROUNDS_COLS), [result.record], ROUNDS_CSV,
                          DECISIONS_JSONL, shadow=[result.shadow])
                landed(result.row)

            run_loop(env, state=state, on_round=on_round)
        job.step = "scoring the run's likely-teen list with its final model"
        write_list(final_list(state, test), live_path(timer.run))
        job.log.append(f"list: {state.mode}, {'stack' if state.live.model is not None else 'starter blend'}")
        job.step = "finished"
    return target


def _pipeline_job(job: Job) -> None:
    for name, cmd in PIPELINE:
        job.step = name
        job.log.append(f"$ {' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))}")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            job.log.append(line.rstrip())
        if proc.wait() != 0:
            raise RuntimeError(f"{name} exited with {proc.returncode}")
    payload._cache.pop("baselines", None)  # noqa: SLF001 - recompute on the next ladder request
    job.step = "finished"


def read_reviews() -> dict:
    try:
        return json.loads(REVIEWS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_review(account: str, verdict: str | None) -> dict:
    with _reviews_lock:
        reviews = read_reviews()
        if verdict is None:
            reviews.pop(account, None)
        else:
            reviews[account] = verdict
        tmp = REVIEWS.with_suffix(".tmp")
        tmp.write_text(json.dumps(reviews, indent=0, sort_keys=True), encoding="utf-8")
        tmp.replace(REVIEWS)
        return reviews


class Handler(BaseHTTPRequestHandler):
    server_version = "SoftSignal"

    def log_message(self, fmt, *args) -> None:  # quiet: the page polls every second
        if not self.path.startswith("/api/runs") and not self.path.startswith("/api/job"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _json(self, obj, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(obj, allow_nan=False, default=payload.clean).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> dict:
        from urllib.parse import parse_qs, urlsplit

        return {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            got = json.loads(self.rfile.read(n))
        except ValueError:
            return {}
        return got if isinstance(got, dict) else {}

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/runs":
                return self._json({"runs": payload.load_runs(), "job": _job.view()})
            if path == "/api/static":
                return self._json(payload.static_payload())
            if path == "/api/ladder":
                return self._json({"rows": payload.ladder()})
            if path == "/api/job":
                return self._json(_job.view())
            if path == "/api/reviews":
                return self._json(read_reviews())
            if path == "/api/list":
                return self._json(payload.run_list(self._query().get("run", "")))
        except Exception as e:  # noqa: BLE001 - a bad file shows as an error on the page, not a dead server
            return self._json({"error": f"{type(e).__name__}: {e}"[:500]}, HTTPStatus.INTERNAL_SERVER_ERROR)
        self._static(path)

    def do_POST(self) -> None:  # noqa: N802
        path, body = self.path.split("?", 1)[0], self._body()
        if path == "/api/run":
            mode = body.get("mode")
            if mode not in ("crew", "rule"):
                return self._json({"error": "mode must be crew or rule"}, HTTPStatus.BAD_REQUEST)
            job = _start(mode, _loop_job(mode))
        elif path == "/api/pipeline":
            job = _start("pipeline", _pipeline_job)
        elif path == "/api/review":
            account, verdict = body.get("account"), body.get("verdict")
            if not isinstance(account, str) or not account or verdict not in VERDICTS:
                return self._json({"error": "account and verdict (agree, disagree or null) needed"},
                                  HTTPStatus.BAD_REQUEST)
            return self._json(write_review(account, verdict))
        else:
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        if job is None:
            return self._json({"error": "a job is already running", "job": _job.view()}, HTTPStatus.CONFLICT)
        time.sleep(0.05)  # let the thread set its run id
        return self._json(job.view(), HTTPStatus.ACCEPTED)

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SoftSignal console on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
