"""Review loop (Tier 2, step 11, the loop.py half): one round = score, verify, reveal, refit, re-threshold.

Rounds and rows. R0 is the organizers' starter rule (blend w=0.45, cutoff 0.50) with no batch and no
labels: its row is ladder row 2. Rounds 1..n_rounds each take one batch from the oracle.

Two rules at a time. `live` decides who goes to the verify band. `candidate` is the stack refit after
the last round, shadowing live. While the mode is SHADOW the starter stays live: each round the candidate
is scored on that round's audit slice before the refit, and after PROMOTE_STREAK rounds in a row within
cap + PROMOTE_SLACK the newest candidate becomes live (mode ACTIVE). Evidence is on the previous
candidate, the promoted one is the one refit on this round's labels too. A round only counts toward
the streak with at least PROMOTE_MIN_ADULTS audit adults, and a promote is refused (action re-tune,
mode stays SHADOW) when the new candidate's thresholds carry an INSUFFICIENT_* flag.

PSI. Each round is compared with the scores the live rule gave earlier batches when they arrived
(state.live_scores), never re-scored, so a refit model is not measured on rows it was fit on. The
history restarts on a promote (starter and stack scores are on different scales), so the round after a
promote has no psi. Once ACTIVE the live rule changes every refit, so the history mixes successive models.

Hold rule (enforced here, from policy.yaml): no refit and no new thresholds until the cumulative
revealed audit adults reach min_audit_adults. Until then the starter stays live and action is "hold".

Label rules. Labels come only through oracle.reveal(). The refit uses every revealed label. Thresholds
come from out-of-fold scores of the audit rows only (threshold_source="audit"). "all_verified" takes
every revealed row instead: the verify band is the top-scored accounts, so adults in it are the hard
ones and the picked cutoff is biased. The toggle exists to show that. The frozen test set is scored
for the report columns only (prec..auc) and never feeds state, thresholds or an agent.

Not here yet: agents. Every round is applied_source "rule" ("starter" for R0); rule_decision() is the
decision A2 will later be compared against and fall back to.

Run: python -m softsignal.loop [--source audit|all_verified] [--rounds N]
"""
import argparse
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.agent_timer import RUN_ROUND_STEP, AgentTimer, get_timer
from softsignal.data import ROOT, cv_folds, load_data
from softsignal.features import FEATURE_COLS, ID_COL, SEED, TARGET
from softsignal.metrics import ROUNDS_COLS, auc, prf, psi
from softsignal.oracle import Batch, Oracle
from softsignal.policy import (INSUFFICIENT_ADULTS, INSUFFICIENT_TEENS, Thresholds, assign_bands, load_policy,
                               pick_thresholds)
from softsignal.stack import Stack
from softsignal.text_model import TextMatrix, build_matrix
from softsignal.tier1 import STARTER_CUT, STARTER_W, activity_score, blend, style_score

ROUNDS_CSV = ROOT / "results" / "rounds.csv"
DECISIONS_JSONL = ROOT / "results" / "decisions.jsonl"

SHADOW, ACTIVE = "SHADOW", "ACTIVE"
HOLD, RETUNE, PROMOTE, STARTER = "hold", "re-tune", "promote", "starter"
SOURCE_RULE = "rule"
CAP_MIN, CAP_MAX = 0.08, 0.30  # clamp for any cap the loop or A2 applies (policy.py does not clamp)
PROMOTE_SLACK = 0.03  # SHADOW -> ACTIVE needs audit false-teen <= cap + this ...
PROMOTE_STREAK = 2  # ... in this many rounds in a row
PROMOTE_MIN_ADULTS = 20  # a round's audit slice needs this many adults to count toward the streak
UNSAFE_FLAGS = (INSUFFICIENT_ADULTS, INSUFFICIENT_TEENS)  # a candidate with these is never promoted
THRESHOLD_SOURCES = ("audit", "all_verified")
AGENT = "loop"
AGENT_KEYS = ("a1", "a2", "a3", "a4", "a5")


def clamp_cap(cap: float) -> float:
    return float(min(CAP_MAX, max(CAP_MIN, cap)))


def starter_score(df: pd.DataFrame) -> np.ndarray:
    """The starter blend. Rounded so a score of exactly 0.50 is not lost to float error (as tier1.flag)."""
    return blend(style_score(df), activity_score(df), STARTER_W).round(9).to_numpy()


@dataclass(frozen=True)
class Rule:
    """A scorer and the thresholds picked for it. model None is the starter blend."""

    th: Thresholds
    model: Stack | None = None

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return starter_score(df) if self.model is None else self.model.score(df)


def starter_rule(cap: float) -> Rule:
    """No soft band: t_soft equals t_verify."""
    th = Thresholds(STARTER_CUT, STARTER_CUT, cap, 0.0, 0, 0, (STARTER,))
    return Rule(th)


@dataclass
class State:
    live: Rule
    candidate: Rule | None = None
    mode: str = SHADOW
    streak: int = 0  # consecutive SHADOW rounds with the candidate within cap + slack
    round: int = 0
    seen: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=[ID_COL, *FEATURE_COLS]))
    # each earlier round's scores as the live rule gave them when the batch arrived (never re-scored)
    live_scores: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class Env:
    """Everything a round needs that is not state. test is for the report columns only."""

    oracle: Oracle
    test: pd.DataFrame
    tm: TextMatrix
    train_ids: tuple[str, ...]
    policy: dict
    timer: AgentTimer
    threshold_source: str = "audit"


@dataclass
class RoundResult:
    row: dict  # ROUNDS_COLS
    record: dict  # decisions.jsonl shape


def new_state(policy: dict) -> State:
    return State(live=starter_rule(clamp_cap(policy["cap_false_teen"])))


def make_env(
    train: pd.DataFrame,
    test: pd.DataFrame,
    policy: dict | None = None,
    threshold_source: str = "audit",
    timer: AgentTimer | None = None,
    tm: TextMatrix | None = None,
    **oracle_kw,
) -> Env:
    """Env over a train/test split. The oracle gets train rows and test ids only, never test labels."""
    if threshold_source not in THRESHOLD_SOURCES:
        raise ValueError(f"threshold_source must be one of {THRESHOLD_SOURCES}, got {threshold_source!r}")
    policy = load_policy() if policy is None else policy
    oracle_kw.setdefault("audit_per_batch", policy["audit_per_batch"])
    oracle_kw.setdefault("review_budget", policy["review_budget"])
    oracle = Oracle(train, test[ID_COL].tolist(), **oracle_kw)
    ids = tuple(train[ID_COL].astype(str))
    return Env(oracle, test, tm if tm is not None else build_matrix(ids), ids, policy,
               timer if timer is not None else get_timer(), threshold_source)


# ---- decisions ----
def rule_decision(state: State, policy: dict, n_audit_adults: int) -> dict:
    """The rule-based decision for this round: {"action", "cap"}. A2's baseline and fallback.

    hold until min_audit_adults audit adults are revealed; then re-tune every round; promote once the
    candidate has been within cap + slack for PROMOTE_STREAK rounds in a row (state.streak, already
    updated for this round).
    """
    cap = clamp_cap(policy["cap_false_teen"])
    if n_audit_adults < policy["min_audit_adults"]:
        return {"action": HOLD, "cap": cap}
    if state.mode == SHADOW and state.streak >= PROMOTE_STREAK:
        return {"action": PROMOTE, "cap": cap}
    return {"action": RETUNE, "cap": cap}


def _tie_order(round_id: int, n: int) -> np.ndarray:
    """A fixed random rank per account (seeded by round), to break score ties without favouring ids."""
    return np.random.default_rng([SEED, round_id, 11]).permutation(n)


def pick_verify(scores: np.ndarray, bands: np.ndarray, ids: list[str], budget: int, round_id: int) -> list[str]:
    """Verify-band ids cut to the budget: highest score first, ties in a seeded random order."""
    flagged = np.flatnonzero(bands == "verify")
    order = flagged[np.lexsort((_tie_order(round_id, len(scores))[flagged], -scores[flagged]))]
    return [ids[i] for i in order[:budget]]


def false_teen(y: np.ndarray, scores: np.ndarray, t_verify: float) -> float | None:
    """False-teen at the verify cutoff on labelled rows; None when there is no adult to measure it on."""
    if not (y == 0).any():
        return None
    return float(prf(y, (scores >= t_verify).astype(int))[2])


# ---- refit ----
def labelled_rows(env: Env, state: State) -> pd.DataFrame:
    """Every revealed label joined to its features: ID_COL, FEATURE_COLS, TARGET, in_audit."""
    lab = env.oracle.revealed("all")[[ID_COL, TARGET, "in_audit"]]
    return lab.merge(state.seen, on=ID_COL, how="left", validate="one_to_one").reset_index(drop=True)


def refit(env: Env, state: State, cap: float) -> Rule:
    """A new candidate: a stack fit on all revealed labels, thresholds from out-of-fold scores.

    Each revealed row is scored by a stack fit without it (5 folds), so the audit rows the thresholds
    come from are never scored by a model that saw them.
    """
    frame = labelled_rows(env, state)
    fit = lambda df: Stack.fit(df, tm=env.tm, train_ids=env.train_ids)  # noqa: E731
    oof = np.full(len(frame), np.nan)
    for fit_idx, val_idx in cv_folds(frame):
        oof[val_idx] = fit(frame.iloc[fit_idx].reset_index(drop=True)).score(frame.iloc[val_idx])
    keep = (frame["in_audit"] if env.threshold_source == "audit" else pd.Series(True, index=frame.index)).to_numpy()
    th = pick_thresholds(oof[keep], frame.loc[keep, TARGET].to_numpy(), cap=cap,
                         soft_recall=env.policy["soft_recall"], margin=env.policy["cap_margin"], prior=state.live.th)
    return Rule(th, fit(frame))


# ---- reporting ----
def report_metrics(rule: Rule, test: pd.DataFrame) -> dict:
    """prec, rec, ft, mt, auc of the live rule on the frozen test set. For the report only."""
    s, y = rule.score(test), test[TARGET].to_numpy()
    prec, rec, ft, mt = prf(y, (s >= rule.th.t_verify).astype(int))
    return {"prec": prec, "rec": rec, "ft": ft, "mt": mt, "auc": auc(y, s)}


def _decision_block(rule_dec: dict, live: Rule) -> dict:
    return {"cutoff": float(live.th.t_verify), "cap": rule_dec["cap"], "action": rule_dec["action"]}


def make_record(run: str, rnd: int, decision: dict, source: str) -> dict:
    """decisions.jsonl line. Agents are not built yet, so every agent block is empty."""
    rec = {"run": run, "round": rnd}
    rec.update({k: {"status": None, "output": None, "fallback_reason": None} for k in AGENT_KEYS})
    rec.update({"rule_decision": dict(decision), "diff": {}, "applied": {"decision": dict(decision), "source": source}})
    return rec


def make_row(env: Env, state: State, rnd: int, action: str, source: str, **kw) -> dict:
    live = state.live
    row = {
        "run": env.timer.run, "round": rnd, "mode": state.mode, "action": action, "applied_source": source,
        "cap": live.th.cap, "t_soft": None if live.model is None else live.th.t_soft,
        "t_verify": live.th.t_verify, "n_flagged": 0, "n_verify": 0,
        "n_labels": len(env.oracle.revealed("all")), "n_audit_adults": env.oracle.audit_counts()["adults"],
        "audit_ft": None, "psi": None, "refit_s": None,
    }
    row.update(kw)
    row.update(report_metrics(live, env.test))
    return {c: row[c] for c in ROUNDS_COLS}


def round0(env: Env, state: State) -> RoundResult:
    """R0: the starter rule on the frozen test set, before any batch or label."""
    env.timer.round = 0
    dec = {"cutoff": float(state.live.th.t_verify), "cap": clamp_cap(env.policy["cap_false_teen"]), "action": STARTER}
    return RoundResult(make_row(env, state, 0, STARTER, STARTER), make_record(env.timer.run, 0, dec, STARTER))


# ---- one round ----
def run_round(state: State, batch: Batch, env: Env) -> RoundResult:
    """Run one round and update state in place. See the module docstring for the order."""
    env.timer.round = batch.round
    with env.timer.call(AGENT, RUN_ROUND_STEP, "tool"):
        return _run_round(state, batch, env)


def _run_round(state: State, batch: Batch, env: Env) -> RoundResult:
    rows, ids, oracle = batch.rows, list(batch.ids), env.oracle
    prior = state.seen

    # 1. score with the live rule, band, cut the verify band to the budget
    live_s = state.live.score(rows)
    bands = assign_bands(live_s, state.live.th)
    verify_ids = pick_verify(live_s, bands, ids, oracle.verify_budget(batch), batch.round)
    n_flagged = int((bands == "verify").sum())

    # 2. labels only through the oracle
    reveal = oracle.reveal(batch, verify_ids)
    lab = reveal.labels.set_index(ID_COL)
    audit_ids = [i for i in ids if i in lab.index and lab.at[i, "in_audit"]]
    pos = {i: k for k, i in enumerate(ids)}
    a_idx = [pos[i] for i in audit_ids]
    y_audit = lab.loc[audit_ids, TARGET].to_numpy()
    # state moves only once the reveal has succeeded; refit reads state.seen with this round in it
    state.seen = pd.concat([prior, rows], ignore_index=True) if len(prior) else rows.copy()
    state.round = batch.round
    ref_scores = state.live_scores
    state.live_scores = np.concatenate([ref_scores, live_s])

    # 3. this round's audit slice, scored BEFORE the refit, so these false-teen rates are never in-sample
    audit_ft = false_teen(y_audit, live_s[a_idx], state.live.th.t_verify)
    cand_ft = None
    if state.candidate is not None:
        cand_ft = false_teen(y_audit, state.candidate.score(rows)[a_idx], state.candidate.th.t_verify)
    psi_val = psi(ref_scores, live_s) if len(ref_scores) else None

    # 4. decision: streak first, then the rule
    if state.mode == SHADOW:
        enough = int((y_audit == 0).sum()) >= PROMOTE_MIN_ADULTS
        ok = enough and cand_ft is not None and cand_ft <= clamp_cap(env.policy["cap_false_teen"]) + PROMOTE_SLACK
        state.streak = state.streak + 1 if ok else 0
    rule_dec = rule_decision(state, env.policy, oracle.audit_counts()["adults"])

    # 5. apply it
    refit_s = None
    if rule_dec["action"] != HOLD:
        t0 = time.perf_counter()
        with env.timer.call(AGENT, "refit", "tool"):
            state.candidate = refit(env, state, rule_dec["cap"])
        refit_s = time.perf_counter() - t0
        if rule_dec["action"] == PROMOTE:
            if any(f in UNSAFE_FLAGS for f in state.candidate.th.flags):
                rule_dec = {**rule_dec, "action": RETUNE}  # refused: try again next round
            else:
                state.mode = ACTIVE
                state.live_scores = np.empty(0)  # the starter's scores are not on the stack's scale
        if state.mode == ACTIVE:
            state.live = state.candidate
    decision = _decision_block(rule_dec, state.live)
    row = make_row(env, state, batch.round, rule_dec["action"], SOURCE_RULE, n_flagged=n_flagged,
                   n_verify=len(verify_ids), audit_ft=audit_ft, psi=psi_val, refit_s=refit_s)
    return RoundResult(row, make_record(env.timer.run, batch.round, decision, SOURCE_RULE))


# ---- whole run ----
def run_loop(env: Env, n_rounds: int | None = None, state: State | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """R0 then each batch the oracle has (at most n_rounds). Returns the rounds table and the records.

    Pass a state to read the final live rule and candidate afterwards (it is updated in place).
    """
    state = new_state(env.policy) if state is None else state
    results = [round0(env, state)]
    for batch in env.oracle:
        if n_rounds is not None and batch.round > n_rounds:
            break
        results.append(run_round(state, batch, env))
    return pd.DataFrame([r.row for r in results], columns=ROUNDS_COLS), [r.record for r in results]


def write_run(rounds: pd.DataFrame, records: list[dict], rounds_path: Path = ROUNDS_CSV,
              decisions_path: Path = DECISIONS_JSONL) -> None:
    """Append this run to rounds.csv and decisions.jsonl (rows carry their run id; header written once).

    Raises ValueError, writing nothing, if rounds.csv already has a different header. Both files are
    built in full next to the originals and then swapped in, so a failure never leaves one half-written.
    """
    rounds_path.parent.mkdir(parents=True, exist_ok=True)
    old_rounds = rounds_path.read_text(encoding="utf-8") if rounds_path.exists() else ""
    if old_rounds.strip():
        header = old_rounds.splitlines()[0].split(",")
        if header != list(rounds.columns):
            raise ValueError(f"{rounds_path.name} has header {header}, this run has {list(rounds.columns)}")
    buf = io.StringIO()
    rounds.to_csv(buf, header=not old_rounds.strip(), index=False, lineterminator="\n")
    old_dec = decisions_path.read_text(encoding="utf-8") if decisions_path.exists() else ""
    new_dec = "".join(json.dumps(r) + "\n" for r in records)
    tmps = []
    try:
        for path, text in ((rounds_path, old_rounds + buf.getvalue()), (decisions_path, old_dec + new_dec)):
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8", newline="")
            tmps.append((tmp, path))
        for tmp, path in tmps:
            os.replace(tmp, path)
    finally:
        for tmp, _ in tmps:
            tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=THRESHOLD_SOURCES, default="audit")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--no-write", action="store_true", help="print only, leave rounds.csv and decisions.jsonl alone")
    args = ap.parse_args()
    train, test = load_data(on_param_mismatch="error")
    env = make_env(train, test, threshold_source=args.source)
    rounds, records = run_loop(env, args.rounds)
    pd.set_option("display.width", 220)
    print(rounds.drop(columns=["run"]).round(3).to_string(index=False))
    if not args.no_write:
        write_run(rounds, records)
        print(f"appended run {env.timer.run} to {ROUNDS_CSV.name} and {DECISIONS_JSONL.name}")


if __name__ == "__main__":
    main()
