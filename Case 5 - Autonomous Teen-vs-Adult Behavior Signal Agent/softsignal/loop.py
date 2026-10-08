"""Review loop (Tier 2, step 11, the loop.py half): one round = score, verify, reveal, refit, re-threshold.

Rounds and rows. R0 is the organizers' starter rule (blend w=0.45, cutoff 0.50) with no batch and no
labels: its row is ladder row 2. Rounds 1..n_rounds each take one batch from the oracle.

Two rules at a time. `live` decides who goes to the verify band. `candidate` is the stack refit after
the last round, shadowing live. While the mode is SHADOW the starter stays live: each round the candidate
is scored on that round's audit slice before the refit. The false teens and adults of the last
PROMOTE_STREAK rounds are pooled (state.window), and when the pooled rate is within cap + PROMOTE_SLACK
on at least PROMOTE_MIN_ADULTS pooled adults the newest candidate becomes live (mode ACTIVE). Pooling
(one test on about twice the adults) replaces judging each round alone, which was too noisy. The pooled
rounds can each have scored a different candidate (each round refits), so this measures the refit
process, not one fixed model. Evidence is on the previous candidate, the promoted one is the one refit
on this round's labels too. A round whose evidence candidate carries an INSUFFICIENT_* flag, or has no
adult, empties the window, and a promote is refused (action re-tune, mode stays SHADOW) when the new
candidate's thresholds carry an INSUFFICIENT_* flag.

PSI. Each round is compared with the scores the live rule gave earlier batches when they arrived
(state.live_scores), never re-scored, so a refit model is not measured on rows it was fit on. The
history restarts on a promote (starter and stack scores are on different scales), so the round after a
promote has no psi. Once ACTIVE the live rule changes every refit, so the history mixes successive models.

Challenger (policy.yaml promote_rule: challenger, the default since the pooled rule left a run flat until round 7).
Each round the model refit last round (the challenger) and the live rule are both scored on this round's audit
slice, before the refit, so neither has seen those labels. The challenger wins when its audit false-teen is within
cap + PROMOTE_SLACK on at least CHALLENGE_MIN_ADULTS audit adults and it catches at least as many audit teens as the
live rule, or when the live rule is over that bar and it is not; an INSUFFICIENT_* candidate never wins. A win sets
the streak to 1, so the rule's decision is promote in SHADOW; in ACTIVE a win lets this round's refit replace the
live rule and a loss keeps the live rule (the pooled rule replaced it on every refit). The newest refit is what
goes live, as with the pooled rule: the evidence is on the previous one. promote_rule: pooled keeps the original
rule described above.

Hold rule (enforced here, from policy.yaml): no refit and no new thresholds until the cumulative
revealed audit adults reach min_audit_adults. Until then the starter stays live and action is "hold".

Label rules. Labels come only through oracle.reveal(). The refit uses every revealed label. Thresholds
come from out-of-fold scores of the audit rows only (threshold_source="audit"). "all_verified" takes
every revealed row instead: the verify band is the top-scored accounts, so adults in it are the hard
ones and the picked cutoff is biased. The toggle exists to show that. The frozen test set is scored
for the report columns only (prec..auc) and never feeds state, thresholds or an agent.

Agents plug in through two optional callbacks (crew.py uses them). before_decision(state, batch, prior,
psi) runs after the reveal and the PSI, before the decision, and returns agent blocks for the record
(A1 now, A2 later reads them here); prior is the rows of every earlier batch, handed over explicitly.
A hook that raises never breaks the round: its error goes in the record (agent_error), the rule decides.
on_round(result) runs after each round, e.g. to append it to the files.

A2 plugs in through a third callback, decide(context, blocks) -> {"a2": block}, called after the streak update and
before the refit (DecisionContext: the live thresholds, audit counts, bounds, guards and rule_decision(), all from
before the apply step). Its block and the code-computed diff against rule_decision() are always logged. With
apply_a2 (crew mode) A2's own action and cap are applied, with the guards enforced again here (no refit before
the audit floor, no promote unless the rule's pooled test passed) and the cap clamped; applied_source is then "A2".
A FALLBACK, a missing block or a failing decider leaves the rule's decision applied. The promote gate always uses
the policy cap, and the refit on a promote reuses the cap the evidence candidate was refit at (what the gate
measured is what goes live, never an untested cap); A2's cap steers the refit on re-tune rounds. The evidence logs
policy_cap, refit_cap and cap_differs.
Without apply_a2 every round is applied_source "rule" ("starter" for R0). diff and the rounds.csv diff_count are
A2's own proposal (action, cap) against the rule's, whether or not it was applied: the guards in step 4b and the
promote cap override in step 5 can make the applied decision differ from A2's proposal, and that is not counted.

Run: python -m softsignal.loop [--mode rule|crew] [--source audit|all_verified] [--rounds N] [--no-write]
  --mode rule (default): the rule decides every round, no agents.
  --mode crew: crew.run_crew (A1, A2; A2's decision applied; agents offline unless ANTHROPIC_API_KEY).
Comparing the two modes' recall needs two full runs: the seed and the batch order are fixed, so both see the
same batches, but the rule's decision is a counterfactual decision, not a counterfactual outcome.
"""
import argparse
import contextlib
import io
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal import side_by_side
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
RUN_MODES = ("rule", "crew")  # the CLI switch: who decides each round
CAP_MIN, CAP_MAX = 0.08, 0.30  # clamp for any cap the loop or A2 applies (policy.py does not clamp)
WINDOW_MIN = 2  # fewest rounds of labels a refit window may keep: fewer leaves too few audit adults for thresholds
MIN_WINDOW_ROWS = 100  # a window with fewer labelled rows (or only one class) is not used: the refit keeps all rounds
MARGIN_MAX = 0.05  # clamp for any cap_margin A2 proposes: the verify cutoff aims at cap - margin (policy.py needs margin <= cap)
PROMOTE_SLACK = 0.03  # SHADOW -> ACTIVE needs the pooled audit false-teen <= cap + this ...
PROMOTE_STREAK = 2  # ... pooled over this many rounds in a row
PROMOTE_MIN_ADULTS = 40  # ... and over at least this many pooled audit adults (2 x 20, the old per-round floor)
UNSAFE_FLAGS = (INSUFFICIENT_ADULTS, INSUFFICIENT_TEENS)  # a candidate with these is never promoted
CHALLENGER, POOLED = "challenger", "pooled"  # policy.yaml promote_rule
CHALLENGE_MIN_ADULTS = 20  # a challenge on fewer audit adults in the round is not decided (about 27-37 per round)
THRESHOLD_SOURCES = ("audit", "all_verified")
AGENT = "loop"
AGENT_KEYS = ("a1", "a2", "a3", "a4", "a5")


def clamp_cap(cap: float) -> float:
    return float(min(CAP_MAX, max(CAP_MIN, cap)))


def clamp_window(window: int) -> int:
    """A refit window of at least WINDOW_MIN rounds (no upper limit: a long window is the same as all rounds)."""
    return max(WINDOW_MIN, int(window))


def clamp_margin(margin: float, cap: float) -> float:
    """A cap_margin inside 0..MARGIN_MAX, and never above the cap it is taken from."""
    return float(min(MARGIN_MAX, cap, max(0.0, margin)))


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
    candidate_round: int | None = None  # the round whose refit made the candidate (None: no candidate yet)
    candidate_cap: float | None = None  # the cap that refit used: what the promote gate's evidence was gathered at
    candidate_margin: float | None = None  # the cap_margin that refit used (None: the policy's), kept with the cap
    candidate_window: int | None = None  # the refit window that refit used (None: all rounds), kept with the cap
    mode: str = SHADOW
    streak: int = 0  # rounds in the window while it fills, then PROMOTE_STREAK if the pooled test passes, else 0
    window: list = field(default_factory=list)  # (audit adults, candidate false teens) of the last rounds, reassigned
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
    shadow: dict | None = None  # SHADOW_COLS: the model in training on the frozen test set (report only), or None


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
    """Env over a train/test split. The oracle gets train rows and test ids only, never test labels.
    With oracle_kw drift=Drift(...) the test report is scored on the drifted test rows (oracle.shift_frame)."""
    if threshold_source not in THRESHOLD_SOURCES:
        raise ValueError(f"threshold_source must be one of {THRESHOLD_SOURCES}, got {threshold_source!r}")
    policy = load_policy() if policy is None else policy
    oracle_kw.setdefault("audit_per_batch", policy["audit_per_batch"])
    oracle_kw.setdefault("review_budget", policy["review_budget"])
    oracle = Oracle(train, test[ID_COL].tolist(), **oracle_kw)
    ids = tuple(train[ID_COL].astype(str))
    report = oracle.shift_frame(test) if oracle.drift is not None else test  # score the drifted world, not the old one
    return Env(oracle, report, tm if tm is not None else build_matrix(ids), ids, policy,
               timer if timer is not None else get_timer(), threshold_source)


# ---- decisions ----
def rule_decision(state: State, policy: dict, n_audit_adults: int) -> dict:
    """The rule-based decision for this round: {"action", "cap"}. A2's baseline and fallback.

    hold until min_audit_adults audit adults are revealed; then re-tune every round; promote once the
    the pooled test over PROMOTE_STREAK rounds has passed (state.streak, already updated for this round).
    """
    cap = clamp_cap(policy["cap_false_teen"])
    if n_audit_adults < policy["min_audit_adults"]:
        return {"action": HOLD, "cap": cap}
    need = 1 if policy.get("promote_rule") == CHALLENGER else PROMOTE_STREAK  # challenger: one won round
    if state.mode == SHADOW and state.streak >= need:
        return {"action": PROMOTE, "cap": cap}
    return {"action": RETUNE, "cap": cap}


def audit_rates(y: np.ndarray, scores: np.ndarray, t_verify: float) -> tuple[float | None, float | None]:
    """(recall, false-teen) on labelled audit rows at a cutoff; None where the class is absent."""
    pred = (scores >= t_verify).astype(int)
    rec = float(pred[y == 1].mean()) if (y == 1).any() else None
    ft = float(pred[y == 0].mean()) if (y == 0).any() else None
    return rec, ft


def challenger_wins(live: tuple, cand: tuple, cap: float, n_adults: int) -> bool:
    """The challenger beats the live rule on this round's audit slice (see the module docstring)."""
    (live_rec, live_ft), (cand_rec, cand_ft) = live, cand
    bar = cap + PROMOTE_SLACK
    if n_adults < CHALLENGE_MIN_ADULTS or cand_rec is None or cand_ft is None or cand_ft > bar:
        return False
    if live_ft is None or live_ft > bar:
        return True
    return live_rec is None or cand_rec >= live_rec


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
def labelled_rows(env: Env, state: State, window: int | None = None) -> pd.DataFrame:
    """Every revealed label joined to its features: ID_COL, FEATURE_COLS, TARGET, in_audit. window keeps only the
    labels revealed in the last `window` rounds (None: all of them), so a refit can drop data from before a drift."""
    lab = env.oracle.revealed("all")
    if window is not None:
        lab = lab[lab["round"] > state.round - window]
    lab = lab[[ID_COL, TARGET, "in_audit"]]
    return lab.merge(state.seen, on=ID_COL, how="left", validate="one_to_one").reset_index(drop=True)


def effective_window(env: Env, state: State, window: int | None) -> int | None:
    """The window a refit will really use: None (all rounds) when none was asked for, when the window already
    covers every round, or when the windowed labels are too few or one-class to fit and threshold on."""
    if window is None or window >= state.round:
        return None
    frame = labelled_rows(env, state, window)
    return window if len(frame) >= MIN_WINDOW_ROWS and frame[TARGET].nunique() == 2 else None


def refit(env: Env, state: State, cap: float, margin: float | None = None, window: int | None = None) -> Rule:
    """A new candidate: a stack fit on the revealed labels (the last `window` rounds, None: all), thresholds from
    out-of-fold scores.

    margin: how far below the cap the verify cutoff aims; None is the policy's cap_margin.

    Each revealed row is scored by a stack fit without it (5 folds), so the audit rows the thresholds
    come from are never scored by a model that saw them.
    """
    frame = labelled_rows(env, state, window)
    fit = lambda df: Stack.fit(df, tm=env.tm, train_ids=env.train_ids)  # noqa: E731
    oof = np.full(len(frame), np.nan)
    for fit_idx, val_idx in cv_folds(frame):
        oof[val_idx] = fit(frame.iloc[fit_idx].reset_index(drop=True)).score(frame.iloc[val_idx])
    keep = (frame["in_audit"] if env.threshold_source == "audit" else pd.Series(True, index=frame.index)).to_numpy()
    th = pick_thresholds(oof[keep], frame.loc[keep, TARGET].to_numpy(), cap=cap,
                         soft_recall=env.policy["soft_recall"], margin=env.policy["cap_margin"] if margin is None else margin, prior=state.live.th)
    return Rule(th, fit(frame))


# ---- reporting ----
def report_metrics(rule: Rule, test: pd.DataFrame) -> dict:
    """prec, rec, ft, mt, auc of the live rule on the frozen test set. For the report only."""
    s, y = rule.score(test), test[TARGET].to_numpy()
    prec, rec, ft, mt = prf(y, (s >= rule.th.t_verify).astype(int))
    return {"prec": prec, "rec": rec, "ft": ft, "mt": mt, "auc": auc(y, s)}


def _decision_block(rule_dec: dict, live: Rule) -> dict:
    block = {"cutoff": float(live.th.t_verify), "cap": rule_dec["cap"], "action": rule_dec["action"]}
    if rule_dec.get("cap_margin") is not None:
        block["cap_margin"] = rule_dec["cap_margin"]
    if rule_dec.get("refit_window") is not None:
        block["refit_window"] = rule_dec["refit_window"]
    return block


def pooled_test(window: list, cap: float) -> tuple[int, int, bool]:
    """(pooled adults, pooled false teens, passes) for a window of (adults, false teens) rounds.

    Passes when there are at least PROMOTE_MIN_ADULTS pooled adults and the pooled false-teen rate is
    within cap + PROMOTE_SLACK. An empty window never passes.
    """
    adults, fts = sum(a for a, _ in window), sum(f for _, f in window)
    return adults, fts, bool(adults >= max(PROMOTE_MIN_ADULTS, 1) and fts / adults <= cap + PROMOTE_SLACK)


def _unsafe(rule: Rule) -> bool:
    """True when the rule's thresholds carry an INSUFFICIENT_* flag (the prior's cutoffs were kept)."""
    return any(f in UNSAFE_FLAGS for f in rule.th.flags)


def make_record(run: str, rnd: int, decision: dict, source: str, evidence: dict | None = None,
                rule_decision: dict | None = None) -> dict:
    """decisions.jsonl line. Every agent block starts empty; the round fills the ones that ran.

    decision is what was applied; rule_decision is what the rule would have decided (None: the same); its cutoff
    is None when the rule was not the one applied and chose another action or cap, because that cutoff is never
    computed.

    evidence (rounds 1+) is why the promote rule did or did not fire: round_audit_adults, cand_ft and
    cand_t_verify (the previous candidate, scored on this round's audit slice), cand_unsafe (that
    candidate carried an INSUFFICIENT_* flag, so the window is emptied), cand_age (rounds since that candidate was
    refit: 1 when every round refits, more when a hold skipped refits, so the pooled evidence is on an older model),
    pooled_adults and pooled_ft
    (the window the test used, None when it is empty), streak (after this round) and promote_refused
    (the new candidate carried an INSUFFICIENT_* flag). policy_cap is the cap the promote gate used, refit_cap
    the cap this round's refit used (None on hold; the evidence candidate's cap on a promote, A2's cap otherwise) and
    cap_differs says whether they differ, so an A2 cap that steered the thresholds is visible in the log.
    """
    rec = {"run": run, "round": rnd}
    rec.update({k: {"status": None, "output": None, "fallback_reason": None} for k in AGENT_KEYS})
    rec.update({"rule_decision": dict(decision if rule_decision is None else rule_decision), "diff": {},
                "applied": {"decision": dict(decision), "source": source}})
    if evidence is not None:
        rec["evidence"] = dict(evidence)
    return rec


def make_row(env: Env, state: State, rnd: int, action: str, source: str, **kw) -> dict:
    """rounds.csv row. mode, cap, t_soft, t_verify and the test columns (prec..auc) are end-of-round: after a
    promote or in ACTIVE they belong to the new live rule, which scores the next batch. n_flagged, n_verify,
    audit_ft and psi describe the rule that scored this round's batch."""
    live = state.live
    row = {
        "run": env.timer.run, "round": rnd, "mode": state.mode, "action": action, "applied_source": source,
        "diff_count": 0,
        "cap": live.th.cap, "t_soft": None if live.model is None else live.th.t_soft,
        "t_verify": live.th.t_verify, "n_flagged": 0, "n_verify": 0,
        "n_labels": len(env.oracle.revealed("all")), "n_audit_adults": env.oracle.audit_counts()["adults"],
        "audit_ft": None, "psi": None, "refit_s": None,
    }
    row.update(kw)
    row.update(report_metrics(live, env.test))
    return {c: row[c] for c in ROUNDS_COLS}


@dataclass(frozen=True)
class DecisionContext:
    """What a decider (A2) is shown for one round, built in code after the streak update and before the refit.
    The fields are the keyword arguments of contracts.a2_input, as plain Python (no numpy). mode, thresholds and
    guards are from BEFORE the round's apply step; audit.streak is already this round's."""

    round: int
    thresholds: dict  # the live rule: t_verify, t_soft, cap, flags
    audit: dict
    bounds: dict
    guards: dict  # hold_required, promote_allowed
    rule: dict  # rule_decision(): {action, cap}


BeforeDecision = Callable[[State, "Batch | None", pd.DataFrame, "float | None"], dict]
Decide = Callable[[DecisionContext, dict], dict]  # (context, the before_decision blocks) -> {"a2": block}
OnRound = Callable[[RoundResult], None]


def _py(v):
    """A numpy scalar as a plain Python value (a2_input rejects anything json cannot dump)."""
    return v.item() if isinstance(v, np.generic) else v


def _plain(d: dict) -> dict:
    return {k: _py(v) for k, v in d.items()}


def _context(env: Env, state: State, rnd: int, rule_dec: dict, evidence: dict,
             cand_ft: "float | None") -> DecisionContext:
    """A2's view of the round. promote_allowed is the rule's own promote (SHADOW, pooled test passed, floor met)."""
    counts = env.oracle.audit_counts()
    floor = int(env.policy["min_audit_adults"])
    th = state.live.th
    return DecisionContext(
        round=rnd,
        thresholds={"t_verify": float(th.t_verify), "t_soft": float(th.t_soft), "cap": float(th.cap),
                    "flags": list(th.flags)},
        audit=_plain({"mode": state.mode, "streak": state.streak, "audit_adults": counts["adults"],
                      "audit_teens": counts["teens"], "round_audit_adults": evidence["round_audit_adults"],
                      "pooled_adults": evidence["pooled_adults"], "pooled_false_teen_rate": evidence["pooled_ft"],
                      "candidate_false_teen": cand_ft}),
        bounds={"cap_min": CAP_MIN, "cap_max": CAP_MAX, "min_audit_adults": floor,
                "cap_default": clamp_cap(env.policy["cap_false_teen"])},
        guards={"hold_required": counts["adults"] < floor, "promote_allowed": rule_dec["action"] == PROMOTE},
        rule=dict(rule_dec),
    )


def _a2_block(decide: Decide | None, make_ctx: Callable[[], DecisionContext], blocks: dict) -> dict:
    """The decider's {"a2": block}. The context is built here, only when there is a decider, and inside the guard.
    Never raises (the reveal cannot be undone): a failing context or decider leaves A2 unrun, the error goes in the
    record (agent_error) and the rule decides."""
    if decide is None:
        return {}
    try:
        out = decide(make_ctx(), blocks)
    except Exception as e:  # noqa: BLE001 - the contract is "agents never break the round"
        return {"agent_error": f"A2 {type(e).__name__}: {e}"[:500]}
    return {k: v for k, v in out.items() if k == "a2"}


def _a2_applied(a2_block: "dict | None", rule_dec: dict, floor_met: bool, promote_ok: bool) -> "dict | None":
    """A2's own {action, cap} as the loop will apply it, or None when A2 has no decision of its own (not run,
    FALLBACK, no output). The guards are enforced here again, in code: no refit before the audit floor, no promote
    unless the pooled test passed (then re-tune), and the cap is clamped. The promote gate itself used the policy
    cap. cap_margin is clamped to 0..MARGIN_MAX (and the cap); None keeps the policy's."""
    out = side_by_side.a2_decision(a2_block)
    if out is None or out.get("action") not in (HOLD, RETUNE, PROMOTE) or not isinstance(out.get("cap"), (int, float)):
        return None
    action = out["action"]
    if not floor_met:
        action = HOLD
    elif action == PROMOTE and not promote_ok:
        action = RETUNE
    cap = clamp_cap(float(out["cap"]))
    applied = {"action": action, "cap": cap}
    margin = out.get("cap_margin")
    if isinstance(margin, (int, float)) and not isinstance(margin, bool):
        applied["cap_margin"] = clamp_margin(float(margin), cap)
    window = out.get("refit_window")
    if isinstance(window, int) and not isinstance(window, bool):
        applied["refit_window"] = clamp_window(window)
    return applied


def _agent_blocks(before_decision: BeforeDecision | None, state: State, batch: "Batch | None",
                  prior: pd.DataFrame, psi_val: "float | None") -> dict:
    """The hook's agent blocks. Never raises: the reveal cannot be undone, so an agent must not break the
    round. A failing hook leaves the agents unrun, the error goes in the record (agent_error) and the rule
    decides. Only AGENT_KEYS are taken, so a hook cannot overwrite the rule's decision."""
    if before_decision is None:
        return {}
    try:
        blocks = before_decision(state, batch, prior, psi_val)
    except Exception as e:  # noqa: BLE001 - the contract is "agents never break the round"
        return {"agent_error": f"{type(e).__name__}: {e}"[:500]}
    return {k: v for k, v in blocks.items() if k in AGENT_KEYS}


def round0(env: Env, state: State, before_decision: BeforeDecision | None = None) -> RoundResult:
    """R0: the starter rule on the frozen test set, before any batch or label (before_decision gets batch None)."""
    env.timer.round = 0
    blocks = _agent_blocks(before_decision, state, None, state.seen, None)
    dec = {"cutoff": float(state.live.th.t_verify), "cap": clamp_cap(env.policy["cap_false_teen"]), "action": STARTER}
    record = make_record(env.timer.run, 0, dec, STARTER)
    record.update(blocks)
    return RoundResult(make_row(env, state, 0, STARTER, STARTER), record)


# ---- one round ----
def run_round(state: State, batch: Batch, env: Env, before_decision: BeforeDecision | None = None,
              decide: Decide | None = None, apply_a2: bool = False) -> RoundResult:
    """Run one round and update state in place. See the module docstring for the order."""
    env.timer.round = batch.round
    with env.timer.call(AGENT, RUN_ROUND_STEP, "tool"):
        return _run_round(state, batch, env, before_decision, decide, apply_a2)


def _run_round(state: State, batch: Batch, env: Env, before_decision: BeforeDecision | None = None,
               decide: Decide | None = None, apply_a2: bool = False) -> RoundResult:
    """Run the round; if anything raises, put the state back to how the last finished round left it.

    The oracle cannot undo a reveal and refuses a second one for the same round, so after a failure past
    the reveal the run cannot be resumed: the restored state is only a consistent one to inspect.
    """
    before = replace(state)  # fields are reassigned during a round, never mutated in place
    try:
        return _apply_round(state, batch, env, before_decision, decide, apply_a2)
    except Exception:
        for f in fields(state):
            setattr(state, f.name, getattr(before, f.name))
        raise


def _apply_round(state: State, batch: Batch, env: Env, before_decision: BeforeDecision | None = None,
                 decide: Decide | None = None, apply_a2: bool = False) -> RoundResult:
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
    cand_ft = cand_fts = None
    if state.candidate is not None:
        cand_s, t_cand = state.candidate.score(rows)[a_idx], state.candidate.th.t_verify
        cand_ft = false_teen(y_audit, cand_s, t_cand)
        cand_fts = int(((cand_s >= t_cand) & (y_audit == 0)).sum())
    psi_val = psi(ref_scores, live_s) if len(ref_scores) else None
    # agents that inform the decision (A1 now): mode is still this round's, prior is the earlier batches only
    blocks = _agent_blocks(before_decision, state, batch, prior, psi_val)

    # 4. decision: streak first, then the rule
    round_adults = int((y_audit == 0).sum())
    evidence = {"round_audit_adults": round_adults, "cand_ft": cand_ft, "streak": state.streak,
                "cand_t_verify": None if state.candidate is None else float(state.candidate.th.t_verify),
                "cand_unsafe": state.candidate is not None and _unsafe(state.candidate),
                "cand_age": None if state.candidate_round is None else batch.round - state.candidate_round,
                "pooled_adults": None, "pooled_ft": None, "promote_refused": False}
    challenger = env.policy.get("promote_rule") == CHALLENGER
    wins = False
    if challenger:
        live_rf = audit_rates(y_audit, live_s[a_idx], state.live.th.t_verify)
        cand_rf = (None, None)
        if state.candidate is not None and not evidence["cand_unsafe"]:
            cand_rf = audit_rates(y_audit, cand_s, t_cand)
            wins = challenger_wins(live_rf, cand_rf, clamp_cap(env.policy["cap_false_teen"]), round_adults)
        state.streak = 1 if wins else 0
        state.window = [(round_adults, cand_fts)] if cand_fts is not None else []
        evidence.update(streak=state.streak, pooled_adults=round_adults if cand_fts is not None else None,
                        pooled_ft=cand_ft, live_audit_rec=live_rf[0], live_audit_ft=live_rf[1],
                        cand_audit_rec=cand_rf[0], challenger_wins=wins)
    elif state.mode == SHADOW:
        # a candidate with INSUFFICIENT_* flags holds the prior's cutoffs, which may be on another scale
        usable = cand_ft is not None and not evidence["cand_unsafe"]
        state.window = (state.window + [(round_adults, cand_fts)])[-PROMOTE_STREAK:] if usable else []
        cap = clamp_cap(env.policy["cap_false_teen"])
        adults, fts, passes = pooled_test(state.window, cap)
        if len(state.window) < PROMOTE_STREAK:
            state.streak = len(state.window)
        else:
            state.streak = PROMOTE_STREAK if passes else 0
        evidence.update(streak=state.streak, pooled_adults=adults or None, pooled_ft=fts / adults if adults else None)
    rule_dec = rule_decision(state, env.policy, oracle.audit_counts()["adults"])

    # 4b. A2 (after the streak update, before the refit; mode and live rule are still this round's, pre-apply).
    # Its decision is always logged; it is applied only when apply_a2 (crew mode), and only if it has a decision
    # of its own (not FALLBACK). The promote gate above used the policy cap; the refit on a promote reuses the tested candidate's cap (step 5).
    a2_blocks = _a2_block(decide, lambda: _context(env, state, batch.round, rule_dec, evidence, cand_ft), blocks)
    a2_block = a2_blocks.get("a2")
    floor_met = oracle.audit_counts()["adults"] >= int(env.policy["min_audit_adults"])
    mine = _a2_applied(a2_block, rule_dec, floor_met, rule_dec["action"] == PROMOTE)
    source = "A2" if apply_a2 and mine is not None else SOURCE_RULE
    applied = dict(mine) if source == "A2" else dict(rule_dec)
    rule_orig = dict(rule_dec)

    # 5. apply it
    refit_s = None
    policy_cap = clamp_cap(env.policy["cap_false_teen"])
    policy_margin = float(env.policy["cap_margin"])
    evidence.update(policy_cap=policy_cap, refit_cap=None, cap_differs=False, policy_margin=policy_margin,
                    refit_margin=None, margin_differs=False, refit_window=None, window_differs=False)
    if applied["action"] != HOLD:
        if applied["action"] == PROMOTE:
            # the model that goes live is thresholded at the cap its evidence candidate was refit at (the one the
            # gate measured), not at whatever cap A2 or the rule proposes now; the gate bar itself is the policy cap
            applied = {**applied, "cap": policy_cap if state.candidate_cap is None else state.candidate_cap,
                       "cap_margin": state.candidate_margin,  # margin and window go with the cap: same model as tested
                       "refit_window": state.candidate_window}
        margin = policy_margin if applied.get("cap_margin") is None else applied["cap_margin"]
        window = effective_window(env, state, applied.get("refit_window"))
        applied = {k: v for k, v in applied.items() if k != "refit_window"} | {"cap_margin": margin} | (
            {} if window is None else {"refit_window": window})  # what the refit really uses
        evidence.update(refit_cap=applied["cap"], cap_differs=applied["cap"] != policy_cap,
                        refit_margin=margin, margin_differs=margin != policy_margin,
                        refit_window=window, window_differs=window is not None)
        t0 = time.perf_counter()
        with env.timer.call(AGENT, "refit", "tool"):
            state.candidate = refit(env, state, applied["cap"], margin, window)
            state.candidate_round, state.candidate_cap = batch.round, applied["cap"]
            state.candidate_margin, state.candidate_window = margin, window
        refit_s = time.perf_counter() - t0
        if applied["action"] == PROMOTE:
            if _unsafe(state.candidate):
                applied = {**applied, "action": RETUNE}  # refused: try again next round
                evidence["promote_refused"] = True
            else:
                state.mode = ACTIVE
                state.live_scores = np.empty(0)  # the starter's scores are not on the stack's scale
                state.live = state.candidate
        # pooled: once ACTIVE every refit goes live. challenger: only after a won round (a loss keeps the live rule)
        if state.mode == ACTIVE and (not challenger or (wins and not _unsafe(state.candidate))):
            state.live = state.candidate
    if source == SOURCE_RULE:
        rule_orig = applied  # a refused promote shows as the rule's own decision, as before A2 existed
    # A2's raw proposal against the rule decision in the record; the rule's margin is the policy's
    diff = side_by_side.diff(a2_block, {**rule_orig, "cap_margin": policy_margin})
    decision = _decision_block(applied, state.live)
    rule_block = _decision_block(rule_orig, state.live)
    if (rule_orig["action"], rule_orig["cap"], rule_orig.get("cap_margin", policy_margin)) != (
            applied["action"], applied["cap"], applied.get("cap_margin", policy_margin)):
        # the live cutoff is the applied decision's; the rule's own cutoff is unknown without a second refit
        rule_block["cutoff"] = None
    row = make_row(env, state, batch.round, applied["action"], source, n_flagged=n_flagged,
                   n_verify=len(verify_ids), audit_ft=audit_ft, psi=psi_val, refit_s=refit_s, diff_count=len(diff))
    record = make_record(env.timer.run, batch.round, decision, source, evidence,
                         rule_decision=rule_block)
    record.update(blocks)
    record.update(a2_blocks)
    record["diff"] = diff
    return RoundResult(row, record, shadow_row(env, state, batch.round))


def shadow_row(env: Env, state: State, rnd: int) -> dict | None:
    """The model in training (the newest candidate, refit this round or earlier) on the frozen test set, at its own
    t_verify, or None before the first refit. Report only: it goes to shadow.csv, never to state, a decision, an
    agent or decisions.jsonl (which carries no test metric), so the screen can show the learning each round while
    the starter is still live in SHADOW."""
    if state.candidate is None:
        return None
    return {"run": env.timer.run, "round": rnd, "candidate_round": state.candidate_round,
            "t_verify": float(state.candidate.th.t_verify), **report_metrics(state.candidate, env.test)}


# ---- whole run ----
def run_loop(env: Env, n_rounds: int | None = None, state: State | None = None,
             before_decision: BeforeDecision | None = None,
             on_round: OnRound | None = None, decide: Decide | None = None,
             apply_a2: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """R0 then each batch the oracle has (at most n_rounds). Returns the rounds table and the records.

    Pass a state to read the final live rule and candidate afterwards (it is updated in place).
    before_decision / on_round: see the module docstring (agents and per-round writes).
    """
    state = new_state(env.policy) if state is None else state
    results = [round0(env, state, before_decision)]
    if on_round is not None:
        on_round(results[0])
    for batch in env.oracle:
        if n_rounds is not None and batch.round > n_rounds:
            break
        results.append(run_round(state, batch, env, before_decision, decide, apply_a2))
        if on_round is not None:
            on_round(results[-1])
    return pd.DataFrame([r.row for r in results], columns=ROUNDS_COLS), [r.record for r in results]


_thread_lock = threading.Lock()


@contextlib.contextmanager
def _write_lock(folder: Path):
    """One writer at a time across threads and processes (the app's background run and a CLI run):
    a thread lock plus fcntl.flock on folder/.write.lock (POSIX; on other systems the thread lock only)."""
    with _thread_lock:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / ".write.lock", "a", encoding="utf-8") as lock:
            try:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX)
            except ImportError:
                pass
            yield


def check_rounds_header(rounds_path: Path, columns: list[str]) -> str:
    """The first line of rounds_path ("" when the file is missing or empty). Raises ValueError if it is not the
    header of columns, so a run can fail before it starts, not after its last round (main() calls this first)."""
    first = ""
    if rounds_path.exists():
        with open(rounds_path, encoding="utf-8") as f:
            first = f.readline().strip()  # the header only: the file is never read in full
    if first and first.split(",") != list(columns):
        raise ValueError(f"{rounds_path.name} has header {first.split(',')}, this run has {list(columns)}. It is from "
                         f"an older schema: move or delete {rounds_path} (a new run recreates it) and run again.")
    return first


SHADOW_COLS = ["run", "round", "candidate_round", "t_verify", "prec", "rec", "ft", "mt", "auc"]


def shadow_path(rounds_path: Path) -> Path:
    """Where a rounds file's shadow rows go: rounds.csv -> shadow.csv, rounds_recorded.csv -> shadow_recorded.csv."""
    return rounds_path.with_name(rounds_path.name.replace("rounds", "shadow", 1))


def write_run(rounds: pd.DataFrame, records: list[dict], rounds_path: Path = ROUNDS_CSV,
              decisions_path: Path = DECISIONS_JSONL, shadow: list[dict | None] | None = None) -> None:
    """Append rows to rounds.csv and records to decisions.jsonl (rows carry their run id; header written once).

    Raises ValueError, writing nothing, if rounds.csv already has a different header. Both appends happen
    under one lock (_write_lock), so two writers (the app's background run and a CLI run) never lose a
    round or race on a temp file. Appends are small; a reader that catches one mid-write sees a last line
    with no newline, which the readers skip. A torn last line (a writer killed mid-append) is cut off first,
    never completed into a malformed row. Nothing checks for a run id that is already present.

    shadow: RoundResult.shadow rows (None entries skipped), appended to shadow_path(rounds_path) under the same lock.
    """
    with _write_lock(rounds_path.parent):
        for path in (rounds_path, decisions_path):
            _drop_torn_tail(path)
        first = check_rounds_header(rounds_path, list(rounds.columns))
        buf = io.StringIO()
        rounds.to_csv(buf, header=not first, index=False, lineterminator="\n")
        new_dec = "".join(json.dumps(r) + "\n" for r in records)
        for path, text in ((rounds_path, buf.getvalue()), (decisions_path, new_dec)):
            with open(path, "a", encoding="utf-8", newline="") as f:
                f.write(text)
        rows = [r for r in shadow or [] if r is not None]
        if rows:
            path = shadow_path(rounds_path)
            _drop_torn_tail(path)
            new = not path.exists() or path.stat().st_size == 0
            pd.DataFrame(rows, columns=SHADOW_COLS).to_csv(path, mode="a", header=new, index=False,
                                                           lineterminator="\n")


def _drop_torn_tail(path: Path, chunk: int = 4096) -> None:
    """Cut an unterminated last line back to the last newline (the readers already skip it), reading the
    file backwards from the end, not in full. Ending it with a newline instead would turn the fragment into
    a complete, malformed row that load_rounds rejects, taking the Loop tab down."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        end = f.seek(0, 2)
        f.seek(end - 1)
        if f.read(1) == b"\n":
            return
        pos = end
        while pos > 0:
            pos = max(0, pos - chunk)
            f.seek(pos)
            i = f.read(min(chunk, end - pos)).rfind(b"\n")
            if i != -1:
                f.truncate(pos + i + 1)
                return
        f.truncate(0)  # no complete line at all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=RUN_MODES, default="rule", help="rule: no agents; crew: A1 and A2 applied")
    ap.add_argument("--source", choices=THRESHOLD_SOURCES, default="audit")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--no-write", action="store_true", help="print only, leave rounds.csv and decisions.jsonl alone")
    args = ap.parse_args()
    if not args.no_write:
        check_rounds_header(ROUNDS_CSV, ROUNDS_COLS)  # fail now, not after every refit has run
    train, test = load_data(on_param_mismatch="error")
    env = make_env(train, test, threshold_source=args.source)
    if args.mode == "crew":
        from softsignal import crew  # crew imports this module, so only here
        from softsignal.agents.base import make_client

        rounds, records = crew.run_crew(env, make_client(), args.rounds, write=not args.no_write)
        written = not args.no_write  # run_crew appended each round as it landed
    else:
        shadows: list = []
        rounds, records = run_loop(env, args.rounds, on_round=lambda r: shadows.append(r.shadow))
        written = False
        if not args.no_write:
            write_run(rounds, records, shadow=shadows)
            written = True
    pd.set_option("display.width", 220)
    print(rounds.drop(columns=["run"]).round(3).to_string(index=False))
    if written:
        print(f"appended run {env.timer.run} to {ROUNDS_CSV.name} and {DECISIONS_JSONL.name}")


if __name__ == "__main__":
    main()
