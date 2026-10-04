"""Input builders for the agents: the information barrier lives here (Crew Plan section 7).

Each builder copies only the fields its agent's contract allows (Combined Plan section 7a). A1-A4 never get
a label, a frozen test account or a test-set metric; check_barrier() enforces the key part on every payload.
A2 and A4 are built so far; each owner adds theirs here.

A4 (verify-band triager) gets explain.py's output for the accounts sent to verification (score, band, the
top 3 signed contributions, the teen-leaning words), summarised in code for the batch: score range and
median, each signal's account count, share and mean contribution, how many accounts each signal leads, and
the recurring words. No per-account rows and no account ids: the note is about the batch, an id is not
evidence, and per-account numbers (ranks 1..n, one account's contribution) would let the numbers-in-input
check pass an invented count or one account's value presented as a batch figure. Every number in the
input is a batch-level fact, so every number A4 may cite is one too. It also keeps the prompt short.
"""
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.agents.base import INSUFFICIENT
from softsignal.data import SPLIT_FILE
from softsignal.features import FORBIDDEN, ID_COL

TEST_METRIC_KEYS = frozenset({"prec", "rec", "ft", "mt", "f1", "auc"})  # test-set metrics: never in A1-A4 input
LABEL_KEYS = frozenset(FORBIDDEN | {"label", "labels", "in_verify", "in_audit"})
A4_COLS = [ID_COL, "score", "band", "c1", "c2", "c3", "f1", "f2", "f3", "v1", "v2", "v3", "words"]
A4_TOP_WORDS = 10


class BarrierError(ValueError):
    """An agent input would carry a label, a frozen test account or a test-set metric."""


@lru_cache(maxsize=4)
def _test_ids_from(path: str, mtime_ns: int) -> frozenset:
    return frozenset(json.loads(Path(path).read_text())["test_ids"])


def frozen_test_ids(split_file: Path = SPLIT_FILE) -> frozenset:
    """The 900 held-out test ids from the committed split (ids only, no labels)."""
    return _test_ids_from(str(split_file), split_file.stat().st_mtime_ns)


def check_barrier(payload) -> None:
    """Raise BarrierError if any key anywhere in the payload names a label or a test-set metric."""
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            bad = (LABEL_KEYS | TEST_METRIC_KEYS) & set(node)
            if bad:
                raise BarrierError(f"agent input carries forbidden keys {sorted(bad)}")
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)


def _phrase(chip: str) -> str:
    """The readable part of a chip: "writes like a teen +7.98" -> "writes like a teen"."""
    return chip.rsplit(" ", 1)[0]


def a4_input(frame: pd.DataFrame, verify_ids, round_id: int | None, test_ids=None) -> dict:
    """A4's input for one batch. frame: explain.apply_bands(explain_frame(model, batch rows), ...) rows of the
    batch (other columns, a label included, are never copied). verify_ids: the accounts sent to verification
    (the verify band cut to the review budget). test_ids: the frozen test set (default: results/split.json).

    Raises BarrierError if a verify id is a frozen test account, ValueError if a verify id is missing from the
    frame or not in its verify band. An empty verify_ids gives n_accounts 0 (A4 returns insufficient_data).
    """
    missing = [c for c in A4_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"A4 needs explain.apply_bands output; missing columns {missing}")
    ids = list(dict.fromkeys(str(i) for i in verify_ids))
    test = frozen_test_ids() if test_ids is None else frozenset(str(i) for i in test_ids)
    leaked = sorted(set(ids) & test)
    if leaked:
        raise BarrierError(f"{len(leaked)} verify ids are frozen test accounts (e.g. {leaked[0]}); A4 never sees them")
    rows = frame[A4_COLS].assign(**{ID_COL: frame[ID_COL].astype(str)})
    rows = rows[rows[ID_COL].isin(ids)]
    if rows[ID_COL].duplicated().any() or len(rows) != len(ids):
        raise ValueError("every verify id must appear exactly once in the frame")
    if (rows["band"] != "verify").any():
        raise ValueError("every verify id must be in the verify band")
    groups, leads, words = {}, Counter(), Counter()
    for r in rows.itertuples(index=False):
        top = [(f, _phrase(c), float(v)) for f, c, v in zip((r.f1, r.f2, r.f3), (r.c1, r.c2, r.c3), (r.v1, r.v2, r.v3))
               if isinstance(f, str) and f and isinstance(c, str) and c]
        for f, p, v in top:
            groups.setdefault((f, p), []).append(v)
        if top:  # c1 is the account's largest contribution by size
            leads[top[0][:2]] += 1
        words.update({w for w in str(r.words).split(", ") if w} if isinstance(r.words, str) else set())
    n = len(rows)
    signals = [{"feature": f, "signal": p, "n_accounts": len(v), "share_pct": round(100 * len(v) / n),
                "mean_contribution": round(float(np.mean(v)), 2), "n_leading": leads[(f, p)]}
               for (f, p), v in groups.items()]
    signals.sort(key=lambda s: (-s["n_accounts"], -abs(s["mean_contribution"]), s["feature"], s["signal"]))
    score = rows["score"].astype(float)
    payload = {
        "agent": "A4",
        "round": None if round_id is None else int(round_id),
        "band": "verify",
        "n_accounts": n,
        "score_min": round(float(score.min()), 2) if n else None,
        "score_median": round(float(score.median()), 2) if n else None,
        "score_max": round(float(score.max()), 2) if n else None,
        "signals": signals,
        "top_words": [{"word": w, "n_accounts": k}
                      for w, k in sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))[:A4_TOP_WORDS]],
    }
    check_barrier(payload)
    return payload


INSUFFICIENT_INPUT = INSUFFICIENT  # what an absent A1 / A3 output looks like to A2 (contract: no guessing)
A2_ACTIONS = ("hold", "re-tune", "promote")
A2_THRESHOLD_KEYS = ("t_verify", "t_soft", "cap", "flags")
A2_AUDIT_KEYS = ("mode", "streak", "audit_adults", "audit_teens", "round_audit_adults", "pooled_adults",
                 "pooled_false_teen_rate", "candidate_false_teen")
A2_BOUND_KEYS = ("cap_min", "cap_max", "min_audit_adults", "cap_default")
A2_GUARD_KEYS = ("hold_required", "promote_allowed")


@lru_cache(maxsize=4)
def _limits_from(path: str, mtime_ns: int) -> tuple[float, float, int]:
    from softsignal.loop import CAP_MAX, CAP_MIN
    from softsignal.policy import load_policy

    return CAP_MIN, CAP_MAX, int(load_policy(Path(path))["min_audit_adults"])


def cap_limits() -> tuple[float, float]:
    """(cap_min, cap_max): loop.CAP_MIN / CAP_MAX, the clamp no payload may loosen. Imported lazily because
    loop.py will import the agents. Reads no file."""
    from softsignal.loop import CAP_MAX, CAP_MIN

    return CAP_MIN, CAP_MAX


def hard_limits(policy: dict | None = None) -> tuple[float, float, int]:
    """(cap_min, cap_max, min_audit_adults). The audit floor is the policy the loop runs with when it is passed
    (loop.rule_decision reads env.policy), else policy.yaml, read once per modification time."""
    lo, hi = cap_limits()
    if policy is not None:
        return lo, hi, int(policy["min_audit_adults"])
    from softsignal.policy import POLICY_FILE

    return _limits_from(str(POLICY_FILE), POLICY_FILE.stat().st_mtime_ns)


ROUND_DIGITS = 3  # rates and thresholds are rounded for the prompt: a model cannot copy 0.1333333333333333 verbatim


def _round(value):
    """A float rounded to ROUND_DIGITS (None and non-numbers unchanged), so the numbers-in-input check can pass."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(float(value), ROUND_DIGITS) if isinstance(value, float) else value


def _pick(src: dict, keys, where: str) -> dict:
    missing = [k for k in keys if k not in src]
    if missing:
        raise ValueError(f"A2 input {where} is missing {missing}")
    return {k: src[k] for k in keys}


def dotted_paths(payload, prefix: str = "") -> set:
    """Every dotted path to a dict node or leaf in the payload ("audit", "audit.audit_adults"): what A2 may cite."""
    out = set()
    if isinstance(payload, dict):
        for k, v in payload.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.add(path)
            out |= dotted_paths(v, path)
    return out


def a2_input(round_id: int, thresholds: dict, audit: dict, bounds: dict, guards: dict, rule: dict,
             a1=INSUFFICIENT_INPUT, a3=INSUFFICIENT_INPUT, policy: dict | None = None) -> dict:
    """A2's input for one round (Combined Plan section 7a): A1 and A3 output, current thresholds, the last
    audit-slice metrics, the rule-based decision. Only the listed keys are copied, so a test metric, a label or
    any other field the caller holds never reaches the prompt; check_barrier() then checks the whole payload.

    thresholds: the live rule's (t_verify, t_soft, cap, flags). audit: audit-slice counts and the candidate's
    audit false-teen rate (candidate_false_teen, never a test-set figure). bounds: the clamp and the hold floor.
    guards: hold_required (fewer than min_audit_adults audit adults) and promote_allowed (SHADOW and the pooled
    test passed), both computed by loop.py in code; the agent is told them and the checks enforce them. They are
    cross-checked here against the counts and the mode, and so is the rule, so a caller that builds them wrongly
    fails loudly. bounds may not be looser than hard_limits(policy): the loop's cap clamp and the audit floor of
    policy (pass env.policy; None means policy.yaml). t_verify, t_soft and the two false-teen rates are rounded
    to ROUND_DIGITS decimals, so the model can copy them and the numbers-in-input check can accept them.
    rule: {action, cap}, loop.rule_decision(). a1 / a3: the agent's output dict, or "insufficient_data" (round 0,
    or an agent not built yet); never a guess.
    """
    rule = _pick(rule, ("action", "cap"), "rule")
    if rule["action"] not in A2_ACTIONS:
        raise ValueError(f"rule action must be one of {A2_ACTIONS}, got {rule['action']!r}")
    bounds = _pick(bounds, A2_BOUND_KEYS, "bounds")
    guards = _pick(guards, A2_GUARD_KEYS, "guards")
    audit = _pick(audit, A2_AUDIT_KEYS, "audit")
    if not 0 <= bounds["cap_min"] <= bounds["cap_max"] <= 1:
        raise ValueError(f"A2 bounds need 0 <= cap_min <= cap_max <= 1, got {bounds['cap_min']}, {bounds['cap_max']}")
    lo, hi, floor = hard_limits(policy)
    if bounds["cap_min"] < lo or bounds["cap_max"] > hi or bounds["min_audit_adults"] < floor:
        raise ValueError(f"A2 bounds are looser than the code limits: cap {lo}..{hi} and {floor} audit adults "
                         f"(got {bounds['cap_min']}..{bounds['cap_max']} and {bounds['min_audit_adults']})")
    if audit["mode"] not in ("SHADOW", "ACTIVE"):
        raise ValueError(f"audit mode must be SHADOW or ACTIVE, got {audit['mode']!r}")
    if not bounds["cap_min"] <= rule["cap"] <= bounds["cap_max"]:
        raise ValueError(f"rule cap {rule['cap']} is outside the bounds {bounds['cap_min']}..{bounds['cap_max']}")
    if guards["hold_required"] != (audit["audit_adults"] < bounds["min_audit_adults"]):
        raise ValueError("guards.hold_required contradicts audit_adults and bounds.min_audit_adults")
    if guards["promote_allowed"] and (guards["hold_required"] or audit["mode"] != "SHADOW"):
        raise ValueError("guards.promote_allowed needs mode SHADOW and the hold rule satisfied")
    if guards["hold_required"] and rule["action"] != "hold":
        raise ValueError(f"rule action {rule['action']} breaks guards.hold_required")
    if rule["action"] == "promote" and not guards["promote_allowed"]:
        raise ValueError("rule action promote breaks guards.promote_allowed")
    payload = {
        "agent": "A2",
        "round": int(round_id),
        "a1": a1,
        "a3": a3,
        "thresholds": {k: _round(v) if k in ("t_verify", "t_soft") else v
                       for k, v in _pick(thresholds, A2_THRESHOLD_KEYS, "thresholds").items()},
        "audit": {k: _round(v) if k in ("pooled_false_teen_rate", "candidate_false_teen") else v
                  for k, v in audit.items()},
        "bounds": bounds,
        "guards": guards,
        "rule": rule,
    }
    check_barrier(payload)
    json.dumps(payload)  # must be plain JSON: fail here, not inside the prompt
    return payload
