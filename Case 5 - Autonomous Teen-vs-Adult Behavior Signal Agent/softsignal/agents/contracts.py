"""Input builders for the agents: the information barrier lives here (Crew Plan section 7).

Each builder copies only the fields its agent's contract allows (Combined Plan section 7a). A1-A4 never get
a label, a frozen test account or a test-set metric; check_barrier() enforces the key part on every payload.
All five builders are here.

A5 (honesty auditor) is the one agent that reads test-set metrics: claims plus the results files as rows
(a5_sources), and the section 8 risk list (load_checklist). It runs after the round, and nothing it writes
reaches A1-A4: the other builders copy fixed keys and never read decisions.jsonl or rounds.csv.

A1 (drift watcher) gets, for this batch: the PSI of each feature group against the batches the loop saw in
earlier rounds (never the whole train set: later batches are not known yet), the loop's PSI of the live
scores (only while the live rule is the starter: once the stack is live every refit changes the model, so
score PSI would measure model change, not drift), the revealed audit counts (adults, teens) so far, the
earlier rounds' PSI, the policy.yaml threshold its fallback uses, and the PSI conventions and group sizes
its prompt teaches (so it may quote them). No rows, no ids, no labels: counts only. Pooling every earlier
batch as the reference absorbs slow drift; the history lets A1 see a trend, and a fixed reference (the first
batches) would be the change to make if cumulative drift ever matters.

A4 (verify-band triager) gets explain.py's output for the accounts sent to verification (score, band, the
top 3 signed contributions, the teen-leaning words), summarised in code for the batch: score range and
median, each signal's account count, share and mean contribution, how many accounts each signal leads, and
the recurring words. No per-account rows and no account ids: the note is about the batch, an id is not
evidence, and per-account numbers (ranks 1..n, one account's contribution) would let the numbers-in-input
check pass an invented count or one account's value presented as a batch figure. Every number in the
input is a batch-level fact, so every number A4 may cite is one too. It also keeps the prompt short.

A3 (error analyst) gets the same kind of summary for the audit-slice accounts the live model got wrong in
earlier rounds: false teens (adults at or above the live t_verify) and missed teens (teens below it), each
with its count, rate, the signals and words they share. The label only decides the error type in code; the
payload holds counts and that derived type, never a label or an id. Audit slice only: verify-band labels are
selected for high scores, so they would show almost no missed teens.
"""
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.agents.base import INSUFFICIENT
from softsignal.data import SPLIT_FILE
from softsignal.features import ACTIVITY_COLS, FORBIDDEN, ID_COL, TEXT_COLS
from softsignal.metrics import psi

TEST_METRIC_KEYS = frozenset({"prec", "rec", "ft", "mt", "f1", "auc"})  # test-set metrics: never in A1-A4 input
LABEL_KEYS = frozenset(FORBIDDEN | {"label", "labels", "in_verify", "in_audit"})
A4_COLS = [ID_COL, "score", "band", "c1", "c2", "c3", "f1", "f2", "f3", "v1", "v2", "v3", "words"]
A4_TOP_WORDS = 10
PSI_GROUPS = {"activity": ACTIVITY_COLS, "text": TEXT_COLS}  # the 9 activity and 7 stylometry columns
PSI_DECIMALS = 3
PSI_CONVENTIONS = {"stable": 0.10, "large": 0.25}  # under stable: no shift; 0.10-0.25 moderate; over large: large


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


def floor_psi(x) -> float | None:
    """PSI floored to PSI_DECIMALS (None stays None). Floored, not rounded: then comparing the shown value
    with a threshold of up to 3 decimals gives the same verdict as the exact PSI (0.2496 never reads 0.25)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    scale = 10 ** PSI_DECIMALS
    return math.floor(float(x) * scale + 1e-9) / scale  # 1e-9: float noise must not drop an exact 0.25 to 0.249


def group_psi(reference: pd.DataFrame, batch: pd.DataFrame) -> dict:
    """PSI of each feature group, batch against reference: per group the largest feature PSI, the mean, and
    which feature is largest. metrics.psi per column (bins from the reference's quantiles), floored."""
    out = {}
    for group, cols in PSI_GROUPS.items():
        per = {c: psi(reference[c].to_numpy(dtype=float), batch[c].to_numpy(dtype=float)) for c in cols}
        top = max(per, key=lambda c: (per[c], c))
        out[f"{group}_max"] = floor_psi(per[top])
        out[f"{group}_mean"] = floor_psi(float(np.mean(list(per.values()))))
        out[f"{group}_top_feature"] = top
    return out


def a1_input(reference: pd.DataFrame, batch: pd.DataFrame, score_psi, audit_counts: dict, history: list[dict],
             round_id: int, psi_drift: float | None) -> dict:
    """A1's input for one round. reference: the rows of every earlier batch (loop state.seen before this
    round); batch: this round's rows; score_psi: the loop's PSI of the live scores (rounds.csv psi, None
    when it has no history, e.g. round 1 or the round after a promote); audit_counts: oracle.audit_counts()
    after this round's reveal; history: earlier rounds' entries of this payload's "psi" (see a1_history);
    psi_drift: policy.yaml's threshold. With no reference rows the PSI is not computed (n_reference 0: A1
    returns insufficient_data without a model call).
    """
    n_ref, n_batch = len(reference), len(batch)
    psi_block = {"score": floor_psi(score_psi)}
    if n_ref and n_batch:
        psi_block |= group_psi(reference, batch)
    payload = {
        "agent": "A1",
        "round": int(round_id),
        "n_reference": int(n_ref),
        "n_batch": int(n_batch),
        "psi": psi_block,
        "psi_drift": None if psi_drift is None else float(psi_drift),
        "psi_conventions": dict(PSI_CONVENTIONS),
        "n_features": {g: len(cols) for g, cols in PSI_GROUPS.items()},
        "audit": {"adults": int(audit_counts["adults"]), "teens": int(audit_counts["teens"])},
        "history": [dict(h) for h in history],
    }
    check_barrier(payload)
    return payload


def a1_history(payload: dict) -> dict:
    """The compact entry a round adds to the next rounds' history: round and the PSI numbers."""
    p = payload["psi"]
    return {"round": payload["round"], **{k: p.get(k) for k in ("score", "activity_max", "text_max")}}


def a1_fields(payload: dict) -> dict:
    """Every numeric input field A1 may cite, by path: "psi.activity_max", "audit.adults",
    "history.round3.score", ... (None values are left out: there is nothing to cite)."""
    out = {k: payload[k] for k in ("n_reference", "n_batch", "psi_drift")}
    out |= {f"psi.{k}": v for k, v in payload["psi"].items() if not isinstance(v, str)}
    out |= {f"psi_conventions.{k}": v for k, v in payload["psi_conventions"].items()}
    out |= {f"n_features.{k}": v for k, v in payload["n_features"].items()}
    out |= {f"audit.{k}": v for k, v in payload["audit"].items()}
    for h in payload["history"]:
        out |= {f"history.round{h['round']}.{k}": v for k, v in h.items() if k != "round"}
    return {k: float(v) for k, v in out.items() if v is not None}


A3_COLS = [ID_COL, "score", "c1", "c2", "c3", "f1", "f2", "f3", "v1", "v2", "v3", "words"]
A3_ERROR_TYPES = ("false_teen", "missed_teen")
A3_TOP_WORDS = 10
A3_MAX_SIGNALS = 6  # per error type: the most common first, so the prompt stays short


def _slug(phrase: str) -> str:
    return "_".join("".join(ch if ch.isalnum() else " " for ch in phrase.lower()).split())


def a3_input(frame: pd.DataFrame, labels: dict, t_verify: float | None, round_id: int, min_errors: int | None,
             test_ids=None) -> dict:
    """A3's input for one round: the audit-slice errors of EARLIER rounds, summarised in code.

    frame: explain.explain_frame rows (A3_COLS, scored by the live model) for the audit accounts revealed
    before this round. labels: {account id: 0 / 1} for exactly those accounts (oracle.revealed("audit",
    before_round=round_id)); a label is used here to say which side of the live t_verify the account's error
    is on, and is never copied: the payload carries counts and the derived error type only. An error is a
    false teen (adult scored at or above t_verify) or a missed teen (teen scored below t_verify).
    min_errors: the policy floor on false + missed teens (None: no floor set, A3 returns insufficient_data).

    Like A4's, the payload is batch-level: no ids and no per-account rows, so every number A3 may cite is a
    group fact. Raises BarrierError for a frozen test account, ValueError for a frame or labels mismatch.
    """
    missing = [c for c in A3_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"A3 needs explain.explain_frame output; missing columns {missing}")
    ids = frame[ID_COL].astype(str)
    if ids.duplicated().any() or set(ids) != {str(k) for k in labels}:
        raise ValueError("labels must cover exactly the accounts in the frame, once each")
    bad = {v for v in labels.values() if v not in (0, 1)}
    if bad:
        raise ValueError(f"labels must be 0 or 1, got {sorted(bad, key=str)}")
    test = frozen_test_ids() if test_ids is None else frozenset(str(i) for i in test_ids)
    leaked = sorted(set(ids) & test)
    if leaked:
        raise BarrierError(f"{len(leaked)} audit ids are frozen test accounts (e.g. {leaked[0]}); A3 never sees them")
    if t_verify is not None and math.isnan(float(t_verify)):
        raise ValueError("t_verify is NaN")
    y = ids.map({str(k): int(v) for k, v in labels.items()}).to_numpy()
    n_adults, n_teens = int((y == 0).sum()), int((y == 1).sum())
    score = frame["score"].astype(float).to_numpy()
    if t_verify is None:  # no live threshold, so nothing is an error
        masks = {k: np.zeros(len(frame), dtype=bool) for k in A3_ERROR_TYPES}
    else:
        flagged = score >= float(t_verify)
        masks = {"false_teen": (y == 0) & flagged, "missed_teen": (y == 1) & ~flagged}
    payload = {
        "agent": "A3",
        "round": int(round_id),
        "t_verify": None if t_verify is None else _round(float(t_verify)),
        "min_errors": None if min_errors is None else int(min_errors),
        "audit": {"adults": n_adults, "teens": n_teens},
        "n_errors": {k: int(m.sum()) for k, m in masks.items()},
    }
    for kind, base_n in zip(A3_ERROR_TYPES, (n_adults, n_teens)):
        rows = frame[masks[kind]]
        groups, leads, words = {}, Counter(), Counter()
        for r in rows.itertuples(index=False):
            top = [(f, _phrase(c), float(v)) for f, c, v in zip((r.f1, r.f2, r.f3), (r.c1, r.c2, r.c3),
                                                                (r.v1, r.v2, r.v3))
                   if isinstance(f, str) and f and isinstance(c, str) and c]
            for f, p, v in top:
                groups.setdefault((f, p), []).append(v)
            if top:
                leads[top[0][:2]] += 1
            words.update({w for w in str(r.words).split(", ") if w} if isinstance(r.words, str) else set())
        n = len(rows)
        signals = [{"id": f"{f}__{_slug(p)}", "feature": f, "signal": p, "n_accounts": len(v),
                    "share_pct": round(100 * len(v) / n), "mean_contribution": round(float(np.mean(v)), 2),
                    "n_leading": leads[(f, p)]} for (f, p), v in groups.items()]
        signals.sort(key=lambda s: (-s["n_accounts"], -abs(s["mean_contribution"]), s["id"]))
        payload[kind] = {
            "n_accounts": n,
            "rate": _round(n / base_n) if base_n else None,  # false teens per audit adult, missed teens per audit teen
            "score_median": round(float(np.median(score[masks[kind]])), 2) if n else None,
            "signals": signals[:A3_MAX_SIGNALS],
            "top_words": [{"word": w, "n_accounts": k}
                          for w, k in sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))[:A3_TOP_WORDS]],
        }
    check_barrier(payload)
    return payload


def a3_fields(payload: dict) -> dict:
    """Every numeric input field A3 may cite as evidence, by path: "false_teen.n_accounts", "false_teen.rate",
    "false_teen.signals.<id>.n_accounts", "audit.adults", "n_errors.missed_teen", ... (None left out)."""
    out = {"t_verify": payload["t_verify"], "min_errors": payload["min_errors"]}
    out |= {f"audit.{k}": v for k, v in payload["audit"].items()}
    out |= {f"n_errors.{k}": v for k, v in payload["n_errors"].items()}
    for kind in A3_ERROR_TYPES:
        block = payload[kind]
        out |= {f"{kind}.{k}": block[k] for k in ("n_accounts", "rate", "score_median")}
        for s in block["signals"]:
            out |= {f"{kind}.signals.{s['id']}.{k}": s[k]
                    for k in ("n_accounts", "share_pct", "mean_contribution", "n_leading")}
        for w in block["top_words"]:
            out[f"{kind}.top_words.{w['word']}"] = w["n_accounts"]
    return {k: float(v) for k, v in out.items() if v is not None}


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


def window_min() -> int:
    """loop.WINDOW_MIN: the shortest refit window A2 may ask for. Lazy import, as cap_limits."""
    from softsignal.loop import WINDOW_MIN

    return WINDOW_MIN


def margin_limit() -> float:
    """loop.MARGIN_MAX: the most a cap_margin may be, whatever A2 proposes. Lazy import, as cap_limits."""
    from softsignal.loop import MARGIN_MAX

    return MARGIN_MAX


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


def _policy_margin(policy: dict | None) -> float:
    """The cap_margin the rule refits with: the loop's policy when passed, else policy.yaml."""
    if policy is None:
        from softsignal.policy import load_policy

        policy = load_policy()
    return float(policy.get("cap_margin", 0.0))


def _pick(src: dict, keys, where: str) -> dict:
    missing = [k for k in keys if k not in src]
    if missing:
        raise ValueError(f"A2 input {where} is missing {missing}")
    return {k: src[k] for k in keys}


def dotted_paths(payload, prefix: str = "") -> set:
    """Every dotted path to a dict node or leaf in the payload ("audit", "audit.audit_adults"), list items by index
    ("a1.evidence[0]", "a1.evidence[0].value"): what A2 may cite."""
    out = set()
    if isinstance(payload, dict):
        for k, v in payload.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.add(path)
            out |= dotted_paths(v, path)
    elif isinstance(payload, list):
        for i, v in enumerate(payload):
            path = f"{prefix}[{i}]"
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

    Added in code, so A2 never has to compute or guess them: audit.audit_total (adults + teens, so a reason can
    state the total without adding), bounds.margin_max and bounds.window_min (the code limits on its two levers,
    loop.MARGIN_MAX / WINDOW_MIN) and rule.cap_margin (the policy margin the rule refits with, what a null keeps).
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
        "audit": {**{k: _round(v) if k in ("pooled_false_teen_rate", "candidate_false_teen") else v
                     for k, v in audit.items()},
                  "audit_total": audit["audit_adults"] + audit["audit_teens"]},
        "bounds": {**bounds, "margin_max": margin_limit(), "window_min": window_min()},
        "guards": guards,
        "rule": {**rule, "cap_margin": _policy_margin(policy)},
    }
    check_barrier(payload)
    json.dumps(payload)  # must be plain JSON: fail here, not inside the prompt
    return payload


# ---- A5: claims, results files as rows, the risk checklist ----
A5_DECIMALS = 4  # source values are rounded for the prompt; a claim states at most this precision
# Columns left out of A5's rounds rows: the run id is in no claim, and refit_s is wall-clock time; both differ on
# every run, so leaving them in made A5's input hash (and so its replay) change from run to run.
A5_ROUND_DROP = ("run", "refit_s")
CLAIMS_DIR = Path(__file__).resolve().parents[2] / "claims"
CLAIMS_FILE, CHECKLIST_FILE = CLAIMS_DIR / "claims.md", CLAIMS_DIR / "risks.yaml"
MEASURED, PROJECTED = "measured", "projected"


def load_claims(path: Path = CLAIMS_FILE) -> list[str]:
    """The claims of a markdown file: every line that starts with "- " (the rest is instructions)."""
    return [ln[2:].strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.startswith("- ") and ln[2:].strip()]


def load_checklist(path: Path = CHECKLIST_FILE) -> list[dict]:
    """The section 8 risk list: [{id, risk, say, watch: [[term, ...], ...]}]. Raises ValueError if malformed."""
    import yaml

    items = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ok = isinstance(items, list) and all(
        isinstance(i, dict) and {"id", "risk", "say", "watch"} <= set(i)
        and all(isinstance(g, list) and g and all(isinstance(w, str) for w in g) for g in i["watch"])
        for i in items)
    if not ok or len({i["id"] for i in items}) != len(items):
        raise ValueError(f"{path} must be a list of unique {{id, risk, say, watch: [[term, ...]]}} items")
    return items


def _value(v):
    """A CSV cell for the prompt: numbers rounded to A5_DECIMALS, blanks dropped by the caller."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else round(float(v), A5_DECIMALS)
    return str(v)


def _csv_rows(df: pd.DataFrame, file: str, row_id, kind) -> list[dict]:
    rows = []
    for i, r in enumerate(df.to_dict("records"), start=1):
        values = {k: _value(v) for k, v in r.items()}
        rows.append({"id": f"{file}:{row_id(i, r)}", "kind": kind(r),
                     "values": {k: v for k, v in values.items() if v is not None and v != ""}})
    return rows


def a5_sources(results_dir: Path, rounds: pd.DataFrame | None = None, rounds_file: str = "rounds_recorded.csv",
               files: tuple = ("eval", "rounds", "policy_grid", "sanity", "ablations")) -> list[dict]:
    """The results files A5 checks claims against, as [{file, status, rows: [{id, kind, values}]}].

    eval: eval.csv (rows with eval_set "projected" are projected), else eval_placeholder.csv (all projected), and
    eval.csv is then also listed as missing.
    rounds: the given rows (a run in progress: its rows so far, named rounds_file), else rounds_file's latest run;
    without the A5_ROUND_DROP columns.
    policy_grid: policy_grid.csv (what the Results tab shows; an addition to the plan's four files).
    sanity: sanity.txt, one row per line with a number. ablations: ablations.csv. A missing file is listed with
    status "missing", so A5 can say cannot_check instead of guessing.
    """
    out = []

    def missing(name: str) -> None:
        out.append({"file": name, "status": "missing", "rows": []})

    for f in files:
        if f == "eval":
            real, ph = results_dir / "eval.csv", results_dir / "eval_placeholder.csv"
            path = real if real.exists() else ph if ph.exists() else None
            if path != real:
                missing("eval.csv")  # the measured ladder is not written yet: say so, even with the placeholder
            if path is None:
                continue
            df = pd.read_csv(path)
            proj = path == ph
            out.append({"file": path.name, "status": "present", "rows": _csv_rows(
                df, path.name, lambda i, r: i,
                lambda r: PROJECTED if proj or str(r.get("eval_set")) == "projected" else MEASURED)})
        elif f == "rounds":
            if rounds is None:
                path = results_dir / rounds_file
                if not path.exists():
                    missing(rounds_file)
                    continue
                df = pd.read_csv(path, dtype={"run": str})
                df = df[df["run"] == df["run"].max()]  # run ids are UTC timestamps: the latest recorded run
            else:
                df = rounds
            name = rounds_file
            df = df.drop(columns=[c for c in A5_ROUND_DROP if c in df.columns])
            out.append({"file": name, "status": "present", "rows": _csv_rows(
                df, name, lambda i, r: f"R{int(r['round'])}", lambda r: MEASURED)})
        elif f == "policy_grid":
            path = results_dir / "policy_grid.csv"
            if not path.exists():
                missing(path.name)
                continue
            out.append({"file": path.name, "status": "present", "rows": _csv_rows(
                pd.read_csv(path), path.name, lambda i, r: f"cap={r['cap']:g}", lambda r: MEASURED)})
        elif f == "sanity":
            path = results_dir / "sanity.txt"
            if not path.exists():
                missing(path.name)
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            out.append({"file": path.name, "status": "present", "rows": [
                {"id": f"{path.name}:line {n}", "kind": MEASURED, "values": {"line": ln.strip()}}
                for n, ln in enumerate(lines, start=1) if any(ch.isdigit() for ch in ln)]})
        elif f == "ablations":
            path = results_dir / "ablations.csv"
            if not path.exists():
                missing(path.name)
                continue
            out.append({"file": path.name, "status": "present", "rows": _csv_rows(
                pd.read_csv(path), path.name, lambda i, r: i, lambda r: MEASURED)})
        else:
            raise ValueError(f"unknown A5 source {f!r}")
    return out


def round_claims(row: dict, record: dict) -> list[str]:
    """What the screen shows for one finished round, as claims for A5's per-round check: the round's headline
    numbers (as the Loop tab renders them, from rounds.csv) and A2's reason when A2 decided."""
    claims = [f"Round {int(row['round'])}: {row['mode']}, {row['action']}; test recall {row['rec']:.1%} at "
              f"{row['ft']:.1%} false-teen; {int(row['n_audit_adults'])} audit adults so far."]
    a2 = (record.get("a2") or {}).get("output")
    if isinstance(a2, dict) and isinstance(a2.get("reason"), str) and a2["reason"]:
        claims.append(f"A2: {a2['reason']}")
    return claims


EVIDENCE_ROW_COLS = ("n_audit_adults", "n_labels", "n_flagged", "n_verify", "cap", "t_verify", "t_soft", "audit_ft")


def evidence_source(record: dict, policy: dict, round_id: int, row: dict | None = None,
                    audit: dict | None = None) -> dict:
    """The round's decision inputs as one A5 source row, for A2's reason: the promote evidence, the hold floor,
    the cap bounds and lever limits, the rule's decision, the round's own loop state (row: its rounds.csv row; only
    EVIDENCE_ROW_COLS, never a test metric), the audit counts A2 saw (audit: oracle.audit_counts(), adults and
    teens, plus their total) and the values A1 cited (its evidence, psi.score as psi_score). Everything A2's input
    carries, so the script does not flag a reason as unsupported only because it could not see a number. Not
    round-keyed, so a reason need not say "round N"."""
    from softsignal.loop import CAP_MAX, CAP_MIN, MARGIN_MAX, WINDOW_MIN

    ev = record.get("evidence") or {}
    values = {k: _value(v) for k, v in ev.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    values |= {k: _value((row or {}).get(k)) for k in EVIDENCE_ROW_COLS
               if isinstance((row or {}).get(k), (int, float)) and not isinstance((row or {}).get(k), bool)}
    values |= {"min_audit_adults": int(policy["min_audit_adults"]), "cap_min": CAP_MIN, "cap_max": CAP_MAX,
               "margin_max": MARGIN_MAX, "window_min": WINDOW_MIN,
               "rule_cap_margin": _value(float(policy.get("cap_margin", 0.0)))}
    if audit:
        values |= {"audit_adults": int(audit["adults"]), "audit_teens": int(audit["teens"]),
                   "audit_total": int(audit["adults"]) + int(audit["teens"])}
    a1 = (record.get("a1") or {}).get("output")
    for ev in (a1.get("evidence") or []) if isinstance(a1, dict) else []:
        if isinstance(ev, dict) and isinstance(ev.get("value"), (int, float)) and not isinstance(ev.get("value"), bool):
            values[str(ev.get("field", "")).replace(".", "_")] = _value(ev["value"])
    rule = record.get("rule_decision") or {}
    values |= {f"rule_{k}": _value(v) for k, v in rule.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {"file": "decisions.jsonl", "status": "present",
            "rows": [{"id": f"decisions.jsonl:R{int(round_id)}", "kind": MEASURED,
                      "values": {k: v for k, v in values.items() if v is not None}}]}


def a5_input(claims: list[str], sources: list[dict], checklist: list[dict], mode: str,
             round_id: int | None = None) -> dict:
    """A5's input: the claims (with ids c1, c2, ...), the results files as rows, and the risk checklist.

    mode is "round" (after a round: its numbers and A2's reason) or "slides" (the slide pass). A5 may read
    test metrics; check_barrier still runs for labels (no per-account label is ever in a results file).
    """
    if mode not in ("round", "slides"):
        raise ValueError(f"mode must be round or slides, got {mode!r}")
    from softsignal.agents.schemas import A5_MAX_CLAIMS

    if len(claims) > A5_MAX_CLAIMS:  # the output schema allows this many verdicts: say so here, not as invalid_output
        raise ValueError(f"A5 takes at most {A5_MAX_CLAIMS} claims per call, got {len(claims)}: "
                         "split them into batches")
    payload = {
        "agent": "A5",
        "mode": mode,
        "round": None if round_id is None else int(round_id),
        "claims": [{"id": f"c{i}", "text": t} for i, t in enumerate(claims, start=1)],
        "sources": sources,
        "checklist": [{"id": i["id"], "risk": i["risk"], "say": i["say"], "watch": i["watch"]} for i in checklist],
    }
    _check_labels(payload)
    json.dumps(payload)
    return payload


def _check_labels(payload) -> None:
    """check_barrier for labels only: A5 is allowed test metrics (prec, rec, ft, ...), never a per-account label."""
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            bad = LABEL_KEYS & set(node)
            if bad:
                raise BarrierError(f"A5 input carries label keys {sorted(bad)}")
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
