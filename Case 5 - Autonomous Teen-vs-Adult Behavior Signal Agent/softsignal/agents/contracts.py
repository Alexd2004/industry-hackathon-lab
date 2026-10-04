"""Input builders for the agents: the information barrier lives here (Crew Plan section 7).

Each builder copies only the fields its agent's contract allows (Combined Plan section 7a). A1-A4 never get
a label, a frozen test account or a test-set metric; check_barrier() enforces the key part on every payload.
A1 and A4 are built; each owner adds theirs here.

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
"""
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

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
