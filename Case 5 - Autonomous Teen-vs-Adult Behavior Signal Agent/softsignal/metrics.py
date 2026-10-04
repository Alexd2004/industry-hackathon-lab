"""Shared metrics, cutoffs and eval.csv rows (Tier 1, steps 2, 3, 6). Every ladder stage is scored here."""
import numpy as np
from sklearn.metrics import roc_auc_score

from softsignal.features import ID_COL

# Frozen eval.csv columns (Combined Plan section 4).
EVAL_COLS = ["stage", "eval_set", "prec", "rec", "ft", "mt", "f1", "auc"]
# Frozen rounds.csv columns (loop scoreboard, one row per (run, round); prec..auc on the frozen test set).
# run: the AgentTimer run id (UTC timestamp), so runs sort newest first and join agent_calls.jsonl.
# mode: SHADOW / ACTIVE after this round's decision; applied_source: A2 / rule / starter.
# n_flagged: batch accounts at or above t_verify; n_verify: those sent to verification after the
# 25% budget cut (n_flagged > n_verify means "budget wins"). n_audit_adults is cumulative.
# audit_ft: false-teen on this round's audit slice, scored BEFORE the refit (never in-sample).
# t_soft, audit_ft, psi, refit_s may be blank.
ROUNDS_COLS = ["run", "round", "mode", "action", "applied_source", "cap", "t_soft", "t_verify", "n_flagged",
               "n_verify", "n_labels", "n_audit_adults", "audit_ft", "psi", "prec", "rec", "ft", "mt", "auc",
               "refit_s"]
DEFAULT_CAP = 0.15  # max false-teen rate when picking a cutoff
# Frozen ranked.csv columns (explain.py, the "likely teen" list; no labels). c1..c3 are readable
# chips like "quiet in school hours +0.82"; f1..f3 their level-2 feature keys and v1..v3 the signed
# logit contributions as numbers (for A3/A4). words: the account's teen-leaning words, "im, lol".
RANKED_COLS = ["rank", ID_COL, "score", "band", "action", "c1", "c2", "c3", "f1", "f2", "f3",
               "v1", "v2", "v3", "words", "reason"]
# Frozen contrib.csv columns (explain.py, the account detail card): long format, one row per
# (account, level-2 feature). raw: the feature's input value; z: standardized; contrib: coef x z.
# Per account, intercept + sum(contrib) == logit(score).
CONTRIB_COLS = [ID_COL, "score", "intercept", "feature", "raw", "z", "contrib"]


def as_binary(a) -> np.ndarray:
    """0/1 int array. Matched by position, not pandas index: pass aligned inputs."""
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"labels and predictions must be 1-D, got shape {a.shape}")
    if not np.isin(a, [0, 1]).all():
        raise ValueError("labels and predictions must be 0 or 1")
    return a.astype(int)


def confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) with teen = 1. Inputs are matched by position."""
    y, p = as_binary(y_true), as_binary(y_pred)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def prf(y_true, y_pred) -> tuple[float, float, float, float]:
    """precision, recall, false-teen rate, missed-teen rate; a rate with an empty denominator is 0.0."""
    tp, fp, fn, tn = confusion(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    false_teen = fp / (fp + tn) if (fp + tn) else 0.0  # adults wrongly called teen
    missed_teen = fn / (fn + tp) if (fn + tp) else 0.0
    return prec, rec, false_teen, missed_teen


def f1(prec: float, rec: float) -> float:
    """Harmonic mean of precision and recall, 0 when both are 0."""
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def auc(y_true, score) -> float:
    """ROC AUC; nan when only one class is present."""
    y = as_binary(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def top_share_cutoff(scores, share: float) -> float:
    """Lowest cutoff t such that (score >= t) selects at most floor(share x n) accounts.

    A budget rule, not a label rule: no labels are involved (it is cap_threshold with every
    account counted). Ties never push the selection over the share; with share x n < 1, nobody
    is selected.
    """
    scores = np.asarray(scores, dtype=float)
    return cap_threshold(scores, np.zeros(len(scores), dtype=int), share)


def top_k_precision(y_true, score, k: int) -> float:
    """Share of teens among the k highest-scoring accounts (ties broken by original order)."""
    y, score = as_binary(y_true), np.asarray(score, dtype=float)
    if score.shape != y.shape:
        raise ValueError(f"shape mismatch: {score.shape} vs {y.shape}")
    if not 1 <= k <= len(y):
        raise ValueError(f"k must be between 1 and {len(y)}, got {k}")
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].mean())


def eval_row(stage: str, eval_set: str, y_true, y_pred, score=None) -> dict:
    """One eval.csv row. AUC uses score if given, else the 0/1 prediction itself."""
    prec, rec, ft, mt = prf(y_true, y_pred)
    return {
        "stage": stage,
        "eval_set": eval_set,
        "prec": prec,
        "rec": rec,
        "ft": ft,
        "mt": mt,
        "f1": f1(prec, rec),
        "auc": auc(y_true, y_pred if score is None else score),
    }


def cap_threshold(scores, y, cap: float = DEFAULT_CAP) -> float:
    """Lowest cutoff t such that (score >= t) flags at most k adults, k / n_adults <= cap.

    This is the conservative (1 - cap) quantile of the adult scores: ties never push
    the false-teen rate (computed as in prf) over the cap on the scores it was
    picked from. Pass OOF train scores and train labels only, never test.
    """
    if not 0.0 <= cap <= 1.0:
        raise ValueError("cap must be between 0 and 1")
    scores, y = np.asarray(scores), as_binary(y)
    if not np.issubdtype(scores.dtype, np.floating):
        scores = scores.astype(float)
    if scores.dtype.itemsize > 8:
        # the float() return would round a wider step back down onto the tie
        raise TypeError(f"scores wider than float64 are not supported, got {scores.dtype}")
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    if scores.shape != y.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {y.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite (no NaN or inf)")
    adults = np.sort(scores[y == 0])[::-1]
    n = len(adults)
    if n == 0:
        raise ValueError("no adults to set a cap on")
    # largest k with k / n <= cap, using the same float division prf uses
    k = int(np.floor(cap * n))
    while k < n and (k + 1) / n <= cap:
        k += 1
    while k > 0 and k / n > cap:
        k -= 1
    if k == n:
        return float(scores.min())  # every adult may be flagged; finite, so JSON-safe
    # step up in the scores' own dtype so float32 callers keep the tie guarantee
    return float(np.nextafter(adults[k], adults.dtype.type(np.inf)))
