"""Shared metrics and eval.csv rows. Every stage of the results ladder is scored here."""
import numpy as np
from sklearn.metrics import roc_auc_score

# Frozen eval.csv columns (Combined Plan section 4).
EVAL_COLS = ["stage", "eval_set", "prec", "rec", "ft", "mt", "f1", "auc"]


def as_binary(a) -> np.ndarray:
    """0/1 int array. Matched by position, not pandas index: pass aligned inputs."""
    a = np.asarray(a)
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
    """precision, recall, false-teen rate, missed-teen rate (same as agent_starter.py)."""
    tp, fp, fn, tn = confusion(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    false_teen = fp / (fp + tn) if (fp + tn) else 0.0  # adults wrongly called teen
    missed_teen = fn / (fn + tp) if (fn + tp) else 0.0
    return prec, rec, false_teen, missed_teen


def f1(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def auc(y_true, score) -> float:
    """ROC AUC; nan when only one class is present."""
    y = as_binary(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


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
