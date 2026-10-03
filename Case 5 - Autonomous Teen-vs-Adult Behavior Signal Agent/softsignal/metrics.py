"""Shared metrics (Tier 1, step 3)."""
import numpy as np


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """precision, recall, false-teen rate, missed-teen rate."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    false_teen = fp / (fp + tn) if (fp + tn) else 0.0  # adults wrongly called teen
    missed_teen = fn / (fn + tp) if (fn + tp) else 0.0
    return prec, rec, false_teen, missed_teen


def f1(prec: float, rec: float) -> float:
    """Harmonic mean of precision and recall, 0 when both are 0."""
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
