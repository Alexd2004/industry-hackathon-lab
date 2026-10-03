"""Keyword baseline (Tier 1, step 2) and tabular LR fallback (Tier 1, step 6).

Thresholds are picked from out-of-fold (OOF) train scores only; the test set is
scored once, for the report. Run: python -m softsignal.baselines
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from softsignal.data import cv_folds, load_data, split_xy
from softsignal.features import TARGET
from softsignal.metrics import EVAL_COLS, as_binary, eval_row

KEYWORD_COL = "keyword_teen_flag"
DEFAULT_CAP = 0.15


def keyword_baseline(df: pd.DataFrame, eval_set: str = "test") -> dict:
    """Ladder row 1: flag = teen. A 0/1 rule, so its AUC is (rec + 1 - ft) / 2."""
    return eval_row("keyword_baseline", eval_set, df[TARGET], df[KEYWORD_COL])


def make_tabular_lr() -> Pipeline:
    """StandardScaler + LR on the 16 allowed columns. C barely matters here (0.1-10 measured equal)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def oof_scores(train: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Out-of-fold p(teen) for every train row; each fold's model never sees its own rows."""
    X, y = split_xy(train)
    out = np.full(len(train), np.nan)
    for fit_idx, val_idx in cv_folds(train, k=k):
        model = make_tabular_lr().fit(X.iloc[fit_idx], y.iloc[fit_idx])
        out[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return out


def cap_threshold(scores, y, cap: float = DEFAULT_CAP) -> float:
    """Lowest cutoff t such that (score >= t) flags at most k adults, k / n_adults <= cap.

    This is the conservative (1 - cap) quantile of the adult scores: ties never push
    the false-teen rate (computed as in metrics.prf) over the cap on the scores it was
    picked from. Pass OOF train scores and train labels only, never test.
    """
    if not 0.0 <= cap <= 1.0:
        raise ValueError("cap must be between 0 and 1")
    scores, y = np.asarray(scores), as_binary(y)
    if not np.issubdtype(scores.dtype, np.floating):
        scores = scores.astype(float)
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


@dataclass
class TabularLRResult:
    model: Pipeline
    threshold: float
    cap: float
    oof: np.ndarray
    rows: list[dict]

    def coefficients(self) -> pd.Series:
        """Standardized LR coefficients, largest magnitude first (for sanity.txt)."""
        lr = self.model.named_steps["logisticregression"]
        coef = pd.Series(lr.coef_[0], index=self.model.feature_names_in_)
        return coef.reindex(coef.abs().sort_values(ascending=False).index)


def tabular_lr(train: pd.DataFrame, test: pd.DataFrame, cap: float = DEFAULT_CAP) -> TabularLRResult:
    """Ladder row 5: threshold from OOF train scores at the cap, then score test once."""
    X_tr, y_tr = split_xy(train)
    oof = oof_scores(train)
    t = cap_threshold(oof, y_tr, cap)
    model = make_tabular_lr().fit(X_tr, y_tr)
    X_te, y_te = split_xy(test)
    p_te = model.predict_proba(X_te)[:, 1]
    stage = f"tabular_lr_cap{cap * 100:g}"
    rows = [
        eval_row(stage, "cv_oof", y_tr, (oof >= t).astype(int), oof),
        eval_row(stage, "test", y_te, (p_te >= t).astype(int), p_te),
    ]
    return TabularLRResult(model=model, threshold=t, cap=cap, oof=oof, rows=rows)


def main() -> None:
    train, test = load_data(on_param_mismatch="error")
    lr = tabular_lr(train, test)
    rows = [keyword_baseline(test), *lr.rows]
    pd.set_option("display.width", 120)
    print(pd.DataFrame(rows, columns=EVAL_COLS).round(3).to_string(index=False))
    print(f"\ntabular LR threshold at {lr.cap:.0%} cap (from OOF): {lr.threshold:.3f}")
    print("cv_oof holds the cap by construction; the test row applies the same cutoff to the")
    print("model refit on all of train, so its false-teen can land above the cap.")
    print("standardized coefficients:")
    print(lr.coefficients().round(2).to_string())


if __name__ == "__main__":
    main()
