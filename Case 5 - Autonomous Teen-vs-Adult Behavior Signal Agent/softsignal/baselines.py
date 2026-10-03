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
from softsignal.metrics import DEFAULT_CAP, EVAL_COLS, cap_threshold, eval_row  # re-exported

KEYWORD_COL = "keyword_teen_flag"


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


def nested_fold(
    train: pd.DataFrame, fit_idx, val_idx, cap: float = DEFAULT_CAP, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """(scores, preds) for one outer fold: cutoff from inner OOF on fit_idx rows only."""
    inner = train.iloc[fit_idx].reset_index(drop=True)
    X_in, y_in = split_xy(inner)
    t_inner = cap_threshold(oof_scores(inner, k=k), y_in, cap)
    model = make_tabular_lr().fit(X_in, y_in)
    X_val, _ = split_xy(train.iloc[val_idx])
    score = model.predict_proba(X_val)[:, 1]
    return score, (score >= t_inner).astype(int)


def nested_cv_scores(train: pd.DataFrame, cap: float = DEFAULT_CAP, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (scores, preds) where each row is scored and cut by a rule it never helped pick."""
    scores = np.full(len(train), np.nan)
    preds = np.full(len(train), -1)
    for fit_idx, val_idx in cv_folds(train, k=k):
        scores[val_idx], preds[val_idx] = nested_fold(train, fit_idx, val_idx, cap, k)
    return scores, preds


def cap_label(cap: float) -> str:
    """Cap as a percent string for stage names and printouts: 0.15 -> '15', 0.145 -> '14.5'."""
    return f"{cap * 100:g}"


@dataclass
class TabularLRResult:
    model: Pipeline
    threshold: float
    cap: float
    oof: np.ndarray
    rows: list[dict]

    def coefficients(self) -> pd.Series:
        """Standardized LR coefficients, largest magnitude first (for sanity.txt)."""
        lr = self.model[-1]
        coef = pd.Series(lr.coef_[0], index=self.model.feature_names_in_)
        return coef.reindex(coef.abs().sort_values(ascending=False).index)


def tabular_lr(train: pd.DataFrame, test: pd.DataFrame, cap: float = DEFAULT_CAP) -> TabularLRResult:
    """Ladder row 5: threshold from OOF train scores at the cap, then score test once.

    The nested_cv row measures that cutoff rule on rows it never saw; oof holds the
    scores the final threshold was picked from (so it meets the cap by construction).
    """
    X_tr, y_tr = split_xy(train)
    oof = oof_scores(train)
    t = cap_threshold(oof, y_tr, cap)
    model = make_tabular_lr().fit(X_tr, y_tr)
    X_te, y_te = split_xy(test)
    p_te = model.predict_proba(X_te)[:, 1]
    stage = f"tabular_lr_cap{cap_label(cap)}"
    nested_score, nested_pred = nested_cv_scores(train, cap)
    rows = [
        eval_row(stage, "nested_cv", y_tr, nested_pred, nested_score),
        eval_row(stage, "test", y_te, (p_te >= t).astype(int), p_te),
    ]
    return TabularLRResult(model=model, threshold=t, cap=cap, oof=oof, rows=rows)


def main() -> None:
    train, test = load_data(on_param_mismatch="error")
    lr = tabular_lr(train, test)
    rows = [keyword_baseline(test), *lr.rows]
    pd.set_option("display.width", 120)
    print(pd.DataFrame(rows, columns=EVAL_COLS).round(3).to_string(index=False))
    print(f"\ntabular LR threshold at {cap_label(lr.cap)}% cap (from OOF): {lr.threshold:.3f}")
    print("nested_cv measures the cutoff rule on train rows it never saw (cutoff picked inside")
    print("each outer fold). test applies the full-train OOF cutoff to the model refit on all of")
    print("train, so either false-teen can land above the cap.")
    print("standardized coefficients:")
    print(lr.coefficients().round(2).to_string())


if __name__ == "__main__":
    main()
