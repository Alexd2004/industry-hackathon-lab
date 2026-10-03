"""Tier 1: style score and its cutoff sweep (step 3). tune() comes in step 4."""
import numpy as np
import pandas as pd

from softsignal.features import TARGET
from softsignal.metrics import DEFAULT_CAP, f1, prf

CAP = DEFAULT_CAP  # max false-teen rate, shared with baselines; move to policy.yaml cap_false_teen when it exists
CUTOFFS = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10 .. 0.90


def style_score(df: pd.DataFrame) -> pd.Series:
    """Weighted sum of five text rules, the starter's exact weights and thresholds."""
    return (
        0.35 * (df["avg_word_len"] < 4.4).astype(float)
        + 0.25 * (df["first_person_rate"] > 0.06).astype(float)
        + 0.20 * (df["exclaim_rate"] > 0.008).astype(float)
        + 0.15 * (df["slang_emoji_rate"] > 0.002).astype(float)
        + 0.20 * (df["school_token_rate"] > 0).astype(float)
    )


def activity_score(df: pd.DataFrame) -> pd.Series:
    """Weighted sum of five activity rules, the starter's exact weights and thresholds."""
    return (
        0.25 * (df["pct_active_school_hours"] < 0.25).astype(float)
        + 0.25 * (df["pct_active_evening"] > 0.35).astype(float)
        + 0.20 * (df["share_short_video_views"] > 0.40).astype(float)
        + 0.15 * (df["night_notification_open_rate"] > 0.22).astype(float)
        + 0.15 * (df["weekend_weekday_session_ratio"] > 1.2).astype(float)
    )


def flag(score: pd.Series, cut: float) -> np.ndarray:
    """Teen call at or above the cutoff. Rounded so 0.55 is not lost to float error."""
    return (score.round(9) >= cut).to_numpy().astype(int)


def sweep_cutoffs(df: pd.DataFrame, score: pd.Series, cutoffs: list[float] = CUTOFFS) -> pd.DataFrame:
    """One row per cutoff: prec, rec, ft, mt, f1 and the number flagged."""
    y = df[TARGET].to_numpy()
    rows = []
    for cut in cutoffs:
        pred = flag(score, cut)
        prec, rec, ft, mt = prf(y, pred)
        rows.append(
            {"cutoff": cut, "prec": prec, "rec": rec, "ft": ft, "mt": mt,
             "f1": f1(prec, rec), "n_flagged": int(pred.sum())}
        )
    return pd.DataFrame(rows)


def best_cutoffs(sweep: pd.DataFrame, cap: float = CAP) -> dict[str, float | None]:
    """F1-best cutoff and cap-best cutoff (max recall with ft <= cap). Ties go to the lower cutoff.

    cap-best is None when no cutoff on the grid meets the cap.
    """
    ordered = sweep.sort_values("cutoff", kind="stable")
    f1_best = ordered.loc[ordered["f1"].idxmax(), "cutoff"]  # idxmax returns the first max
    under = ordered[ordered["ft"] <= cap]
    cap_best = under.loc[under["rec"].idxmax(), "cutoff"] if len(under) else None
    return {"f1_best": float(f1_best), "cap_best": None if cap_best is None else float(cap_best)}


def flag_all_f1(df: pd.DataFrame) -> float:
    """F1 of calling every account a teen. Reference line: a pick below this adds nothing."""
    prec, rec, _, _ = prf(df[TARGET].to_numpy(), np.ones(len(df), dtype=int))
    return f1(prec, rec)


def style_sweep_report(train: pd.DataFrame, test: pd.DataFrame, cap: float = CAP) -> dict:
    """Sweep on train, pick the two cutoffs on train, evaluate each once on test."""
    sweep = sweep_cutoffs(train, style_score(train))
    picks = best_cutoffs(sweep, cap)
    test_score = style_score(test)
    test_rows = []
    for name, cut in picks.items():
        if cut is None:
            continue
        pred = flag(test_score, cut)
        prec, rec, ft, mt = prf(test[TARGET].to_numpy(), pred)
        test_rows.append(
            {"pick": name, "cutoff": cut, "prec": prec, "rec": rec, "ft": ft, "mt": mt,
             "f1": f1(prec, rec), "cap_ok": ft <= cap}
        )
    return {
        "train_sweep": sweep,
        "picks": picks,
        "test": pd.DataFrame(test_rows),
        "flag_all_f1": flag_all_f1(test),
    }
