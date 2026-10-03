"""Tier 1: style score, its cutoff sweep (step 3) and tune() under the false-teen cap (step 4)."""
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.features import FEATURE_COLS, ID_COL, N_TEST, SEED, TARGET
from softsignal.metrics import f1, prf

# Max false-teen rate (adults wrongly called teen), 15% cap from the Combined Plan.
# Move to policy.yaml cap_false_teen when it exists.
CAP = 0.15
CUTOFFS = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10 .. 0.90

W_GRID = [round(0.05 * i, 2) for i in range(21)]  # weight on activity, 0.00 .. 1.00
CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "best_params.json"


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


def blend(style: pd.Series, activity: pd.Series, w: float) -> pd.Series:
    """(1 - w) * style + w * activity, w is the weight on activity as in the starter."""
    return (1.0 - w) * style + w * activity


def sweep_grid(df: pd.DataFrame, w_grid: list[float] = W_GRID, cutoffs: list[float] = CUTOFFS) -> pd.DataFrame:
    """One row per (w, cutoff): prec, rec, ft, mt, f1 and the number flagged. w=0 equals the step 3 sweep."""
    style, activity = style_score(df), activity_score(df)
    parts = []
    for w in w_grid:
        part = sweep_cutoffs(df, blend(style, activity, w), cutoffs)
        part.insert(0, "w", w)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def pick_points(grid: pd.DataFrame, cap: float = CAP) -> dict[str, dict | None]:
    """cap_best: max recall with ft <= cap, ties to lower ft, then lower w, then lower cutoff.

    f1_best: max f1, ties to lower w, then lower cutoff. cap_best is None when nothing meets the cap.
    """
    under = grid[grid["ft"] <= cap]
    cap_row = (
        under.sort_values(["rec", "ft", "w", "cutoff"], ascending=[False, True, True, True], kind="stable").iloc[0]
        if len(under) else None
    )
    f1_row = grid.sort_values(["f1", "w", "cutoff"], ascending=[False, True, True], kind="stable").iloc[0]
    return {
        name: None if row is None else {"w": float(row["w"]), "cutoff": float(row["cutoff"])}
        for name, row in (("cap_best", cap_row), ("f1_best", f1_row))
    }


def eval_point(df: pd.DataFrame, w: float, cutoff: float) -> dict:
    """Metrics of one (w, cutoff) point on df."""
    pred = flag(blend(style_score(df), activity_score(df), w), cutoff)
    prec, rec, ft, mt = prf(df[TARGET].to_numpy(), pred)
    return {"prec": prec, "rec": rec, "ft": ft, "mt": mt, "f1": f1(prec, rec)}


def rules_id() -> str:
    """Short hash of the scoring code and grids, so a checkpoint is tied to the code that made it."""
    parts = (style_score, activity_score, blend, flag, sweep_cutoffs, prf, f1)
    text = "".join(inspect.getsource(fn) for fn in parts) + repr(W_GRID) + repr(CUTOFFS)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def data_id(df: pd.DataFrame) -> str:
    """Short hash of the train rows (ids, label, model inputs), so a checkpoint is tied to its data."""
    rows = pd.util.hash_pandas_object(df[[ID_COL, TARGET, *FEATURE_COLS]], index=False)
    return hashlib.sha256(rows.to_numpy().tobytes()).hexdigest()[:12]


def load_best(path: Path = CHECKPOINT) -> dict | None:
    """The stored cap-best checkpoint, or None when it is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save_best(
    point: dict,
    train_metrics: dict,
    cap: float,
    n_train: int,
    path: Path = CHECKPOINT,
    rules: str | None = None,
    data: str | None = None,
) -> bool:
    """Write the cap-best point. Under the same cap, split, data and scoring rules, overwrite only on strictly higher train recall."""
    new = {"w": point["w"], "cutoff": point["cutoff"], "cap": cap, "seed": SEED, "n_test": N_TEST,
           "n_train": n_train, "rules": rules or rules_id(), "data": data, "train": train_metrics}
    if path.exists():
        try:
            old = json.loads(path.read_text())
            same = all(old.get(k) == new[k] for k in ("cap", "seed", "n_test", "n_train", "rules", "data"))
            if same and not new["train"]["rec"] > old["train"]["rec"]:
                return False
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            pass  # unreadable checkpoint: replace it
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(new, indent=2) + "\n")
    os.replace(tmp, path)  # atomic, so an interrupted write cannot leave a truncated checkpoint
    return True


def tune(train: pd.DataFrame, test: pd.DataFrame, cap: float = CAP, checkpoint: Path = CHECKPOINT) -> dict:
    """Grid w x cutoff on train, pick under the cap on train, evaluate each pick once on test."""
    grid = sweep_grid(train)
    picks = pick_points(grid, cap)
    rows = []
    for name, pt in picks.items():
        if pt is None:
            continue
        rows.append({"pick": name, **pt, **eval_point(test, pt["w"], pt["cutoff"])})
    test_df = pd.DataFrame(rows)
    test_df["cap_ok"] = test_df["ft"] <= cap
    written = False
    if picks["cap_best"] is not None:
        at = eval_point(train, picks["cap_best"]["w"], picks["cap_best"]["cutoff"])
        written = save_best(picks["cap_best"], at, cap, len(train), checkpoint, data=data_id(train))
    return {"grid": grid, "picks": picks, "test": test_df, "checkpoint_written": written}
