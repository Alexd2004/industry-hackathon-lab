"""Shared data loading and the frozen train/test split (Tier 1, step 1)."""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from softsignal.features import FEATURE_COLS, ID_COL, N_TEST, SEED, TARGET

ROOT = Path(__file__).resolve().parents[1]
JOINED = ROOT / "data" / "teen_adult_joined.csv"
SPLIT_FILE = ROOT / "results" / "split.json"


def check_frame(df: pd.DataFrame) -> None:
    """One row per account and a binary target, or raise."""
    if df[ID_COL].duplicated().any():
        raise ValueError(f"{ID_COL} must be unique: expected one row per account")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must be 0 or 1 in every row (no NaN or other values)")


def make_split(df: pd.DataFrame, seed: int = SEED, n_test: int = N_TEST) -> dict:
    """Split by blogger_id, stratified on label_teen, so each account is in exactly one set."""
    check_frame(df)
    train_df, test_df = train_test_split(
        df[[ID_COL, TARGET]], test_size=n_test, stratify=df[TARGET], random_state=seed
    )
    train_ids, test_ids = sorted(train_df[ID_COL]), sorted(test_df[ID_COL])
    is_test = df[ID_COL].isin(set(test_ids))
    return {
        "seed": seed,
        "split_by": ID_COL,
        "stratify_on": TARGET,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "label_counts": {
            "train": {str(k): int(v) for k, v in df.loc[~is_test, TARGET].value_counts().sort_index().items()},
            "test": {str(k): int(v) for k, v in df.loc[is_test, TARGET].value_counts().sort_index().items()},
        },
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def load_data(
    path: Path = JOINED, split_file: Path = SPLIT_FILE, on_param_mismatch: str = "warn"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test). Reads the shared split file, or creates it if missing.

    The file wins over SEED / N_TEST. If they differ from the file, on_param_mismatch
    is "warn" (use the file, warn) or "error" (raise). The file is never regenerated here.
    """
    if on_param_mismatch not in ("warn", "error"):
        raise ValueError('on_param_mismatch must be "warn" or "error"')
    df = pd.read_csv(path, dtype={ID_COL: str})
    check_frame(df)
    if split_file.exists():
        try:
            split = json.loads(split_file.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{split_file} is not valid JSON; delete it to regenerate the split") from e
    else:
        split = make_split(df)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(split, indent=2) + "\n")

    stored = {"seed": split.get("seed"), "n_test": split.get("n_test")}
    current = {"seed": SEED, "n_test": N_TEST}
    if stored != current:
        msg = (
            f"{split_file.name} was made with {stored} but the code has {current}; "
            "the file is used as is. Delete it to regenerate with the code values."
        )
        if on_param_mismatch == "error":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    try:
        train_ids, test_ids = set(split["train_ids"]), set(split["test_ids"])
        stored_counts = split["label_counts"]["test"]
    except KeyError as e:
        raise ValueError(f"{split_file} has no {e}; delete it to regenerate the split") from e
    if train_ids & test_ids or (train_ids | test_ids) != set(df[ID_COL]):
        raise ValueError(
            f"{split_file} does not match {path.name}; delete it to regenerate the split"
        )
    is_test = df[ID_COL].isin(test_ids)
    actual = {str(k): int(v) for k, v in df.loc[is_test, TARGET].value_counts().sort_index().items()}
    if actual != stored_counts:
        raise ValueError(
            f"{split_file} label counts {stored_counts} do not match {path.name} {actual}; "
            "delete it to regenerate the split"
        )
    for label, grp in df.groupby(TARGET):
        expected = len(grp) * len(test_ids) / len(df)
        if abs(grp[ID_COL].isin(test_ids).sum() - expected) > 1:
            raise ValueError(
                f"{split_file} is not stratified on {TARGET} for {path.name}; "
                "delete it to regenerate the split"
            )
    train = df[df[ID_COL].isin(train_ids)].reset_index(drop=True)
    test = df[df[ID_COL].isin(test_ids)].reset_index(drop=True)
    return train, test


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with X limited to the allow-listed FEATURE_COLS."""
    return df[FEATURE_COLS].copy(), df[TARGET]


def cv_folds(train: pd.DataFrame, k: int = 5, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified (fit_idx, val_idx) folds over the train set, as positional indices.

    Pick cutoffs, blend weights and other settings with these folds. Never use the
    test set for that: touch it once, for the final report.
    """
    if TARGET not in train.columns:
        raise ValueError(f"cv_folds needs the train frame with a {TARGET} column")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(train)), train[TARGET]))
