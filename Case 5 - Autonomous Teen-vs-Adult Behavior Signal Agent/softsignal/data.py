"""Shared data loading and the frozen train/test split (Tier 1, step 1)."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.features import FEATURE_COLS, ID_COL, TARGET

ROOT = Path(__file__).resolve().parents[1]
JOINED = ROOT / "data" / "teen_adult_joined.csv"
SPLIT_FILE = ROOT / "results" / "split.json"
SEED = 42
TEST_FRACTION = 0.30


def make_split(df: pd.DataFrame, seed: int = SEED, test_fraction: float = TEST_FRACTION) -> dict:
    """Split by blogger_id, stratified on label_teen, so each account is in exactly one set."""
    if df["blogger_id"].duplicated().any():
        raise ValueError("blogger_id must be unique: expected one row per account")
    rng = np.random.default_rng(seed)
    test_ids: list[str] = []
    for label in sorted(df["label_teen"].unique()):
        group = np.sort(df.loc[df["label_teen"] == label, "blogger_id"].to_numpy())
        n_test = int(round(len(group) * test_fraction))
        test_ids.extend(rng.permutation(group)[:n_test].tolist())
    test_set = set(test_ids)
    train_ids = sorted(i for i in df["blogger_id"] if i not in test_set)
    return {
        "seed": seed,
        "test_fraction": test_fraction,
        "split_by": "blogger_id",
        "stratify_on": "label_teen",
        "train_ids": train_ids,
        "test_ids": sorted(test_ids),
    }


def load_data(
    path: Path = JOINED, split_file: Path = SPLIT_FILE, on_param_mismatch: str = "warn"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test). Reads the shared split file, or creates it if missing.

    The file wins over SEED / TEST_FRACTION. If they differ from the file, on_param_mismatch
    is "warn" (use the file, warn) or "error" (raise). The file is never regenerated here.
    """
    if on_param_mismatch not in ("warn", "error"):
        raise ValueError('on_param_mismatch must be "warn" or "error"')
    df = pd.read_csv(path)
    if split_file.exists():
        split = json.loads(split_file.read_text())
    else:
        split = make_split(df)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(split, indent=2) + "\n")

    stored = {"seed": split.get("seed"), "test_fraction": split.get("test_fraction")}
    current = {"seed": SEED, "test_fraction": TEST_FRACTION}
    if stored != current:
        msg = (
            f"{split_file.name} was made with {stored} but the code has {current}; "
            "the file is used as is. Delete it to regenerate with the code values."
        )
        if on_param_mismatch == "error":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    train_ids, test_ids = set(split["train_ids"]), set(split["test_ids"])
    if train_ids & test_ids or (train_ids | test_ids) != set(df["blogger_id"]):
        raise ValueError(
            f"{split_file} does not match {path.name}; delete it to regenerate the split"
        )
    for label, grp in df.groupby(TARGET):
        expected = int(round(len(grp) * split["test_fraction"]))
        if grp[ID_COL].isin(test_ids).sum() != expected:
            raise ValueError(
                f"{split_file} is not stratified on {TARGET} for {path.name}; "
                "delete it to regenerate the split"
            )
    train = df[df["blogger_id"].isin(train_ids)].reset_index(drop=True)
    test = df[df["blogger_id"].isin(test_ids)].reset_index(drop=True)
    return train, test


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with X limited to the allow-listed FEATURE_COLS."""
    return df[FEATURE_COLS].copy(), df[TARGET]
