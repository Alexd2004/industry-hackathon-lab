"""Label oracle for the review loop (Tier 2, step 11).

The oracle plays the human verifier. It hands out train accounts in batches of 300 (features only),
then reveals labels for the accounts the model sent to review (the verify band, at most 25% of the
batch) plus 60 random audit accounts per batch. Labels live only in this object. Test accounts are
never handed out or revealed: test metrics never come from the oracle.

Simulation choice: batches are stratified on the label, so every batch is 150 teens / 150 adults
and round-to-round changes come from learning, not from batch mix.

Drift (optional, off by default): a Drift shifts chosen feature columns, in train standard deviations, on every batch
from its start_round on. Labels are untouched and the stored rows stay as they were, so the shift is a pure covariate
shift added at hand-out. It uses only the train features' spread, never a label. shift_frame() applies the same shift
to any frame (the test report in step 2), so a drifted run can be scored against drifted test rows.
"""
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from softsignal.data import JOINED, SPLIT_FILE, check_frame, load_data
from softsignal.features import FEATURE_COLS, ID_COL, SEED, TARGET
from softsignal.policy import load_policy

_POLICY = load_policy()
BATCH_SIZE = 300
AUDIT_PER_BATCH = _POLICY["audit_per_batch"]  # policy.yaml
REVIEW_BUDGET = _POLICY["review_budget"]  # max share of a batch in the verify band, policy.yaml

LABEL_COLS = [ID_COL, TARGET, "in_verify", "in_audit"]
REVEALED_COLS = ["round", *LABEL_COLS]
LABEL_DTYPES = {TARGET: "int64", "in_verify": bool, "in_audit": bool}
# Log counts are for that round only. rounds.csv's n_audit_adults is cumulative: take it from audit_counts().
LOG_KEYS = ["round", "verify_ids", "audit_ids", "n_verify", "n_audit", "n_overlap", "n_labels", "n_audit_adults"]


class OracleError(ValueError):
    """A reveal or setup that would break the loop rules (test id, repeat, over budget, wrong batch)."""


@dataclass(frozen=True)
class Drift:
    """A covariate shift: from start_round on, each column in columns moves by shift train standard deviations."""

    start_round: int
    shift: float
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.start_round < 1 or not math.isfinite(self.shift) or not self.columns:
            raise OracleError(f"a drift needs start_round >= 1, a finite shift and columns, got {self}")
        bad = [c for c in self.columns if c not in FEATURE_COLS]
        if bad:
            raise OracleError(f"a drift may only move allowed feature columns, not {bad}")


@dataclass(frozen=True)
class Batch:
    """One round of unlabelled accounts. rows has ID_COL + FEATURE_COLS only, in ids order."""

    round: int
    ids: tuple[str, ...]
    rows: pd.DataFrame


@dataclass(frozen=True)
class Reveal:
    """Labels revealed for one round: one row per unique id, flagged in_verify and/or in_audit."""

    round: int
    labels: pd.DataFrame
    n_verify: int
    n_audit: int
    n_overlap: int


class Oracle:
    """Hands out train batches and reveals verify-band + audit labels, once per round."""

    def __init__(
        self,
        train: pd.DataFrame,
        test_ids: Iterable[str],
        seed: int = SEED,
        batch_size: int = BATCH_SIZE,
        audit_per_batch: int = AUDIT_PER_BATCH,
        review_budget: float = REVIEW_BUDGET,
        drift: Drift | None = None,
    ) -> None:
        missing = [c for c in (ID_COL, TARGET, *FEATURE_COLS) if c not in train.columns]
        if missing:
            raise OracleError(f"train frame is missing columns {missing}")
        check_frame(train)
        if batch_size < 1 or audit_per_batch < 0 or not 0.0 <= review_budget <= 1.0:
            raise OracleError(
                f"need batch_size >= 1, audit_per_batch >= 0 and 0 <= review_budget <= 1, got "
                f"{batch_size}, {audit_per_batch}, {review_budget}"
            )
        self._test_ids = frozenset(str(i) for i in test_ids)
        ids = train[ID_COL].astype(str)
        overlap = self._test_ids & set(ids)
        if overlap:
            raise OracleError(f"{len(overlap)} ids are in both train and test, e.g. {sorted(overlap)[0]}")

        # The only place labels live. Rows keep the allow-listed features and nothing else.
        self._labels: dict[str, int] = dict(zip(ids, train[TARGET].astype(int)))
        self._rows = train[FEATURE_COLS].copy().set_index(ids.rename(ID_COL)).sort_index()

        self.seed = seed
        self.review_budget = review_budget
        self.drift = drift
        self._lo, self._hi = self._rows.min(), self._rows.max()  # a shifted value stays inside the train range
        self._std = self._rows.std(ddof=0)
        self._batches = self._make_batches(batch_size)
        self._audit = [
            tuple(sorted(np.random.default_rng([seed, r]).choice(
                list(b), min(audit_per_batch, len(b)), replace=False
            ).tolist()))
            for r, b in enumerate(self._batches, start=1)
        ]
        self._cursor = 0  # round of the last batch handed out, 0 before the first
        self._revealed_rounds: set[int] = set()
        self._store: list[pd.DataFrame] = []
        self.log: list[dict] = []

    def _make_batches(self, batch_size: int) -> list[tuple[str, ...]]:
        """Stratified folds over sorted ids; fold k is round k + 1. Fold sizes differ by at most 1."""
        ids = np.array(self._rows.index)  # sorted
        y = np.array([self._labels[i] for i in ids])
        n_batches = max(1, round(len(ids) / batch_size))
        if n_batches == 1:
            return [tuple(ids)]
        skf = StratifiedKFold(n_splits=n_batches, shuffle=True, random_state=self.seed)
        return [tuple(ids[val]) for _, val in skf.split(np.zeros(len(ids)), y)]

    @classmethod
    def from_split(cls, path: Path = JOINED, split_file: Path = SPLIT_FILE, **kw) -> "Oracle":
        """Oracle over the committed split. Gets the train frame and test ids only, never test labels."""
        train, test = load_data(path, split_file)
        return cls(train, test[ID_COL].tolist(), **kw)

    @property
    def n_rounds(self) -> int:
        return len(self._batches)

    def verify_budget(self, batch: Batch) -> int:
        """Max unique ids in the verify band for this batch: floor(review_budget * batch size).

        The 1e-9 keeps float error from costing a slot (0.29 * 100 is 28.999999999999996).
        """
        return math.floor(self.review_budget * len(batch.ids) + 1e-9)

    def shift_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """df with the drift applied (a copy; df unchanged). No drift: an unchanged copy."""
        out = df.copy()
        if self.drift is None:
            return out
        for c in self.drift.columns:
            out[c] = (out[c] + self.drift.shift * self._std[c]).clip(self._lo[c], self._hi[c])
        return out

    def next_batch(self) -> Batch | None:
        """The next round's batch (features only), or None after the last round."""
        if self._cursor >= len(self._batches):
            return None
        self._cursor += 1
        ids = self._batches[self._cursor - 1]
        rows = self._rows.loc[list(ids)].reset_index()
        if self.drift is not None and self._cursor >= self.drift.start_round:
            rows = self.shift_frame(rows)
        return Batch(round=self._cursor, ids=ids, rows=rows)

    def __iter__(self) -> Iterator[Batch]:
        while (batch := self.next_batch()) is not None:
            yield batch

    def reveal(self, batch: Batch, verify_ids: Iterable[str]) -> Reveal:
        """Labels for the verify band plus this round's fixed audit slice. Once per round.

        The caller truncates the band to the budget (top ids by score); duplicates are dropped.
        """
        band = list(dict.fromkeys(verify_ids))
        for i in band:
            if i in self._test_ids:
                raise OracleError(f"test id {i} is never revealed")
        if not isinstance(batch, Batch) or not 1 <= batch.round <= len(self._batches):
            raise OracleError("reveal needs a Batch handed out by this oracle")
        r = batch.round
        if r in self._revealed_rounds:
            raise OracleError(f"round {r} was already revealed; each round is revealed once")
        if r != self._cursor:
            raise OracleError(f"round {r} is not the current round ({self._cursor}); reveal it before next_batch()")
        if tuple(batch.ids) != self._batches[r - 1]:
            raise OracleError(f"batch ids do not match round {r} of this oracle")
        in_batch = set(self._batches[r - 1])
        outside = [i for i in band if i not in in_batch]
        if outside:
            raise OracleError(f"{len(outside)} verify ids are not in round {r}'s batch, e.g. {outside[0]}")
        budget = self.verify_budget(batch)
        if len(band) > budget:
            raise OracleError(
                f"verify band has {len(band)} unique ids, budget is {budget} "
                f"(floor({self.review_budget} * {len(batch.ids)})); truncate to the top {budget} by score"
            )

        audit = self._audit[r - 1]
        verify_set, audit_set = set(band), set(audit)
        ids = sorted(verify_set | audit_set)
        labels = pd.DataFrame({
            ID_COL: ids,
            TARGET: [self._labels[i] for i in ids],
            "in_verify": [i in verify_set for i in ids],
            "in_audit": [i in audit_set for i in ids],
        }, columns=LABEL_COLS).astype(LABEL_DTYPES)
        n_overlap = len(verify_set & audit_set)

        self._revealed_rounds.add(r)
        self._store.append(labels.assign(round=r)[REVEALED_COLS])
        self.log.append({
            "round": r,
            "verify_ids": band,
            "audit_ids": list(audit),
            "n_verify": len(band),
            "n_audit": len(audit),
            "n_overlap": n_overlap,
            "n_labels": len(ids),
            "n_audit_adults": sum(self._labels[i] == 0 for i in audit),
        })
        return Reveal(round=r, labels=labels.copy(), n_verify=len(band), n_audit=len(audit), n_overlap=n_overlap)

    def revealed(
        self, source: Literal["audit", "verify", "all"] = "all", before_round: int | None = None
    ) -> pd.DataFrame:
        """Labels revealed so far, with a round column. before_round=r keeps rounds < r only."""
        if source not in ("audit", "verify", "all"):
            raise OracleError(f'source must be "audit", "verify" or "all", got {source!r}')
        if not self._store:
            # Typed even when empty: an object-dtype in_audit column would make df[df["in_audit"]] select columns.
            return pd.DataFrame(columns=REVEALED_COLS).astype({"round": "int64", **LABEL_DTYPES})
        out = pd.concat(self._store, ignore_index=True)
        if before_round is not None:
            out = out[out["round"] < before_round]
        if source != "all":
            out = out[out[f"in_{source}"]]
        return out.reset_index(drop=True)

    def audit_counts(self, before_round: int | None = None) -> dict[str, int]:
        """Revealed audit accounts by class: {"adults": n, "teens": n}. Drives the hold rule."""
        audit = self.revealed("audit", before_round)
        teens = int((audit[TARGET] == 1).sum())
        return {"adults": len(audit) - teens, "teens": teens}
