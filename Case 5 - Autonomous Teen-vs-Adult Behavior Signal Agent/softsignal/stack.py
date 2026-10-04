"""Level 2 of the stack (Tier 2, step 8): text_score + the 16 allowed columns -> one score.

Level 1 is the TF-IDF text model (text_model.py). Level 2 is StandardScaler + LR(C=1) on
logit(text_score) plus the 16 FEATURE_COLS. The level-2 model is trained on out-of-fold
(OOF) text scores, never on text scores a row's own label helped produce.

stack_oof.csv holds one honest level-2 score per train account, from a nested 5x5 OOF:
per outer fold, the inner 5-fold OOF text scores of the outer-fit rows train level 2, the
outer-val rows get a text score from a text model refit on the outer-fit rows only, and
level 2 scores them. So no train row's final score comes from a model that saw its label,
and policy.py can pick thresholds from it. Test rows are scored once, by the full-train
text model and the full-train level 2.

use_text=False is the checkpoint fallback: the same level 2 on the 16 columns alone, i.e.
the tabular LR of step 6, with no text model and no TF-IDF cache needed.

Run: python -m softsignal.stack
"""
import hashlib
import inspect
import json
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import softsignal.text_model as tmod
from softsignal.baselines import cap_label, make_tabular_lr, oof_scores
from softsignal.data import ROOT, cv_folds, load_data
from softsignal.features import FEATURE_COLS, ID_COL, TARGET
from softsignal.metrics import DEFAULT_CAP, EVAL_COLS, auc, cap_threshold, eval_row
from softsignal.text_model import TextMatrix, build_matrix, fit_text_model, oof_text_score, score as text_score

STACK_OOF = ROOT / "cache" / "stack_oof.csv"
TEXT_FEATURE = "logit_text_score"
CLIP = 1e-6  # keeps logit(0) and logit(1) finite


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def stack_features(df: pd.DataFrame, text_p=None) -> pd.DataFrame:
    """Level-2 input: the 16 allowed columns, plus logit(text_score) first when text_p is given."""
    X = df[FEATURE_COLS].copy()  # the allow-list only; works on frames with or without a label
    if text_p is not None:
        X.insert(0, TEXT_FEATURE, logit(text_p))
    return X.reset_index(drop=True)


def _fit_level2(X: pd.DataFrame, y) -> Pipeline:
    return make_tabular_lr().fit(X, np.asarray(y))


def check_vocabulary(tm: TextMatrix, train_ids, stacklevel: int = 2) -> None:
    """Raise if the TF-IDF vocabulary was fit on any account outside the train set (e.g. on test text).

    train_ids is the whole train set. A frame smaller than train (a subsample, or the revealed
    labels of a loop refit) is fine as long as the matrix was fit inside train. A matrix with
    no fit_ids (hand-built) cannot be checked, so it only warns; build_matrix always sets them.
    stacklevel is where a warning is attributed: 2 is the caller of this function.
    """
    if tm.fit_ids is None:
        warnings.warn("the text matrix has no fit_ids, so its vocabulary cannot be checked as train-only",
                      stacklevel=stacklevel)
        return
    outside = tm.fit_ids - set(pd.Index(train_ids).astype(str))
    if outside:
        raise ValueError(
            f"the text matrix vocabulary was fit on {len(outside)} account(s) outside train, "
            f"e.g. {sorted(outside)[:3]}; build it with build_matrix(train[ID_COL]), or if the frame is "
            "only part of train, pass the whole train set as train_ids="
        )


def nested_oof(tm: TextMatrix, train: pd.DataFrame, k: int = 5, train_ids=None) -> np.ndarray:
    """One level-2 score per train row, each from models that never saw that row (nested k x k).

    train_ids: the whole train set when `train` is only part of it; defaults to train's own ids.
    The vocabulary was fit once on all those rows (no labels), so it has seen each outer-val
    row's text but never its label.
    """
    check_vocabulary(tm, train[ID_COL] if train_ids is None else train_ids, stacklevel=3)
    ids, y = train[ID_COL].to_numpy(), train[TARGET].to_numpy()
    out = np.full(len(train), np.nan)
    for fit_idx, val_idx in cv_folds(train, k=k):
        fit_rows = train.iloc[fit_idx]
        p_fit = oof_text_score(tm, fit_rows, k=k)  # inner OOF: level 2 trains on honest scores
        p_val = text_score(fit_text_model(tm, ids[fit_idx], y[fit_idx]), tm, ids[val_idx])
        level2 = _fit_level2(stack_features(fit_rows, p_fit), y[fit_idx])
        out[val_idx] = level2.predict_proba(stack_features(train.iloc[val_idx], p_val))[:, 1]
    return out


@dataclass
class Stack:
    """The fitted full-train stack. text_model and tm are None when use_text is False."""

    level2: Pipeline
    use_text: bool
    text_model: LogisticRegression | None = None
    tm: TextMatrix | None = None

    @classmethod
    def fit(
        cls, train: pd.DataFrame, tm: TextMatrix | None = None, use_text: bool = True, train_ids=None
    ) -> "Stack":
        y = train[TARGET].to_numpy()
        if not use_text:
            return cls(level2=_fit_level2(stack_features(train), y), use_text=False)
        tm = tm if tm is not None else build_matrix(train[ID_COL])
        check_vocabulary(tm, train[ID_COL] if train_ids is None else train_ids, stacklevel=3)
        level2 = _fit_level2(stack_features(train, oof_text_score(tm, train)), y)
        return cls(level2=level2, use_text=True, text_model=fit_text_model(tm, train[ID_COL], y), tm=tm)

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_text:
            return stack_features(df)
        return stack_features(df, text_score(self.text_model, self.tm, df[ID_COL]))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """p(teen) for these accounts."""
        return self.level2.predict_proba(self._features(df))[:, 1]

    def coefs(self) -> pd.Series:
        """Standardized level-2 coefficients, largest magnitude first (for sanity.txt)."""
        coef = pd.Series(self.level2[-1].coef_[0], index=self.level2.feature_names_in_)
        return coef.reindex(coef.abs().sort_values(ascending=False).index)

    def contributions(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
        """(X, z, contrib, intercept): level-2 input, its transformed values, coef x z, and the intercept.

        logit(score(df)) == intercept + contrib.sum(axis=1), exactly. Every step before the final
        LR is applied (level2[:-1]), so this stays right if the pipeline gains a step.
        """
        X = self._features(df)
        z = np.asarray(self.level2[:-1].transform(X), dtype=float)
        lr = self.level2[-1]
        return X, z, z * lr.coef_[0], float(lr.intercept_[0])

    def explain(self, df: pd.DataFrame, n: int = 3) -> list[list[tuple[str, float]]]:
        """Per account, its n largest signed contributions (coef x standardized value) to the logit.

        Positive pushes toward teen, negative toward adult. Feature names are level-2 columns.
        """
        X, _, contrib, _ = self.contributions(df)
        names = np.asarray(X.columns)
        out = []
        for row in contrib:
            top = np.argsort(-np.abs(row), kind="stable")[:n]
            out.append([(str(names[j]), float(row[j])) for j in top])
        return out


def _key_functions(use_text: bool) -> list:
    """Every function whose code decides the OOF scores. Their source and signature (so default
    arguments such as cv_folds' seed count too) go into the cache key."""
    funcs = [nested_oof, _fit_level2, stack_features, logit, make_tabular_lr, cv_folds, check_vocabulary]
    if use_text:
        funcs += [oof_text_score, fit_text_model, tmod.score, tmod.make_text_lr, tmod.make_vectorizer,
                  tmod.load_docs, tmod.build_matrix, TextMatrix.rows]
    else:
        funcs.append(oof_scores)
    return funcs


def oof_cache_key(train: pd.DataFrame, use_text: bool = True, tm_key: str | None = None) -> str:
    """Hash of everything the nested OOF scores depend on, so a stale cache is detectable.

    Covers the train rows (ids, label, the 16 columns and their names), the text matrix (tm_key, which
    already covers the posts file, the vocabulary ids, the text settings and sklearn; None means the
    default build_matrix(train ids) that policy.py would rebuild), the Python, numpy, scipy and sklearn versions,
    and the source and signature of _key_functions. Any edit to those source texts, even a comment,
    changes it: the cost of a false alarm is rerunning stack.py.
    """
    cols = [ID_COL, TARGET, *FEATURE_COLS]
    data = train[cols].astype({ID_COL: str})
    if use_text and tm_key is None:
        tm_key = tmod.cache_key(tmod.POSTS, train[ID_COL], True)
    payload = {
        "columns": cols,
        "data": hashlib.sha256(pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes()).hexdigest(),
        "use_text": use_text,
        "text_matrix": tm_key if use_text else None,
        "text_params": [tmod.VECTORIZER_PARAMS, tmod.C, tmod.DOCS_VERSION] if use_text else None,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "clip": CLIP,
        "source": [inspect.getsource(f) + str(inspect.signature(f)) for f in _key_functions(use_text)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def csv_sha256(path: Path) -> str:
    """Hash of the OOF csv's bytes, kept in the meta file so an edited or swapped csv is detected."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def meta_path(path: Path) -> Path:
    """Sidecar next to the OOF csv: cache/stack_oof.csv -> cache/stack_oof.meta.json."""
    return path.with_suffix(".meta.json")


def write_oof(
    train: pd.DataFrame,
    oof: np.ndarray,
    path: Path = STACK_OOF,
    use_text: bool = True,
    tm_key: str | None = None,
) -> None:
    """cache/stack_oof.csv: blogger_id and stack_oof, in train order (labels stay in the data file).

    Also writes the meta file (meta_path) with the cache key of the text matrix that scored it (tm_key).
    The old sidecar is removed first and the new one written last, so a crash in between leaves a csv
    with no sidecar, which policy.py rejects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = meta_path(path)
    meta.unlink(missing_ok=True)
    pd.DataFrame({ID_COL: train[ID_COL].to_numpy(), "stack_oof": oof}).to_csv(path, index=False)
    meta.write_text(json.dumps({"cache_key": oof_cache_key(train, use_text, tm_key), "use_text": use_text,
                                "n": len(train), "csv_sha256": csv_sha256(path)}), encoding="utf-8")


@dataclass
class StackResult:
    stack: Stack
    oof: np.ndarray  # train rows, in train order
    test_score: np.ndarray  # test rows, in test order
    threshold: float
    cap: float
    rows: list[dict]


def stack(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cap: float = DEFAULT_CAP,
    tm: TextMatrix | None = None,
    use_text: bool = True,
    oof_path: Path | None = None,
    train_ids=None,
) -> StackResult:
    """Ladder row 7: nested OOF on train, cutoff at the cap from that OOF, test scored once.

    train_ids: the whole train set when `train` is only part of it (see check_vocabulary).
    oof_path: where to write the OOF cache and its meta file; None (the default) writes nothing, so a
    partial-train call such as a loop refit can never overwrite the audit cache. main() passes STACK_OOF.
    """
    y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    if use_text:
        tm = tm if tm is not None else build_matrix(train[ID_COL])
        check_vocabulary(tm, train[ID_COL] if train_ids is None else train_ids, stacklevel=3)
        with warnings.catch_warnings():  # already reported above, at the caller's line
            warnings.filterwarnings("ignore", message="the text matrix has no fit_ids")
            oof = nested_oof(tm, train, train_ids=train_ids)
            model = Stack.fit(train, tm=tm, train_ids=train_ids)
    else:
        oof = oof_scores(train)
        model = Stack.fit(train, use_text=False)
    t = cap_threshold(oof, y_tr, cap)
    p_te = model.score(test)
    if oof_path is not None:
        write_oof(train, oof, oof_path, use_text=use_text, tm_key=tm.key if use_text else None)
    stage = f"{'stack' if use_text else 'stack_tabular_only'}_cap{cap_label(cap)}"
    rows = [
        eval_row(stage, "cv_oof", y_tr, (oof >= t).astype(int), oof),
        eval_row(stage, "test", y_te, (p_te >= t).astype(int), p_te),
    ]
    return StackResult(stack=model, oof=oof, test_score=p_te, threshold=t, cap=cap, rows=rows)


def main() -> None:
    train, test = load_data(on_param_mismatch="error")
    res = stack(train, test, oof_path=STACK_OOF)
    pd.set_option("display.width", 120)
    print(pd.DataFrame(res.rows, columns=EVAL_COLS).round(3).to_string(index=False))
    print(f"\nOOF AUC (nested 5x5, train): {auc(train[TARGET], res.oof):.4f}")
    print(f"test AUC (full-train stack): {auc(test[TARGET], res.test_score):.4f}")
    print("go/no-go gate AUC >= 0.94, target >= 0.95")
    print(f"\nstack cutoff at {cap_label(res.cap)}% cap (from nested OOF): {res.threshold:.3f}")
    print("train false-teen meets the cap by construction (the cutoff comes from these nested OOF scores);")
    print("test applies that cutoff to the model refit on all of train, so test false-teen can land above it.")
    print(f"wrote {len(res.oof)} OOF scores to {STACK_OOF}")
    print("standardized level-2 coefficients:")
    print(res.stack.coefs().round(2).to_string())


if __name__ == "__main__":
    main()
