"""TF-IDF text model, level 1 of the stack (Tier 2, step 7).

Each account's posts are joined into one document. A word (1,2) + char_wb (2,4)
TF-IDF vocabulary is fit once on TRAIN text only (no labels, no test text), every
account is transformed, and the matrix is cached in cache/. A logistic regression
on it gives text_score = p(teen):
- train rows get a 5-fold out-of-fold (OOF) score, for stack.py level 2;
- test rows get the score of the model refit on all of train.

Only blogger_id, post_ix and text are read from the posts file; its age, gender,
job and is_teen columns are never loaded. Digits are masked by default, so an age
typed in a post ("I'm 26") can't be read; measured on our split, AUC moves by under
0.001 (OOF 0.8878 -> 0.8880, test 0.8879 -> 0.8885).

For nested OOF in stack.py, don't reuse the full-train oof: per outer fold, take
oof_text_score(tm, train.iloc[outer_fit]) for the outer-fit rows and
score(fit_text_model(tm, outer_fit ids, labels), tm, outer_val ids) for the outer-val
rows, so no row's text_score comes from a model that saw its own label.

Run: python -m softsignal.text_model
"""
import hashlib
import json
import os
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from softsignal.data import ROOT, cv_folds, load_data
from softsignal.features import ID_COL, TARGET
from softsignal.metrics import DEFAULT_CAP, EVAL_COLS, cap_threshold, eval_row

POSTS = ROOT / "data" / "blogger_posts_sample.csv"
CACHE_DIR = ROOT / "cache"
POST_COLS = [ID_COL, "post_ix", "text"]  # nothing else from the posts file is read

# Winning Plan settings (section 3, level 1); measured, not tuned on test.
VECTORIZER_PARAMS = {
    "word": {"ngram_range": (1, 2), "min_df": 3, "max_features": 40_000, "sublinear_tf": True},
    "char": {"analyzer": "char_wb", "ngram_range": (2, 4), "min_df": 5, "max_features": 60_000,
             "sublinear_tf": True},
}
C = 4.0
DOCS_VERSION = 1  # bump when load_docs changes how documents are built; it is part of the cache key


def load_docs(path: Path = POSTS, mask_digits: bool = True) -> pd.Series:
    """One document per account: its posts in post_ix order, joined by a space."""
    posts = pd.read_csv(path, usecols=POST_COLS, dtype={ID_COL: str})
    posts["text"] = posts["text"].fillna("")
    if mask_digits:
        posts["text"] = posts["text"].str.replace(r"\d", "0", regex=True)
    posts = posts.sort_values([ID_COL, "post_ix"], kind="stable")
    return posts.groupby(ID_COL)["text"].agg(" ".join)


def make_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(**VECTORIZER_PARAMS["word"])),
        ("char", TfidfVectorizer(**VECTORIZER_PARAMS["char"])),
    ])


def make_text_lr() -> LogisticRegression:
    return LogisticRegression(C=C, solver="liblinear")


@dataclass
class TextMatrix:
    """TF-IDF rows for every account with posts; vocabulary fit on fit_ids only."""

    X: sp.csr_matrix
    ids: np.ndarray
    feature_names: np.ndarray
    key: str

    def __post_init__(self) -> None:
        self._pos = pd.Series(np.arange(len(self.ids)), index=self.ids)

    def rows(self, ids) -> sp.csr_matrix:
        """Rows for these accounts, in the order given. Raises if an id has no posts."""
        ids = pd.Index(ids).astype(str)  # ids are strings like "B1005545"
        missing = ids.difference(self._pos.index)
        if len(missing):
            raise KeyError(f"{len(missing)} account(s) have no posts, e.g. {list(missing[:3])}")
        return self.X[self._pos.loc[ids].to_numpy()]


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_key(posts_path: Path, fit_ids, mask_digits: bool) -> str:
    """Changes when the posts, the vocabulary rows, the settings or sklearn change."""
    payload = {
        "posts": _file_sha(posts_path),
        "fit_ids": sorted(map(str, fit_ids)),
        "params": VECTORIZER_PARAMS,
        "mask_digits": mask_digits,
        "docs_version": DOCS_VERSION,
        "sklearn": sklearn.__version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    """The key is in both file names, so a matrix can only ever be paired with its own meta."""
    return cache_dir / f"tfidf_{key}.npz", cache_dir / f"tfidf_{key}.json"


def _load_cache(cache_dir: Path, key: str) -> TextMatrix | None:
    X_path, meta_path = _cache_paths(cache_dir, key)
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("key") != key:
            return None
        X = sp.load_npz(X_path).tocsr()
        ids, names = np.array(meta["ids"], dtype=object), np.array(meta["feature_names"], dtype=object)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, zipfile.BadZipFile):
        # missing, damaged or hand-edited: rebuild
        return None
    if X.shape != (len(ids), len(names)):
        return None
    return TextMatrix(X=X, ids=ids, feature_names=names, key=key)


def _save_cache(cache_dir: Path, tm: TextMatrix) -> None:
    """Best effort: write to per-process temp files, then rename; a failure only warns.

    The matrix is renamed in first and the meta last, so a reader never finds a meta
    without its matrix. Older keys' files are removed afterwards (each is about 50 MB).
    """
    X_path, meta_path = _cache_paths(cache_dir, tm.key)
    tmp_paths: list[str] = []
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_X = tempfile.mkstemp(dir=cache_dir, suffix=".npz.tmp")
        tmp_paths.append(tmp_X)
        with os.fdopen(fd, "wb") as f:
            sp.save_npz(f, tm.X)
        fd, tmp_meta = tempfile.mkstemp(dir=cache_dir, suffix=".json.tmp")
        tmp_paths.append(tmp_meta)
        with os.fdopen(fd, "w") as f:
            json.dump({"key": tm.key, "ids": tm.ids.tolist(), "feature_names": tm.feature_names.tolist()}, f)
        os.replace(tmp_X, X_path)
        os.replace(tmp_meta, meta_path)
    except OSError as e:
        warnings.warn(f"TF-IDF cache not saved to {cache_dir} ({e}); the next run rebuilds it", stacklevel=3)
        return
    finally:
        for tmp in tmp_paths:
            Path(tmp).unlink(missing_ok=True)
    for old in cache_dir.glob("tfidf_*"):
        if old not in (X_path, meta_path):
            try:
                old.unlink(missing_ok=True)
            except OSError:  # pruning is housekeeping; never fail a finished build over it
                pass


def build_matrix(
    fit_ids,
    posts_path: Path = POSTS,
    cache_dir: Path | None = CACHE_DIR,
    mask_digits: bool = True,
) -> TextMatrix:
    """TF-IDF for every account, vocabulary fit on fit_ids' text only. Cached when cache_dir is set."""
    key = cache_key(posts_path, fit_ids, mask_digits)
    if cache_dir is not None and (tm := _load_cache(Path(cache_dir), key)) is not None:
        return tm
    docs = load_docs(posts_path, mask_digits)
    fit_docs = docs.reindex(pd.Index(fit_ids).astype(str))
    if fit_docs.isna().any():
        raise KeyError(f"{int(fit_docs.isna().sum())} fit account(s) have no posts")
    vec = make_vectorizer().fit(fit_docs.to_numpy())
    tm = TextMatrix(
        X=sp.csr_matrix(vec.transform(docs.to_numpy())),
        ids=docs.index.to_numpy(dtype=object),
        feature_names=vec.get_feature_names_out().astype(object),
        key=key,
    )
    if cache_dir is not None:
        _save_cache(Path(cache_dir), tm)
    return tm


def fit_text_model(tm: TextMatrix, ids, y) -> LogisticRegression:
    """Text LR on these labelled accounts (loop.py refits through this)."""
    return make_text_lr().fit(tm.rows(ids), np.asarray(y))


def score(model: LogisticRegression, tm: TextMatrix, ids) -> np.ndarray:
    """text_score = p(teen) for these accounts."""
    return model.predict_proba(tm.rows(ids))[:, 1]


def account_top_words(model: LogisticRegression, tm: TextMatrix, ids, n: int = 3) -> list[list[str]]:
    """Per account, the n words that push its score most toward teen (for explain.py / A4).

    Contribution = TF-IDF value x coefficient. Only word features are used, since
    character n-grams mean little to a reviewer; the "word__" prefix is dropped.
    """
    is_word = np.char.startswith(tm.feature_names.astype(str), "word__")
    contrib = tm.rows(ids)[:, is_word].multiply(model.coef_[0][is_word]).tocsr()
    names = np.char.replace(tm.feature_names[is_word].astype(str), "word__", "", count=1)
    out = []
    for i in range(contrib.shape[0]):
        row = contrib.getrow(i)
        order = np.argsort(-row.data)
        out.append([str(names[row.indices[j]]) for j in order[:n] if row.data[j] > 0])
    return out


def oof_text_score(tm: TextMatrix, train: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Out-of-fold text_score for every train row, using the shared cv_folds."""
    ids, y = train[ID_COL].to_numpy(), train[TARGET].to_numpy()
    out = np.full(len(train), np.nan)
    for fit_idx, val_idx in cv_folds(train, k=k):
        out[val_idx] = score(fit_text_model(tm, ids[fit_idx], y[fit_idx]), tm, ids[val_idx])
    return out


@dataclass
class TextModelResult:
    model: LogisticRegression
    tm: TextMatrix
    oof: np.ndarray  # train rows, in train order
    test_score: np.ndarray  # test rows, in test order
    threshold: float
    cap: float
    rows: list[dict]

    def top_words(self, n: int = 20) -> dict[str, list[str]]:
        """Strongest teen and adult features of the full-train model (for sanity.txt / explain.py)."""
        order = np.argsort(self.model.coef_[0])
        return {
            "teen": self.tm.feature_names[order[::-1][:n]].tolist(),
            "adult": self.tm.feature_names[order[:n]].tolist(),
        }


def text_model(
    train: pd.DataFrame, test: pd.DataFrame, cap: float = DEFAULT_CAP, tm: TextMatrix | None = None
) -> TextModelResult:
    """Ladder row 6: OOF text_score on train, cutoff at the cap from OOF, test scored once."""
    tm = tm if tm is not None else build_matrix(train[ID_COL])
    y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    oof = oof_text_score(tm, train)
    t = cap_threshold(oof, y_tr, cap)
    model = fit_text_model(tm, train[ID_COL], y_tr)
    p_te = score(model, tm, test[ID_COL])
    stage = f"text_tfidf_cap{cap * 100:g}"
    rows = [
        eval_row(stage, "cv_oof", y_tr, (oof >= t).astype(int), oof),
        eval_row(stage, "test", y_te, (p_te >= t).astype(int), p_te),
    ]
    return TextModelResult(model=model, tm=tm, oof=oof, test_score=p_te, threshold=t, cap=cap, rows=rows)


def main() -> None:
    train, test = load_data(on_param_mismatch="error")
    res = text_model(train, test)
    pd.set_option("display.width", 120)
    print(pd.DataFrame(res.rows, columns=EVAL_COLS).round(3).to_string(index=False))
    print(f"\nTF-IDF matrix {res.tm.X.shape[0]} accounts x {res.tm.X.shape[1]} features "
          f"(vocabulary fit on {len(train)} train accounts only), cache key {res.tm.key}")
    print(f"text cutoff at {res.cap * 100:g}% cap (from OOF): {res.threshold:.3f}")
    print("cv_oof meets the cap by construction; test applies that cutoff to the full-train refit.")
    words = res.top_words(15)
    print("teen features: ", ", ".join(words["teen"]))
    print("adult features:", ", ".join(words["adult"]))


if __name__ == "__main__":
    main()
