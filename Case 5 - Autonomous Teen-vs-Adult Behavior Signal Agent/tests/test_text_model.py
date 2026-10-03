"""TF-IDF text model (step 7): docs, train-only vocabulary, cache, OOF score, real-data numbers.

Small tests use a synthetic posts file in tmp_path; the real-data tests cache to tmp_path too,
never to the repo's cache/. Pinned numbers depend on results/split.json and the sklearn version.
"""
import json

import numpy as np
import pandas as pd
import pytest

import softsignal.text_model as tmod
from softsignal.data import load_data
from softsignal.features import ID_COL, TARGET
from softsignal.metrics import auc

WORDS_A = "school lol homework class teacher bored summer haha"
WORDS_B = "work office meeting boss commute mortgage husband project"


def write_posts(path, n=12, extra_cols=True):
    """n accounts, 2 posts each; even ids write like teens, odd like adults; the last 4 say 'zebra'."""
    rows = []
    for i in range(n):
        words = WORDS_A if i % 2 == 0 else WORDS_B
        tail = " zebra zebra" if i >= n - 4 else ""
        for ix in (1, 0):  # written out of order on purpose
            row = {ID_COL: f"B{i:03d}", "post_ix": ix, "text": f"post{ix} {words} 2003{tail}"}
            if extra_cols:
                row.update({"age": 15 if i % 2 == 0 else 30, "is_teen": i % 2 == 0, "gender": "x", "job": "y"})
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def posts(tmp_path):
    return write_posts(tmp_path / "posts.csv")


@pytest.fixture
def small_params(monkeypatch):
    """The real min_df values would empty a 12-account vocabulary."""
    monkeypatch.setattr(tmod, "VECTORIZER_PARAMS", {
        "word": {"ngram_range": (1, 1), "min_df": 1, "sublinear_tf": True},
        "char": {"analyzer": "char_wb", "ngram_range": (2, 3), "min_df": 1, "sublinear_tf": True},
    })


# --- documents ---------------------------------------------------------------------------

def test_load_docs_joins_posts_in_post_ix_order(posts):
    docs = tmod.load_docs(posts, mask_digits=False)
    assert len(docs) == 12
    assert docs["B000"].startswith("post0 ") and " post1 " in docs["B000"]


def test_load_docs_masks_digits_by_default(posts):
    doc = tmod.load_docs(posts)["B000"]
    assert "2003" not in doc and "0000" in doc and "post0" in doc


def test_load_docs_needs_only_id_ix_text(tmp_path):
    """Works on a file with no age / is_teen / gender / job columns: they are never read."""
    docs = tmod.load_docs(write_posts(tmp_path / "p.csv", extra_cols=False))
    assert len(docs) == 12


def test_load_docs_reads_only_id_ix_text(posts, monkeypatch):
    seen = {}
    real_read = pd.read_csv

    def spy(*a, **k):
        seen.update(k)
        return real_read(*a, **k)
    monkeypatch.setattr(tmod.pd, "read_csv", spy)
    tmod.load_docs(posts)
    assert seen["usecols"] == [ID_COL, "post_ix", "text"]


def test_load_docs_empty_text_is_empty_string(tmp_path):
    path = tmp_path / "p.csv"
    pd.DataFrame({ID_COL: ["B1", "B1"], "post_ix": [0, 1], "text": ["hi", None]}).to_csv(path, index=False)
    assert tmod.load_docs(path)["B1"] == "hi "


# --- matrix, vocabulary and cache --------------------------------------------------------

def test_vocabulary_comes_from_fit_ids_only(posts, small_params, tmp_path):
    fit = [f"B{i:03d}" for i in range(8)]  # zebra only appears in B008-B011
    tm = tmod.build_matrix(fit, posts_path=posts, cache_dir=None)
    assert not [f for f in tm.feature_names if "zeb" in f]  # neither word__zebra nor char__zeb
    assert tm.X.shape[0] == 12  # every account is still transformed


def test_rows_follow_requested_order_and_reject_unknown_ids(posts, small_params):
    tm = tmod.build_matrix([f"B{i:03d}" for i in range(12)], posts_path=posts, cache_dir=None)
    got = tm.rows(["B001", "B000"])  # an adult then a teen: their rows differ
    assert (tm.rows(["B000"]) != tm.rows(["B001"])).nnz > 0
    assert (got[0] != tm.X[1]).nnz == 0 and (got[1] != tm.X[0]).nnz == 0
    with pytest.raises(KeyError, match="no posts"):
        tm.rows(["B001", "NOPE"])


def test_fit_ids_without_posts_raise(posts, small_params):
    with pytest.raises(KeyError, match="no posts"):
        tmod.build_matrix(["B000", "NOPE"], posts_path=posts, cache_dir=None)


def test_cache_is_reused(posts, small_params, tmp_path, monkeypatch):
    fit = [f"B{i:03d}" for i in range(8)]
    first = tmod.build_matrix(fit, posts_path=posts, cache_dir=tmp_path / "c")
    monkeypatch.setattr(tmod, "make_vectorizer", lambda: pytest.fail("cache was not used"))
    second = tmod.build_matrix(fit, posts_path=posts, cache_dir=tmp_path / "c")
    assert (first.X != second.X).nnz == 0
    assert list(first.ids) == list(second.ids) and list(first.feature_names) == list(second.feature_names)


def test_cache_key_changes_with_fit_ids_posts_masking_and_docs_version(posts, small_params, tmp_path, monkeypatch):
    base = tmod.cache_key(posts, ["B000", "B001"], True)
    assert tmod.cache_key(posts, ["B001", "B000"], True) == base  # order does not matter
    assert tmod.cache_key(posts, ["B000"], True) != base
    assert tmod.cache_key(posts, ["B000", "B001"], False) != base
    other = write_posts(tmp_path / "other.csv", n=14)
    assert tmod.cache_key(other, ["B000", "B001"], True) != base
    monkeypatch.setattr(tmod, "DOCS_VERSION", tmod.DOCS_VERSION + 1)
    bumped = tmod.cache_key(posts, ["B000", "B001"], True)
    assert bumped != base
    monkeypatch.setattr(tmod, "VECTORIZER_PARAMS", {**tmod.VECTORIZER_PARAMS, "word": {"min_df": 2}})
    assert tmod.cache_key(posts, ["B000", "B001"], True) != bumped


@pytest.mark.parametrize(
    "damage", ["bad_json", "wrong_key", "missing_npz", "corrupt_npz", "truncated_npz", "shape", "meta_not_dict"]
)
def test_bad_cache_is_rebuilt(posts, small_params, tmp_path, damage):
    fit, cdir = [f"B{i:03d}" for i in range(8)], tmp_path / "c"
    good = tmod.build_matrix(fit, posts_path=posts, cache_dir=cdir)
    X_path, meta = tmod._cache_paths(cdir, good.key)
    if damage == "bad_json":
        meta.write_text("{not json")
    elif damage == "wrong_key":
        meta.write_text(json.dumps({**json.loads(meta.read_text()), "key": "stale"}))
    elif damage == "missing_npz":
        X_path.unlink()
    elif damage == "corrupt_npz":
        X_path.write_bytes(b"PK\x03\x04garbage")
    elif damage == "truncated_npz":
        X_path.write_bytes(X_path.read_bytes()[:40])
    elif damage == "meta_not_dict":
        meta.write_text("[1]")
    else:
        meta.write_text(json.dumps({**json.loads(meta.read_text()), "ids": ["B000"]}))
    again = tmod.build_matrix(fit, posts_path=posts, cache_dir=cdir)
    assert (good.X != again.X).nnz == 0 and list(again.ids) == list(good.ids)
    assert json.loads(meta.read_text())["key"] == good.key  # cache repaired


def test_failed_cache_write_warns_and_still_returns(posts, small_params, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(tmod.os, "replace", boom)
    with pytest.warns(UserWarning, match="not saved"):
        tm = tmod.build_matrix(["B000", "B001"], posts_path=posts, cache_dir=tmp_path / "c")
    assert tm.X.shape[0] == 12
    assert list((tmp_path / "c").iterdir()) == []  # temp files cleaned up


def test_new_key_replaces_old_cache_files(posts, small_params, tmp_path):
    cdir = tmp_path / "c"
    old = tmod.build_matrix(["B000", "B001"], posts_path=posts, cache_dir=cdir)
    new = tmod.build_matrix(["B000", "B001", "B002"], posts_path=posts, cache_dir=cdir)
    assert sorted(f.name for f in cdir.iterdir()) == sorted(p.name for p in tmod._cache_paths(cdir, new.key))
    assert old.key != new.key


def test_concurrent_builds_leave_a_valid_cache(posts, small_params, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    cdir, fit = tmp_path / "c", [f"B{i:03d}" for i in range(8)]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: tmod.build_matrix(fit, posts_path=posts, cache_dir=cdir), range(8)))
    assert all((r.X != results[0].X).nnz == 0 for r in results)
    assert tmod._load_cache(cdir, results[0].key) is not None
    assert not list(cdir.glob("*.tmp"))


def test_integer_ids_are_read_as_strings(posts, small_params):
    tm = tmod.build_matrix([f"B{i:03d}" for i in range(12)], posts_path=posts, cache_dir=None)
    assert tm.rows(np.array(["B001"], dtype=object)).shape[0] == 1
    with pytest.raises(KeyError, match="'7'"):
        tm.rows(np.array([7]))


# --- model on the small corpus -----------------------------------------------------------

def test_oof_scores_every_row_and_separates_styles(posts, small_params):
    ids = [f"B{i:03d}" for i in range(12)]
    train = pd.DataFrame({ID_COL: ids, TARGET: [1 - i % 2 for i in range(12)]})
    tm = tmod.build_matrix(ids, posts_path=posts, cache_dir=None)
    oof = tmod.oof_text_score(tm, train, k=3)
    assert not np.isnan(oof).any()
    assert auc(train[TARGET], oof) == 1.0


def test_account_top_words_are_the_accounts_own_teen_words(posts, small_params, tmp_path):
    ids = [f"B{i:03d}" for i in range(12)]
    tm = tmod.build_matrix(ids, posts_path=posts, cache_dir=None)
    model = tmod.fit_text_model(tm, ids, [1 - i % 2 for i in range(12)])
    teen, adult = tmod.account_top_words(model, tm, ["B000", "B001"], n=3)
    assert len(teen) == 3 and set(teen) <= set(WORDS_A.split())
    assert not any("__" in w for w in teen + adult)  # word features only, prefix stripped
    # an account using only two teen words must get those two, not the global top words
    one = pd.DataFrame({ID_COL: ["B099"], "post_ix": [0], "text": ["bored summer bored summer"]})
    mixed = tmp_path / "mixed.csv"
    pd.concat([pd.read_csv(posts, dtype={ID_COL: str}), one]).to_csv(mixed, index=False)
    tm2 = tmod.build_matrix(ids, posts_path=mixed, cache_dir=None)
    model2 = tmod.fit_text_model(tm2, ids, [1 - i % 2 for i in range(12)])
    assert set(tmod.account_top_words(model2, tm2, ["B099"], n=3)[0]) == {"bored", "summer"}
    assert not set(adult) & set(WORDS_B.split())  # adult-style words never count as teen evidence


# --- real data ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    train, test = load_data(on_param_mismatch="error")
    tm = tmod.build_matrix(train[ID_COL], cache_dir=tmp_path_factory.mktemp("cache"))
    return train, test, tmod.text_model(train, test, tm=tm)


def test_real_shapes_and_vocabulary(real):
    train, test, res = real
    assert res.tm.X.shape[0] == len(train) + len(test)
    assert len(res.oof) == len(train) and len(res.test_score) == len(test)
    assert not np.isnan(res.oof).any()
    digits = {c for c in "".join(res.tm.feature_names) if c.isdigit()}
    assert digits <= {"0"}  # every digit was masked to 0, so no typed age survives


def test_real_oof_meets_cap(real):
    _, _, res = real
    row = next(r for r in res.rows if r["eval_set"] == "cv_oof")
    assert row["ft"] <= res.cap


def test_real_auc_matches_measured(real):
    _, _, res = real
    oof_row, test_row = (next(r for r in res.rows if r["eval_set"] == s) for s in ("cv_oof", "test"))
    assert oof_row["auc"] == pytest.approx(0.888, abs=0.003)
    assert test_row["auc"] == pytest.approx(0.888, abs=0.003)


def test_test_labels_are_never_used(real):
    train, test, res = real
    flipped = test.assign(**{TARGET: 1 - test[TARGET]})
    again = tmod.text_model(train, flipped, tm=res.tm)
    assert np.allclose(again.test_score, res.test_score)
    assert again.threshold == res.threshold


def test_top_words_look_like_style(real):
    words = real[2].top_words(30)
    assert "word__school" in words["teen"] and "word__work" in words["adult"]
