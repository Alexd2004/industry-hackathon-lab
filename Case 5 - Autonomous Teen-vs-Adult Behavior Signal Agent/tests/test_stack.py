"""Level-2 stack (step 8): logit, features, nested OOF, Stack, fallback flag, real-data numbers.

Small tests use synthetic accounts and posts in tmp_path; the real-data tests cache to tmp_path
too, never to the repo's cache/. Pinned numbers depend on results/split.json and the sklearn version.
"""
import json

import numpy as np
import pandas as pd
import pytest

import softsignal.stack as stk
import softsignal.text_model as tmod
from softsignal.baselines import make_tabular_lr, oof_scores
from softsignal.data import cv_folds, load_data
from softsignal.features import FEATURE_COLS, FORBIDDEN, ID_COL, TARGET
from softsignal.metrics import auc

N = 60
WORDS_A = "school lol homework class teacher bored summer haha"
WORDS_B = "work office meeting boss commute mortgage husband project"


@pytest.fixture
def small_params(monkeypatch):
    """The real min_df values would empty a 60-account vocabulary."""
    monkeypatch.setattr(tmod, "VECTORIZER_PARAMS", {
        "word": {"ngram_range": (1, 1), "min_df": 1, "sublinear_tf": True},
        "char": {"analyzer": "char_wb", "ngram_range": (2, 3), "min_df": 1, "sublinear_tf": True},
    })


@pytest.fixture
def frame():
    """60 accounts, even ids are teens; the 16 columns carry a weak signal."""
    rng = np.random.default_rng(0)
    y = np.array([1 - i % 2 for i in range(N)])  # B000 is a teen
    df = pd.DataFrame(rng.normal(size=(N, len(FEATURE_COLS))), columns=FEATURE_COLS)
    df["pct_active_school_hours"] -= y * 0.8
    df.insert(0, ID_COL, [f"B{i:03d}" for i in range(N)])
    df[TARGET] = y
    df["age"] = np.where(y == 1, 15, 30)  # off-limits columns present on purpose
    df["job"] = "x"
    return df


@pytest.fixture
def posts(tmp_path):
    rows = []
    for i in range(N):
        words = WORDS_A if i % 2 == 0 else WORDS_B
        rows += [{ID_COL: f"B{i:03d}", "post_ix": ix, "text": f"post{ix} {words} {words.split()[ix]}"} for ix in (0, 1)]
    path = tmp_path / "posts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def tm(posts, frame, small_params):
    return tmod.build_matrix(frame[ID_COL], posts_path=posts, cache_dir=None)


# --- logit and features ------------------------------------------------------------------

def test_logit_is_finite_at_zero_and_one_and_symmetric():
    out = stk.logit([0.0, 1.0, 0.5, 0.2, 0.8])
    assert np.isfinite(out).all()
    assert out[2] == pytest.approx(0.0) and out[3] == pytest.approx(-out[4])
    assert out[1] == pytest.approx(-out[0]) and out[1] == pytest.approx(np.log((1 - 1e-6) / 1e-6))


def test_stack_features_puts_text_first_and_uses_only_the_allow_list(frame):
    X = stk.stack_features(frame, np.full(N, 0.5))
    assert list(X.columns) == [stk.TEXT_FEATURE] + FEATURE_COLS
    assert not FORBIDDEN & set(X.columns)
    assert list(stk.stack_features(frame).columns) == FEATURE_COLS


def test_stack_features_works_without_a_label_column(frame):
    X = stk.stack_features(frame.drop(columns=[TARGET, "age", "job"]), np.full(N, 0.3))
    assert X.shape == (N, 17)


# --- nested OOF --------------------------------------------------------------------------

def test_nested_oof_scores_every_row_deterministically(tm, frame):
    a, b = stk.nested_oof(tm, frame), stk.nested_oof(tm, frame)
    assert a.shape == (N,) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
    assert np.array_equal(a, b)


def test_nested_oof_text_models_never_see_the_rows_they_score(tm, frame, monkeypatch):
    """Per outer fold: k inner fits then one outer fit, none of them on the outer-val ids."""
    calls = []
    real = stk.fit_text_model

    def spy(tm_, ids, y):
        calls.append(set(ids))
        return real(tm_, ids, y)
    monkeypatch.setattr(stk, "fit_text_model", spy)
    monkeypatch.setattr(tmod, "fit_text_model", spy)  # oof_text_score fits through the module global
    stk.nested_oof(tm, frame, k=5)
    folds = cv_folds(frame, k=5)
    assert len(calls) == 5 * 6
    for f, (_, val_idx) in enumerate(folds):
        val_ids = set(frame[ID_COL].iloc[val_idx])
        assert all(not (c & val_ids) for c in calls[f * 6:(f + 1) * 6])


# --- Stack -------------------------------------------------------------------------------

def test_stack_fit_score_and_coefs(tm, frame):
    s = stk.Stack.fit(frame, tm=tm)
    p = s.score(frame)
    assert p.shape == (N,) and ((p >= 0) & (p <= 1)).all()
    assert auc(frame[TARGET], p) > 0.9
    coefs = s.coefs()
    assert set(coefs.index) == {stk.TEXT_FEATURE, *FEATURE_COLS}
    assert list(coefs.abs()) == sorted(coefs.abs(), reverse=True)
    assert coefs[stk.TEXT_FEATURE] > 0


def test_stack_fit_trains_level2_on_oof_text_scores_not_in_sample_ones(tm, frame, monkeypatch):
    calls = []
    real = stk.oof_text_score

    def spy(tm_, train, k=5):
        calls.append(train)
        return real(tm_, train, k)
    monkeypatch.setattr(stk, "oof_text_score", spy)
    s = stk.Stack.fit(frame, tm=tm)
    assert len(calls) == 1 and calls[0] is frame
    oof_logit = stk.logit(real(tm, frame))  # level 2 must have been fitted on exactly these inputs
    ref = stk._fit_level2(stk.stack_features(frame, real(tm, frame)), frame[TARGET])
    assert np.allclose(s.level2[-1].coef_, ref[-1].coef_) and np.isfinite(oof_logit).all()


def test_level2_never_receives_a_forbidden_column(tm, frame):
    s = stk.Stack.fit(frame, tm=tm)
    assert not FORBIDDEN & set(s.level2.feature_names_in_)


def test_explain_returns_top_signed_contributions_that_add_up(tm, frame):
    s = stk.Stack.fit(frame, tm=tm)
    top = s.explain(frame, n=3)
    assert len(top) == N and all(len(t) == 3 for t in top)
    for t in top:
        assert [abs(c) for _, c in t] == sorted((abs(c) for _, c in t), reverse=True)
    full = s.explain(frame.iloc[:5], n=17)  # every feature: contributions + intercept = the logit
    logit_p = stk.logit(s.score(frame.iloc[:5]))
    for row, lp in zip(full, logit_p):
        assert sum(c for _, c in row) + s.level2[-1].intercept_[0] == pytest.approx(lp, abs=1e-4)


def test_explain_names_match_level2_columns(tm, frame):
    names = {n for t in stk.Stack.fit(frame, tm=tm).explain(frame, n=2) for n, _ in t}
    assert names <= {stk.TEXT_FEATURE, *FEATURE_COLS}


# --- level 2 must not see the rows it scores --------------------------------------------

def test_nested_oof_level2_is_fit_only_on_the_outer_fit_rows(tm, frame, monkeypatch):
    fits = []
    real = stk._fit_level2

    def spy(X, y):
        fits.append((X, np.asarray(y)))
        return real(X, y)
    monkeypatch.setattr(stk, "_fit_level2", spy)
    stk.nested_oof(tm, frame, k=5)
    assert len(fits) == 5
    rows = lambda d: {tuple(r) for r in d[FEATURE_COLS].round(9).to_numpy()}  # noqa: E731  (rows are unique)
    for (X, y), (fit_idx, val_idx) in zip(fits, cv_folds(frame, k=5)):
        assert len(X) == len(y) == len(fit_idx)
        assert np.array_equal(y, frame[TARGET].to_numpy()[fit_idx])
        assert not rows(X) & rows(frame.iloc[val_idx])


def test_tabular_fallback_oof_is_the_out_of_fold_score_not_an_in_sample_one(frame, tmp_path):
    train, test = frame.iloc[:40].reset_index(drop=True), frame.iloc[40:].reset_index(drop=True)
    res = stk.stack(train, test, use_text=False, oof_path=tmp_path / "o.csv")
    assert np.array_equal(res.oof, oof_scores(train))
    in_sample = stk.Stack.fit(train, use_text=False).score(train)
    assert not np.allclose(res.oof, in_sample)


# --- vocabulary must come from train only -----------------------------------------------

def test_a_matrix_fit_on_test_text_is_rejected_everywhere(posts, small_params, frame):
    train = frame.iloc[:40].reset_index(drop=True)
    leaky = tmod.build_matrix(frame[ID_COL], posts_path=posts, cache_dir=None)  # fit on train and test
    with pytest.raises(ValueError, match="outside train"):
        stk.nested_oof(leaky, train)
    with pytest.raises(ValueError, match="outside train"):
        stk.Stack.fit(train, tm=leaky)
    with pytest.raises(ValueError, match="outside train"):
        stk.stack(train, frame.iloc[40:], tm=leaky, oof_path=None)


def test_a_matrix_fit_on_a_subset_of_train_is_accepted(posts, small_params, frame, tm):
    sub = tmod.build_matrix(frame[ID_COL].iloc[:30], posts_path=posts, cache_dir=None)
    stk.check_vocabulary(sub, frame[ID_COL])  # nothing from outside train, so no error
    stk.check_vocabulary(tm, frame[ID_COL])


def test_a_smaller_frame_is_fine_when_the_whole_train_set_is_given(tm, frame):
    """A refit on part of train (e.g. revealed labels) uses a matrix fit on all of train."""
    part = frame.iloc[:40].reset_index(drop=True)
    with pytest.raises(ValueError, match="outside train"):
        stk.nested_oof(tm, part)  # default: the frame itself is taken as train
    assert np.isfinite(stk.nested_oof(tm, part, train_ids=frame[ID_COL])).all()
    assert stk.Stack.fit(part, tm=tm, train_ids=frame[ID_COL]).use_text


def test_train_ids_cannot_hide_test_text(posts, small_params, frame):
    leaky = tmod.build_matrix(frame[ID_COL], posts_path=posts, cache_dir=None)
    with pytest.raises(ValueError, match="outside train"):
        stk.nested_oof(leaky, frame.iloc[:40], train_ids=frame[ID_COL].iloc[:40])


def test_matrix_from_cache_still_knows_its_fit_ids(posts, small_params, frame, tmp_path):
    ids = frame[ID_COL].iloc[:40]
    first = tmod.build_matrix(ids, posts_path=posts, cache_dir=tmp_path / "c")
    again = tmod.build_matrix(ids, posts_path=posts, cache_dir=tmp_path / "c")  # cache hit
    assert first.fit_ids == again.fit_ids == frozenset(ids)


def test_a_matrix_without_fit_ids_warns_instead_of_passing_silently(tm, frame):
    tm.fit_ids = None  # hand-built, or dataclasses.replace(tm, fit_ids=None)
    with pytest.warns(UserWarning, match="no fit_ids"):
        stk.check_vocabulary(tm, frame[ID_COL])
    with pytest.warns(UserWarning, match="no fit_ids"):
        stk.nested_oof(tm, frame)


# --- fallback flag -----------------------------------------------------------------------

def test_use_text_false_is_the_tabular_lr_and_needs_no_text_model(frame, monkeypatch):
    monkeypatch.setattr(stk, "build_matrix", lambda *a, **k: pytest.fail("text matrix was built"))
    s = stk.Stack.fit(frame, use_text=False)
    assert s.text_model is None and s.tm is None
    ref = make_tabular_lr().fit(frame[FEATURE_COLS], frame[TARGET])
    assert np.allclose(s.score(frame), ref.predict_proba(frame[FEATURE_COLS])[:, 1])
    assert stk.TEXT_FEATURE not in s.coefs().index
    assert all(n != stk.TEXT_FEATURE for t in s.explain(frame, n=3) for n, _ in t)


def test_stack_fallback_skips_the_text_model_and_names_its_stage(frame, tmp_path, monkeypatch):
    monkeypatch.setattr(stk, "build_matrix", lambda *a, **k: pytest.fail("text matrix was built"))
    res = stk.stack(frame.iloc[:40], frame.iloc[40:], use_text=False, oof_path=tmp_path / "o.csv")
    assert {r["stage"] for r in res.rows} == {"stack_tabular_only_cap15"}


# --- stack(): rows and the OOF file ------------------------------------------------------

def test_stack_rows_and_oof_file(posts, small_params, frame, tmp_path):
    train, test = frame.iloc[:40].reset_index(drop=True), frame.iloc[40:].reset_index(drop=True)
    tm = tmod.build_matrix(train[ID_COL], posts_path=posts, cache_dir=None)
    out = tmp_path / "sub" / "stack_oof.csv"
    res = stk.stack(train, test, tm=tm, oof_path=out)
    assert [(r["stage"], r["eval_set"]) for r in res.rows] == [("stack_cap15", "cv_oof"), ("stack_cap15", "test")]
    saved = pd.read_csv(out, dtype={ID_COL: str})
    assert list(saved.columns) == [ID_COL, "stack_oof"] and list(saved[ID_COL]) == list(train[ID_COL])
    assert np.allclose(saved["stack_oof"], res.oof)
    assert res.rows[0]["ft"] <= 0.15 + 1e-9  # the cutoff is picked from this very OOF at the cap
    assert res.test_score.shape == (len(test),)


def test_oof_path_none_writes_nothing(posts, small_params, frame, monkeypatch):
    train = frame.iloc[:40].reset_index(drop=True)
    tm = tmod.build_matrix(train[ID_COL], posts_path=posts, cache_dir=None)
    monkeypatch.setattr(stk, "write_oof", lambda *a, **k: pytest.fail("OOF file was written"))
    stk.stack(train, frame.iloc[40:], tm=tm, oof_path=None)


# --- real data ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    train, test = load_data(on_param_mismatch="error")
    cache = tmp_path_factory.mktemp("cache")
    tm = tmod.build_matrix(train[ID_COL], cache_dir=cache)
    out = cache / "stack_oof.csv"
    return train, test, stk.stack(train, test, tm=tm, oof_path=out), out


def test_real_oof_file_has_2100_honest_rows(real):
    train, _, res, out = real
    saved = pd.read_csv(out, dtype={ID_COL: str})
    assert len(saved) == 2100 and list(saved[ID_COL]) == list(train[ID_COL])
    assert saved["stack_oof"].between(0, 1).all()


def test_real_auc_clears_the_gate(real):
    train, test, res, _ = real
    oof_auc, test_auc = auc(train[TARGET], res.oof), auc(test[TARGET], res.test_score)
    assert oof_auc >= 0.94 and test_auc >= 0.94  # go/no-go gate (Combined Plan section 6)
    assert oof_auc == pytest.approx(0.951, abs=0.005) and test_auc == pytest.approx(0.949, abs=0.005)


def test_real_coefficient_signs(real):
    """text_score dominates; the plan's other expected signs hold for the tabular fallback (below)."""
    coefs = real[2].stack.coefs()
    assert coefs.index[0] == stk.TEXT_FEATURE and coefs[stk.TEXT_FEATURE] > 1.5
    assert coefs["pct_active_school_hours"] < 0 and coefs["slang_emoji_rate"] >= 0
    assert abs(coefs["avg_word_len"]) < 0.2  # text_score absorbs it: tabular-only it is about -0.7
    assert not FORBIDDEN & set(coefs.index)


def test_real_tabular_fallback_has_the_plans_expected_signs(real):
    """Combined Plan step 8 signs (WP: avg_word_len -0.63, slang +0.62, school hours -0.52) are tabular-LR values."""
    train = real[0]
    coefs = stk.Stack.fit(train, use_text=False).coefs()
    assert coefs["avg_word_len"] < -0.4 and coefs["slang_emoji_rate"] > 0.4
    assert coefs["pct_active_school_hours"] < -0.3


def test_real_run_is_deterministic(real):
    train, test, res, _ = real
    again = stk.Stack.fit(train, tm=res.stack.tm).score(test)
    assert np.allclose(again, res.test_score)


# --- guard messages and warning locations ------------------------------------------------

def test_the_error_for_a_smaller_frame_names_train_ids(tm, frame):
    with pytest.raises(ValueError, match="train_ids="):
        stk.nested_oof(tm, frame.iloc[:40].reset_index(drop=True))


def test_stack_accepts_train_ids_for_a_smaller_frame(tm, frame, tmp_path):
    train, test = frame.iloc[:40].reset_index(drop=True), frame.iloc[40:].reset_index(drop=True)
    with pytest.raises(ValueError, match="outside train"):
        stk.stack(train, test, tm=tm, oof_path=None)
    res = stk.stack(train, test, tm=tm, oof_path=None, train_ids=frame[ID_COL])
    assert len(res.oof) == 40 and np.isfinite(res.oof).all()


@pytest.mark.parametrize("call", ["check", "nested_oof", "fit", "stack"])
def test_missing_fit_ids_warning_points_at_the_callers_line_once(tm, frame, call):
    tm.fit_ids = None
    calls = {
        "check": lambda: stk.check_vocabulary(tm, frame[ID_COL]),
        "nested_oof": lambda: stk.nested_oof(tm, frame),
        "fit": lambda: stk.Stack.fit(frame, tm=tm),
        "stack": lambda: stk.stack(frame.iloc[:40].reset_index(drop=True), frame.iloc[40:], tm=tm,
                                   oof_path=None, train_ids=frame[ID_COL]),
    }
    with pytest.warns(UserWarning, match="no fit_ids") as rec:
        calls[call]()
    got = [w for w in rec if "no fit_ids" in str(w.message)]
    assert len(got) == 1 and got[0].filename == __file__


def test_main_explains_why_test_false_teen_can_exceed_the_cap(capsys, monkeypatch):
    class Fake:
        rows, oof, test_score, threshold, cap = [], np.array([0.5]), np.array([0.5]), 0.4, 0.15
        stack = type("S", (), {"coefs": staticmethod(lambda: pd.Series({"a": 1.0}))})()
    monkeypatch.setattr(stk, "load_data", lambda **k: (pd.DataFrame({TARGET: [0, 1]}), pd.DataFrame({TARGET: [0, 1]})))
    monkeypatch.setattr(stk, "stack", lambda *a, **k: Fake)
    monkeypatch.setattr(stk, "auc", lambda *a: 0.9)
    stk.main()
    assert "test false-teen can land above it" in capsys.readouterr().out


# --- cache key: a stale OOF cache must be detectable --------------------------------------

def tm_key_of(posts, frame):
    return tmod.cache_key(posts, frame[ID_COL], True)


def test_cache_key_is_deterministic_and_a_sha256_hex(frame, posts):
    key = tm_key_of(posts, frame)
    a = stk.oof_cache_key(frame, tm_key=key)
    assert a == stk.oof_cache_key(frame.copy(), tm_key=key)
    assert len(a) == 64


def test_cache_key_changes_with_label_feature_or_id(frame, posts):
    key = tm_key_of(posts, frame)
    base = stk.oof_cache_key(frame, tm_key=key)
    flipped = frame.copy()
    flipped.loc[0, TARGET] = 1 - flipped.loc[0, TARGET]
    moved = frame.copy()
    moved.loc[0, FEATURE_COLS[0]] += 1.0
    renamed = frame.copy()
    renamed.loc[0, ID_COL] = "B999"
    for other in (flipped, moved, renamed):
        assert stk.oof_cache_key(other, tm_key=key) != base


def test_cache_key_ignores_columns_outside_the_allow_list(frame, posts):
    key = tm_key_of(posts, frame)
    other = frame.copy()
    other["age"] = 99
    assert stk.oof_cache_key(other, tm_key=key) == stk.oof_cache_key(frame, tm_key=key)


def test_cache_key_changes_when_a_feature_column_is_renamed_consistently(frame, posts, monkeypatch):
    key = tm_key_of(posts, frame)
    base = stk.oof_cache_key(frame, tm_key=key)
    renamed = frame.rename(columns={FEATURE_COLS[0]: "zz_renamed"})
    monkeypatch.setattr(stk, "FEATURE_COLS", ["zz_renamed", *FEATURE_COLS[1:]])
    assert stk.oof_cache_key(renamed, tm_key=key) != base  # same values, new name


def test_cache_key_follows_the_text_matrix_key_not_the_default_posts(frame, posts, tmp_path):
    base = stk.oof_cache_key(frame, tm_key=tm_key_of(posts, frame))
    edited = tmp_path / "posts2.csv"
    edited.write_text(posts.read_text() + "B000,9,extra post\n")
    assert stk.oof_cache_key(frame, tm_key=tm_key_of(edited, frame)) != base
    assert stk.oof_cache_key(frame, tm_key=tm_key_of(posts, frame.iloc[:-1])) != base  # other fit ids
    assert stk.oof_cache_key(frame, tm_key=tmod.cache_key(posts, frame[ID_COL], False)) != base  # mask_digits


def test_cache_key_default_matches_the_default_matrix_key(frame):
    default = tmod.cache_key(tmod.POSTS, frame[ID_COL], True)
    assert stk.oof_cache_key(frame) == stk.oof_cache_key(frame, tm_key=default)


def test_cache_key_changes_with_mode_text_params_and_versions(frame, posts, monkeypatch):
    key = tm_key_of(posts, frame)
    base = stk.oof_cache_key(frame, tm_key=key)
    assert stk.oof_cache_key(frame, use_text=False) != base
    monkeypatch.setattr(tmod, "C", tmod.C + 1.0)
    assert stk.oof_cache_key(frame, tm_key=key) != base
    monkeypatch.undo()
    monkeypatch.setattr(np, "__version__", "0.0.0")
    assert stk.oof_cache_key(frame, tm_key=key) != base
    monkeypatch.undo()
    monkeypatch.setattr(stk.scipy, "__version__", "0.0.0")
    assert stk.oof_cache_key(frame, tm_key=key) != base
    monkeypatch.undo()
    monkeypatch.setattr(stk.platform, "python_version", lambda: "0.0.0")
    assert stk.oof_cache_key(frame, tm_key=key) != base


def test_tabular_cache_key_does_not_depend_on_posts_or_text_params(frame, monkeypatch):
    base = stk.oof_cache_key(frame, use_text=False)
    monkeypatch.setattr(tmod, "C", tmod.C + 1.0)
    assert stk.oof_cache_key(frame, use_text=False) == base
    assert stk.oof_cache_key(frame, use_text=False, tm_key="anything") == base


def altered(*args, **kwargs):  # a different body and signature from every function it replaces
    return "altered"


KEY_FUNCTIONS = [
    (stk, "nested_oof"), (stk, "_fit_level2"), (stk, "stack_features"), (stk, "logit"),
    (stk, "make_tabular_lr"), (stk, "cv_folds"), (stk, "check_vocabulary"), (stk, "oof_text_score"),
    (stk, "fit_text_model"), (tmod, "score"), (tmod, "make_text_lr"), (tmod, "make_vectorizer"),
    (tmod, "load_docs"), (tmod, "build_matrix"), (tmod.TextMatrix, "rows"),
]


@pytest.mark.parametrize("owner,name", KEY_FUNCTIONS, ids=[n for _, n in KEY_FUNCTIONS])
def test_cache_key_changes_when_any_scoring_function_changes(frame, posts, monkeypatch, owner, name):
    key = tm_key_of(posts, frame)
    base = stk.oof_cache_key(frame, tm_key=key)
    monkeypatch.setattr(owner, name, altered)
    assert stk.oof_cache_key(frame, tm_key=key) != base


def test_tabular_cache_key_changes_when_oof_scores_changes(frame, monkeypatch):
    base = stk.oof_cache_key(frame, use_text=False)
    monkeypatch.setattr(stk, "oof_scores", altered)
    assert stk.oof_cache_key(frame, use_text=False) != base


def test_every_key_function_has_a_test(frame):
    # a function added to the key without a case in KEY_FUNCTIONS would go untested otherwise
    assert len(stk._key_functions(True)) == len(KEY_FUNCTIONS)


def test_stack_writes_the_meta_file_with_the_matrix_key(posts, small_params, frame, tmp_path):
    train, test = frame.iloc[:40].reset_index(drop=True), frame.iloc[40:].reset_index(drop=True)
    tm = tmod.build_matrix(train[ID_COL], posts_path=posts, cache_dir=None)
    out = tmp_path / "stack_oof.csv"
    stk.stack(train, test, tm=tm, oof_path=out)
    meta = json.loads(stk.meta_path(out).read_text())
    assert stk.meta_path(out).name == "stack_oof.meta.json"
    assert meta["use_text"] is True and meta["n"] == len(train)
    assert meta["csv_sha256"] == stk.csv_sha256(out)
    assert meta["cache_key"] == stk.oof_cache_key(train, tm_key=tm.key)
    # the matrix came from the synthetic posts, not the default file, and the key says so
    assert meta["cache_key"] != stk.oof_cache_key(train)


def test_stack_writes_nothing_by_default(posts, small_params, frame, monkeypatch):
    train, test = frame.iloc[:40].reset_index(drop=True), frame.iloc[40:].reset_index(drop=True)
    tm = tmod.build_matrix(train[ID_COL], posts_path=posts, cache_dir=None)
    monkeypatch.setattr(stk, "write_oof", lambda *a, **k: pytest.fail("the default call wrote an OOF file"))
    stk.stack(train, test, tm=tm)


def test_write_oof_removes_the_old_meta_file_if_the_csv_write_fails(frame, tmp_path, monkeypatch):
    out = tmp_path / "o.csv"
    stk.write_oof(frame, np.zeros(len(frame)), out, use_text=False)
    assert stk.meta_path(out).exists()

    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    with pytest.raises(OSError):
        stk.write_oof(frame, np.zeros(len(frame)), out, use_text=False)
    assert not stk.meta_path(out).exists()  # no old meta file left to bless a half-written csv


def test_csv_sha256_changes_with_one_byte(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes(b"blogger_id,stack_oof\nB1,0.5\n")
    a = stk.csv_sha256(p)
    p.write_bytes(b"blogger_id,stack_oof\nB1,0.6\n")
    assert stk.csv_sha256(p) != a and len(a) == 64
