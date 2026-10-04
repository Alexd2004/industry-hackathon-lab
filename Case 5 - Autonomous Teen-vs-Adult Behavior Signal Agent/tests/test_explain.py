"""Explanations and the ranked list (step 10).

Real-data tests fit the full-train stack once per module (TF-IDF cached to tmp_path, never
the repo's cache/). Thresholds here are fixed numbers: band policy belongs to policy.py.
"""
import numpy as np
import pandas as pd
import pytest

import softsignal.explain as ex
from softsignal.data import load_data
from softsignal.features import ACTIVITY_COLS, FEATURE_COLS, FORBIDDEN, ID_COL, TARGET
from softsignal.metrics import RANKED_COLS, top_k_precision
from softsignal.stack import TEXT_FEATURE, Stack, logit
from softsignal.text_model import account_top_words, build_matrix

T_SOFT, T_VERIFY = 0.3, 0.6


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    train, test = load_data(on_param_mismatch="error")
    tm = build_matrix(train[ID_COL], cache_dir=tmp_path_factory.mktemp("cache"))
    return train, test, Stack.fit(train, tm=tm)


@pytest.fixture(scope="module")
def ranked(fitted):
    _, test, model = fitted
    return ex.ranked(model, test, T_SOFT, T_VERIFY)


# --- exactness of the explanation -------------------------------------------------------

def test_contributions_add_up_to_the_score(fitted):
    _, test, model = fitted
    exp = ex.contributions(model, test)
    assert np.allclose(exp.intercept + exp.contrib.sum(axis=1), logit(model.score(test)), atol=1e-6)


def test_top_terms_follow_stack_explain_order(fitted):
    _, test, model = fitted
    rows = test.iloc[:20]
    for ours, theirs in zip(ex.top_terms(ex.contributions(model, rows)), model.explain(rows)):
        assert [(f, round(v, 9)) for f, v, _ in ours] == [(f, round(v, 9)) for f, v in theirs]
        assert [c.rsplit(" ", 1)[1] for _, _, c in ours] == [f"{v:+.2f}" for _, v in theirs]


def test_every_level2_feature_has_a_phrase():
    assert set(FEATURE_COLS) | {TEXT_FEATURE} == set(ex.PHRASES)
    assert not FORBIDDEN & set(ex.PHRASES)


@pytest.mark.parametrize("z,expected", [(1.2, "active in school hours"), (0.0, "active in school hours"),
                                        (-0.4, "quiet in school hours")])
def test_chip_phrase_follows_the_accounts_side_of_average(z, expected):
    assert ex.chip("pct_active_school_hours", -0.5 * z, z) == f"{expected} {-0.5 * z:+.2f}"


def test_text_chip_follows_the_text_score_not_the_average():
    # a text score just above 0.5 (logit > 0) can sit below the training mean (z < 0)
    assert ex.chip(TEXT_FEATURE, -0.05, z=-0.02, raw=0.1).startswith("writes like a teen")
    assert ex.chip(TEXT_FEATURE, 0.05, z=0.02, raw=-0.1).startswith("writes like an adult")


def test_no_text_chip_contradicts_the_text_score(ranked, fitted):
    _, test, model = fitted
    p_text = 1 / (1 + np.exp(-model._features(test)[TEXT_FEATURE]))
    by_id = dict(zip(test[ID_COL].astype(str), p_text))
    for _, r in ranked.iterrows():
        for i in (1, 2, 3):
            if r[f"f{i}"] == TEXT_FEATURE:
                assert r[f"c{i}"].startswith("writes like a teen") == (by_id[r[ID_COL]] >= 0.5)


# --- bands, reasons, re-banding ----------------------------------------------------------

def test_bands_boundaries():
    got = ex.bands([0.1, 0.3, 0.59, 0.6, 0.9], T_SOFT, T_VERIFY)
    assert got.tolist() == ["none", "soft", "soft", "verify", "verify"]


def test_soft_band_is_empty_when_t_soft_is_not_below_t_verify():
    assert set(ex.bands(np.linspace(0, 1, 50), 0.39, 0.37)) == {"none", "verify"}


@pytest.mark.parametrize("t", [(np.nan, 0.6), (0.3, np.inf)])
def test_bands_reject_non_finite_thresholds(t):
    with pytest.raises(ValueError, match="finite"):
        ex.bands([0.5], *t)


def test_bands_reject_nan_scores():
    with pytest.raises(ValueError, match="NaN"):
        ex.bands([0.5, np.nan], T_SOFT, T_VERIFY)


def test_reason_sentence():
    chips = ["writes like a teen +7.98", "quiet in school hours +1.00", ""]
    assert ex.reason(0.973, "verify", chips, "im, lol") == (
        "Request verification (score 0.97): writes like a teen +7.98; quiet in school hours +1.00."
        " Teen-leaning words: im, lol.")
    assert "words" not in ex.reason(0.1, "none", ["x +0.10"], "friends, today")  # unflagged: no word evidence


def test_rebanding_a_frame_keeps_reasons_consistent(fitted):
    _, test, model = fitted
    frame = ex.explain_frame(model, test.iloc[:200])
    for t_soft, t_verify in [(0.3, 0.6), (0.2, 0.9), (0.5, 0.5)]:
        out = ex.apply_bands(frame, t_soft, t_verify)
        assert (out["band"] == ex.bands(out["score"], t_soft, t_verify)).all()
        assert out.apply(lambda r: r["reason"].startswith(ex.ACTIONS[r["band"]] + " ("), axis=1).all()
    assert "band" not in frame.columns  # the frame itself stays band-free


# --- ranked list -------------------------------------------------------------------------

def test_ranked_shape_order_and_ids(ranked, fitted):
    _, test, _ = fitted
    assert list(ranked.columns) == RANKED_COLS
    assert ranked["rank"].tolist() == list(range(1, len(test) + 1))
    assert ranked["score"].is_monotonic_decreasing
    assert sorted(ranked[ID_COL]) == sorted(test[ID_COL].astype(str))


def test_ranked_numbers_match_chips_and_contributions(ranked, fitted):
    _, test, model = fitted
    exp = ex.contributions(model, test)
    contrib = exp.contrib.set_index(test[ID_COL].astype(str).to_numpy())
    for _, r in ranked.head(50).iterrows():
        for i in (1, 2, 3):
            assert r[f"v{i}"] == pytest.approx(contrib.loc[r[ID_COL], r[f"f{i}"]])
            assert r[f"c{i}"].endswith(f"{r[f'v{i}']:+.2f}")


def test_ranked_words_belong_to_their_own_account(ranked, fitted):
    _, _, model = fitted
    top = ranked.head(20)
    expected = account_top_words(model.text_model, model.tm, top[ID_COL], n=3)
    assert top["words"].tolist() == [", ".join(w) for w in expected]
    flagged = top[top["band"] != "none"]
    assert flagged.apply(lambda r: r["words"] in r["reason"], axis=1).all()


def test_ranked_bands_actions_and_reasons_agree(ranked):
    assert (ranked["band"] == ex.bands(ranked["score"], T_SOFT, T_VERIFY)).all()
    assert (ranked["action"] == ranked["band"].map(ex.ACTIONS)).all()
    assert ranked[["c1", "c2", "c3"]].ne("").all().all()
    assert ranked.apply(lambda r: r["reason"].startswith(r["action"]) and r["c1"] in r["reason"], axis=1).all()


def test_ranked_never_carries_labels(ranked, fitted):
    assert not (FORBIDDEN | {"is_teen"}) & set(ranked.columns)
    _, test, model = fitted
    flipped = test.assign(**{TARGET: 1 - test[TARGET]})
    pd.testing.assert_frame_equal(ex.ranked(model, flipped, T_SOFT, T_VERIFY), ranked)


def test_ranked_is_independent_of_the_frame_index(ranked, fitted):
    _, test, model = fitted
    shuffled = test.sample(frac=1, random_state=3)
    shuffled.index = shuffled.index + 1000
    pd.testing.assert_frame_equal(ex.ranked(model, shuffled, T_SOFT, T_VERIFY), ranked)


def test_empty_frame_raises_a_clear_error(fitted):
    _, test, model = fitted
    with pytest.raises(ValueError, match="no accounts"):
        ex.ranked(model, test.iloc[:0], T_SOFT, T_VERIFY)


def test_tabular_only_stack_ranks_without_words(fitted):
    train, test, _ = fitted
    model = Stack.fit(train, use_text=False)
    out = ex.ranked(model, test.iloc[:30], T_SOFT, T_VERIFY)
    assert list(out.columns) == RANKED_COLS and (out["words"] == "").all()
    assert not (out[["f1", "f2", "f3"]] == TEXT_FEATURE).any().any()


def test_ranking_quality_on_held_out(ranked, fitted):
    _, test, _ = fitted
    y = test.set_index(ID_COL)[TARGET].reindex(ranked[ID_COL]).to_numpy()
    assert top_k_precision(y, ranked["score"], 100) >= 0.98
    assert top_k_precision(y, ranked["score"], 300) >= 0.93


# --- files -------------------------------------------------------------------------------

def test_write_ranked_round_trip_and_schema_check(ranked, tmp_path):
    path = tmp_path / "r" / "ranked.csv"
    ex.write_ranked(ranked, path)
    back = pd.read_csv(path, dtype={ID_COL: str}, keep_default_na=False)
    assert list(back.columns) == RANKED_COLS and len(back) == len(ranked)
    assert back[ID_COL].tolist() == ranked[ID_COL].tolist()
    assert list(path.parent.iterdir()) == [path]  # no temp file left behind
    with pytest.raises(ValueError, match="columns"):
        ex.write_ranked(ranked.drop(columns="reason"), path)


def test_failed_write_leaves_no_temp_file(ranked, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(ex.os, "replace", boom)
    with pytest.raises(OSError):
        ex.write_ranked(ranked, tmp_path / "ranked.csv")
    assert list(tmp_path.iterdir()) == []


def test_contrib_table_rebuilds_every_score(fitted, tmp_path):
    _, test, model = fitted
    table = ex.contrib_table(model, test)
    assert list(table.columns[:2]) == [ID_COL, "intercept"]
    assert set(table.columns[2:]) == set(FEATURE_COLS) | {TEXT_FEATURE}
    assert np.allclose(table.iloc[:, 1:].sum(axis=1), logit(model.score(test)), atol=1e-6)
    ex.write_contrib(table, tmp_path / "contrib.csv")
    with pytest.raises(ValueError, match="intercept"):
        ex.write_contrib(table.drop(columns="intercept"), tmp_path / "contrib.csv")


# --- contrarians and interim thresholds --------------------------------------------------

def test_contrarians_are_caught_only_because_of_the_text(fitted):
    _, test, model = fitted
    cands = ex.contrarian_candidates(model, test, T_VERIFY, n=10)
    assert len(cands) > 0
    contrib = ex.contributions(model, test).contrib.set_index(test[ID_COL].astype(str).to_numpy())
    sub = contrib.loc[cands[ID_COL]]
    assert np.allclose(cands["activity_contrib"], sub[ACTIVITY_COLS].sum(axis=1))
    assert (cands["score"] >= T_VERIFY).all()
    assert (cands["text_contrib"] > 0).all() and (cands["activity_contrib"] < 0).all()
    assert (logit(cands["score"]) - cands["text_contrib"] < logit(T_VERIFY)).all()
    assert cands["activity_contrib"].is_monotonic_increasing
    # over all accounts: some pass the activity/text test but would verify without the text; none may appear
    every = ex.contrarian_candidates(model, test, T_VERIFY, n=len(test))
    score = model.score(test)
    text, act = contrib[TEXT_FEATURE].to_numpy(), contrib[ACTIVITY_COLS].sum(axis=1).to_numpy()
    loose = (score >= T_VERIFY) & (text > 0) & (act < 0)
    strict = loose & (logit(score) - text < logit(T_VERIFY))
    assert loose.sum() > strict.sum() and len(every) == strict.sum()


def test_interim_thresholds_follow_the_plan_and_the_budget():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 500)
    oof = np.where(y == 1, rng.uniform(0.3, 1, 1000), rng.uniform(0, 0.7, 1000))
    scores = rng.uniform(0, 1, 400)
    t_soft, t_verify = ex.interim_thresholds(oof, y, scores, cap=0.15, budget=0.25)
    t_cap = ex.cap_threshold(oof, y, 0.15)
    assert t_soft == pytest.approx(min(np.quantile(oof[y == 1], 0.10), t_cap))
    assert t_verify >= t_cap and (scores >= t_verify).sum() <= 100  # the budget wins
    assert ((oof >= t_soft) & (y == 0)).sum() / 500 >= ((oof >= t_cap) & (y == 0)).sum() / 500


# --- metrics.top_k_precision ---------------------------------------------------------------

def test_top_k_precision():
    y, s = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])
    assert top_k_precision(y, s, 1) == 1.0 and top_k_precision(y, s, 2) == 0.5
    assert top_k_precision(y, s, 4) == 0.5
    assert top_k_precision([0, 1], [0.5, 0.5], 1) == 0.0  # ties keep input order
    for k in (0, 5):
        with pytest.raises(ValueError, match="k must be"):
            top_k_precision(y, s, k)
    with pytest.raises(ValueError, match="shape"):
        top_k_precision(y, s[:3], 1)
