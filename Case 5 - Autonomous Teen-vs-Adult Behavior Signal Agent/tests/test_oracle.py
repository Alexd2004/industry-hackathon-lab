"""Oracle rules: batches, audit slice, budget, never-test, reveal-once. Synthetic frames plus one real split."""
import numpy as np
import pandas as pd
import pytest

from softsignal.data import load_data
from softsignal.features import FEATURE_COLS, ID_COL, TARGET
from softsignal.oracle import AUDIT_PER_BATCH, BATCH_SIZE, Batch, Oracle, OracleError

N_TRAIN, N_TEST = 2100, 900


def make_frame(n: int, prefix: str = "T", seed: int = 0) -> pd.DataFrame:
    """Balanced synthetic accounts with features, label and the off-limits columns the CSV also has."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    df.insert(0, ID_COL, [f"{prefix}{i:05d}" for i in range(n)])
    df[TARGET] = np.arange(n) % 2
    df["is_teen"] = df[TARGET].astype(bool)
    df["age"] = np.where(df[TARGET] == 1, 15, 30)
    df["gender"] = "x"
    return df


@pytest.fixture
def train():
    return make_frame(N_TRAIN)


@pytest.fixture
def test_ids():
    return make_frame(N_TEST, prefix="X")[ID_COL].tolist()


@pytest.fixture
def oracle(train, test_ids):
    return Oracle(train, test_ids)


def labels_of(train: pd.DataFrame) -> dict[str, int]:
    return dict(zip(train[ID_COL], train[TARGET]))


def top_band(batch: Batch, k: int) -> list[str]:
    """A stand-in for 'top k by score': the first k ids of the batch."""
    return list(batch.ids[:k])


def all_batches(o: Oracle) -> list[Batch]:
    return list(o)


# batches


def test_seven_stratified_batches_cover_train_once(oracle, train):
    batches = all_batches(oracle)
    assert [b.round for b in batches] == list(range(1, 8))
    y = labels_of(train)
    for b in batches:
        assert len(b.ids) == BATCH_SIZE
        assert sum(y[i] for i in b.ids) == BATCH_SIZE // 2
    flat = [i for b in batches for i in b.ids]
    assert len(flat) == len(set(flat)) == N_TRAIN
    assert set(flat) == set(train[ID_COL])


def test_next_batch_is_none_after_last_round(oracle):
    for _ in range(7):
        assert oracle.next_batch() is not None
    assert oracle.next_batch() is None
    assert oracle.next_batch() is None
    assert list(oracle) == []


def test_batch_rows_carry_features_only_and_match_ids(oracle, train):
    by_id = train.set_index(ID_COL)
    for b in oracle:
        assert list(b.rows.columns) == [ID_COL, *FEATURE_COLS]
        assert not {TARGET, "is_teen", "age", "gender"} & set(b.rows.columns)
        assert tuple(b.rows[ID_COL]) == b.ids
        np.testing.assert_allclose(b.rows[FEATURE_COLS].to_numpy(), by_id.loc[list(b.ids), FEATURE_COLS].to_numpy())


def test_labels_live_only_in_the_label_map(oracle):
    """No frame held by the oracle carries a label or label proxy column; the labels sit in one private dict."""
    for name, value in vars(oracle).items():
        if isinstance(value, pd.DataFrame):
            assert not {TARGET, "is_teen", "age"} & (set(value.columns) | {value.index.name}), name
    assert set(oracle._labels.values()) == {0, 1}


def test_uneven_train_size_still_partitions(test_ids):
    train = make_frame(N_TRAIN + 1)
    o = Oracle(train, test_ids)
    batches = all_batches(o)
    sizes = sorted(len(b.ids) for b in batches)
    assert len(batches) == 7 and sizes[-1] - sizes[0] <= 1 and sum(sizes) == N_TRAIN + 1
    assert sizes[-1] == 301


def test_small_last_batch_audits_whole_batch_and_budget_floors(test_ids):
    """One batch of 40: audit = min(60, 40), budget = floor(0.25 * 40) = 10."""
    o = Oracle(make_frame(40), test_ids)
    b = o.next_batch()
    assert len(b.ids) == 40 and o.verify_budget(b) == 10
    rv = o.reveal(b, top_band(b, 10))
    assert rv.n_audit == 40 and len(rv.labels) == 40


# determinism


def test_same_seed_same_batches_and_audit(train, test_ids):
    runs = []
    for _ in range(2):
        o = Oracle(train, test_ids, seed=7)
        runs.append([(b.ids, o.reveal(b, [])) for b in o])
    for (ids_a, rv_a), (ids_b, rv_b) in zip(*runs):
        assert ids_a == ids_b
        pd.testing.assert_frame_equal(rv_a.labels, rv_b.labels)


def test_other_seed_gives_other_batches_and_audit(train, test_ids):
    a, b = Oracle(train, test_ids, seed=1), Oracle(train, test_ids, seed=2)
    ba, bb = a.next_batch(), b.next_batch()
    assert ba.ids != bb.ids
    a.reveal(ba, [])
    b.reveal(bb, [])
    assert set(a.log[-1]["audit_ids"]) != set(b.log[-1]["audit_ids"])


def test_audit_ids_do_not_depend_on_the_verify_band(train, test_ids):
    a, b = Oracle(train, test_ids), Oracle(train, test_ids)
    for ba, bb in zip(a, b):
        ra = a.reveal(ba, top_band(ba, 75))
        rb = b.reveal(bb, list(reversed(bb.ids))[:30])
        audit_a = set(ra.labels.loc[ra.labels["in_audit"], ID_COL])
        audit_b = set(rb.labels.loc[rb.labels["in_audit"], ID_COL])
        assert audit_a == audit_b and len(audit_a) == AUDIT_PER_BATCH
        assert audit_a <= set(ba.ids)


def test_audit_differs_between_rounds(oracle):
    audits = []
    for b in oracle:
        oracle.reveal(b, [])
        audits.append(frozenset(oracle.log[-1]["audit_ids"]))
    assert len(set(audits)) == 7


# test ids


def test_constructor_rejects_train_test_overlap(train):
    with pytest.raises(OracleError, match="both train and test"):
        Oracle(train, [train[ID_COL].iloc[3], "X1"])


def test_test_id_is_never_revealed_even_on_a_bad_call(oracle, test_ids):
    b = oracle.next_batch()
    with pytest.raises(OracleError, match="never revealed"):
        oracle.reveal(b, [b.ids[0], test_ids[0]])
    # checked first: a stale or repeated batch still gets the test-id error
    oracle.reveal(b, [])
    with pytest.raises(OracleError, match="never revealed"):
        oracle.reveal(b, [test_ids[5]])


def test_failed_reveal_leaves_round_open(oracle, test_ids):
    b = oracle.next_batch()
    with pytest.raises(OracleError):
        oracle.reveal(b, [test_ids[0]])
    rv = oracle.reveal(b, top_band(b, 5))
    assert rv.n_verify == 5 and oracle.revealed()[ID_COL].isin(test_ids).sum() == 0


# batch and round rules


def test_id_from_another_batch_raises(oracle):
    b1 = oracle.next_batch()
    other = next(i for i in oracle._batches[1] if i not in b1.ids)
    with pytest.raises(OracleError, match="not in round 1"):
        oracle.reveal(b1, [other])


def test_unknown_id_raises(oracle):
    b = oracle.next_batch()
    with pytest.raises(OracleError, match="not in round 1"):
        oracle.reveal(b, ["NOPE"])


def test_second_reveal_of_a_round_raises(oracle):
    b = oracle.next_batch()
    oracle.reveal(b, top_band(b, 10))
    with pytest.raises(OracleError, match="already revealed"):
        oracle.reveal(b, [])


def test_reveal_of_an_earlier_unrevealed_batch_raises(oracle):
    b1 = oracle.next_batch()
    oracle.next_batch()
    with pytest.raises(OracleError, match="not the current round"):
        oracle.reveal(b1, [])


def test_reveal_past_the_cursor_raises(oracle):
    b1 = oracle.next_batch()
    forged = Batch(round=2, ids=oracle._batches[1], rows=b1.rows)
    with pytest.raises(OracleError, match="not the current round"):
        oracle.reveal(forged, [])


def test_reveal_before_any_batch_raises(oracle):
    with pytest.raises(OracleError):
        oracle.reveal(Batch(round=1, ids=oracle._batches[0], rows=pd.DataFrame()), [])


def test_batch_with_swapped_ids_raises(oracle):
    b = oracle.next_batch()
    forged = Batch(round=1, ids=oracle._batches[1], rows=b.rows)
    with pytest.raises(OracleError, match="do not match"):
        oracle.reveal(forged, [])


# budget


def test_band_of_76_raises_and_75_is_ok(train, test_ids):
    o = Oracle(train, test_ids)
    b = o.next_batch()
    assert o.verify_budget(b) == 75
    with pytest.raises(OracleError, match="budget is 75"):
        o.reveal(b, top_band(b, 76))
    rv = o.reveal(b, top_band(b, 75))
    assert rv.n_verify == 75 and rv.labels["in_verify"].sum() == 75


def test_duplicates_are_dropped_before_the_budget(oracle):
    b = oracle.next_batch()
    band = top_band(b, 75)
    rv = oracle.reveal(b, band + band[:10])
    assert rv.n_verify == 75 and oracle.log[-1]["verify_ids"] == band


def test_empty_band_still_reveals_the_audit(oracle, train):
    b = oracle.next_batch()
    rv = oracle.reveal(b, [])
    assert rv.n_verify == 0 and rv.n_audit == AUDIT_PER_BATCH and len(rv.labels) == AUDIT_PER_BATCH
    assert rv.labels["in_audit"].all() and not rv.labels["in_verify"].any()
    y = labels_of(train)
    assert rv.labels[TARGET].tolist() == [y[i] for i in rv.labels[ID_COL]]


def test_overlap_is_revealed_once_with_both_flags(train, test_ids):
    probe = Oracle(train, test_ids)
    b = probe.next_batch()
    audit = probe.reveal(b, []).labels[ID_COL].tolist()
    non_audit = [i for i in b.ids if i not in set(audit)]

    o = Oracle(train, test_ids)
    b = o.next_batch()
    band = audit[:20] + non_audit[:30]
    rv = o.reveal(b, band)
    assert rv.n_overlap == 20 and rv.n_verify == 50 and rv.n_audit == 60
    assert len(rv.labels) == rv.labels[ID_COL].nunique() == 90 == o.log[-1]["n_labels"]
    both = rv.labels[rv.labels["in_verify"] & rv.labels["in_audit"]]
    assert set(both[ID_COL]) == set(audit[:20])


def test_max_labels_over_a_full_run_is_945(oracle):
    for b in oracle:
        oracle.reveal(b, [i for i in b.ids if i not in set(oracle._audit[b.round - 1])][:75])
    assert sum(e["n_labels"] for e in oracle.log) == 945 == len(oracle.revealed())


# revealed store, audit counts and log


def test_revealed_sources_and_before_round(oracle, train):
    y = labels_of(train)
    for _ in range(3):
        b = oracle.next_batch()
        oracle.reveal(b, top_band(b, 40))
    every = oracle.revealed()
    assert list(every.columns) == ["round", ID_COL, TARGET, "in_verify", "in_audit"]
    assert sorted(every["round"].unique()) == [1, 2, 3]
    assert every[TARGET].tolist() == [y[i] for i in every[ID_COL]]
    audit = oracle.revealed("audit")
    assert audit["in_audit"].all() and len(audit) == 3 * AUDIT_PER_BATCH
    verify = oracle.revealed("verify")
    assert verify["in_verify"].all() and len(verify) == 3 * 40
    early = oracle.revealed("all", before_round=3)
    assert set(early["round"]) == {1, 2}
    assert oracle.revealed(before_round=1).empty
    with pytest.raises(OracleError, match="source"):
        oracle.revealed("test")


def test_returned_frames_are_copies(oracle):
    b = oracle.next_batch()
    rv = oracle.reveal(b, [])
    rv.labels.loc[:, TARGET] = -1
    out = oracle.revealed()
    out.loc[:, TARGET] = -1
    assert set(oracle.revealed()[TARGET]) <= {0, 1}


def test_audit_counts_match_labels_and_log(oracle, train):
    assert oracle.audit_counts() == {"adults": 0, "teens": 0}
    y = labels_of(train)
    for _ in range(2):
        b = oracle.next_batch()
        oracle.reveal(b, top_band(b, 75))
    audit_ids = [i for e in oracle.log for i in e["audit_ids"]]
    adults = sum(y[i] == 0 for i in audit_ids)
    assert oracle.audit_counts() == {"adults": adults, "teens": len(audit_ids) - adults}
    assert sum(e["n_audit_adults"] for e in oracle.log) == adults
    first = oracle.log[0]
    assert oracle.audit_counts(before_round=2)["adults"] == first["n_audit_adults"]


def test_log_has_one_entry_per_reveal_with_the_spec_keys(oracle):
    b = oracle.next_batch()
    oracle.reveal(b, top_band(b, 12))
    assert len(oracle.log) == 1
    e = oracle.log[0]
    assert list(e) == ["round", "verify_ids", "audit_ids", "n_verify", "n_audit", "n_overlap", "n_labels",
                       "n_audit_adults"]
    assert e["n_labels"] == e["n_verify"] + e["n_audit"] - e["n_overlap"]


# setup checks


def test_train_without_label_raises(train, test_ids):
    with pytest.raises(OracleError, match=TARGET):
        Oracle(train.drop(columns=TARGET), test_ids)


@pytest.mark.parametrize("kw", [{"batch_size": 0}, {"audit_per_batch": -1}, {"review_budget": 1.5}])
def test_bad_settings_raise(train, test_ids, kw):
    with pytest.raises(OracleError):
        Oracle(train, test_ids, **kw)


# real split


def test_from_split_on_the_committed_split():
    train, test = load_data()
    o = Oracle.from_split()
    test_set = set(test[ID_COL])
    y = labels_of(train)
    seen = []
    for b in o:
        assert len(b.ids) == BATCH_SIZE and sum(y[i] for i in b.ids) == BATCH_SIZE // 2
        assert not test_set & set(b.ids)
        assert not {TARGET, "is_teen", "age"} & set(b.rows.columns)
        o.reveal(b, top_band(b, 75))
        seen += b.ids
    assert o.n_rounds == 7 and set(seen) == set(train[ID_COL])
    assert not test_set & set(o.revealed()[ID_COL])
    assert not test_set & set(o._labels)
