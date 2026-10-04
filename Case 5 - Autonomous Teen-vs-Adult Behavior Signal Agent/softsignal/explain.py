"""Explanations and the ranked "likely teen" list (Tier 2, step 10).

Every score is explained exactly: level 2 is a logistic regression on standardized inputs,
so logit(score) = intercept + sum(coef x standardized value). Each term is one feature's
signed contribution (positive pushes toward teen). Stack.contributions does that maths; here
the score itself is computed from the terms, so the explanation is exact by construction.
The top 3 by size become readable chips ("quiet in school hours +0.82"), with their feature
keys (f1..f3) and values (v1..v3) kept as data for A3/A4. Top words are the account's own
teen-leaning words (text_model.account_top_words).

Two steps, so the cap slider can re-band without refitting or stale text:
- explain_frame(model, df): score, chips, f/v columns and words. No band; built once.
- apply_bands(frame, t_soft, t_verify): band, action and the one-sentence reason (also A4's
  deterministic fallback). Cheap; rerun on every slider move.
ranked() does both and sorts. ranked.csv (metrics.RANKED_COLS) holds the held-out test
accounts and never a label; contrib.csv (metrics.CONTRIB_COLS, long format) holds every
contribution with its raw and standardized value, for the account detail card.

Thresholds belong to policy.py (step 9). Until it lands, main() uses interim_thresholds(): the
plan's rules plus the review budget, so the demo list is never 55% "verify".

Run: python -m softsignal.explain
"""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.data import ROOT, load_data
from softsignal.features import ACTIVITY_COLS, ID_COL, TARGET
from softsignal.metrics import (
    CONTRIB_COLS, DEFAULT_CAP, RANKED_COLS, cap_threshold, top_k_precision, top_share_cutoff,
)
from softsignal.oracle import REVIEW_BUDGET
from softsignal.stack import TEXT_FEATURE, Stack, stack
from softsignal.text_model import account_top_words

RANKED = ROOT / "results" / "ranked.csv"
CONTRIB = ROOT / "results" / "contrib.csv"
N_TOP = 3

# (phrase when the account's value is above average, phrase when below). For the text score the
# split is the score itself (p >= 0.5), not the average, so a 0.6 text score never reads "adult".
PHRASES = {
    TEXT_FEATURE: ("writes like a teen", "writes like an adult"),
    "pct_active_school_hours": ("active in school hours", "quiet in school hours"),
    "pct_active_evening": ("active in the evening", "rarely active in the evening"),
    "pct_active_late_night": ("active late at night", "rarely up late at night"),
    "weekend_weekday_session_ratio": ("busier on weekends", "busier on weekdays"),
    "sessions_per_day": ("many sessions a day", "few sessions a day"),
    "avg_session_minutes": ("long sessions", "short sessions"),
    "share_short_video_views": ("watches a lot of short video", "watches little short video"),
    "share_news_views": ("reads a lot of news", "reads little news"),
    "night_notification_open_rate": ("opens notifications at night", "rarely opens notifications at night"),
    "avg_word_len": ("uses longer words", "uses shorter words"),
    "first_person_rate": ("writes a lot in first person", "writes little in first person"),
    "school_token_rate": ("mentions school", "rarely mentions school"),
    "birthday_token_rate": ("mentions birthdays", "rarely mentions birthdays"),
    "exclaim_rate": ("uses many exclamation marks", "uses few exclamation marks"),
    "slang_emoji_rate": ("uses slang and emoji", "uses little slang or emoji"),
    "keyword_teen_flag": ("uses school or birthday keywords", "no school or birthday keywords"),
}
# A text score just past 0.5 can sit on the other side of the training average, so its
# contribution's sign disagrees with the teen/adult phrase; it then reads as neutral.
TEXT_NEUTRAL = "writing near average"
BANDS = ("none", "soft", "verify")
ACTIONS = {"none": "No action", "soft": "Teen-safe defaults", "verify": "Request verification"}
FRAME_COLS = [ID_COL, "score"] + [f"{p}{i}" for p in ("c", "f", "v") for i in range(1, N_TOP + 1)] + ["words"]


@dataclass
class Explanation:
    """One model pass over some accounts (rows in df order); every other function reuses it.

    ids: account ids as strings; contrib/z/raw: per level-2 feature; intercept; score is
    sigmoid(intercept + contrib.sum(axis=1)), so it equals model.score(df) up to float error.
    """

    ids: np.ndarray
    contrib: pd.DataFrame
    z: pd.DataFrame
    raw: pd.DataFrame
    intercept: float

    @property
    def logit(self) -> np.ndarray:
        return self.intercept + self.contrib.to_numpy().sum(axis=1)

    @property
    def score(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit))


def contributions(model: Stack, df: pd.DataFrame) -> Explanation:
    """coef x standardized value per account and level-2 feature (one pass of the model)."""
    if len(df) == 0:
        raise ValueError("no accounts to explain")
    X, z, contrib, intercept = model.contributions(df)
    cols = list(X.columns)
    return Explanation(ids=df[ID_COL].astype(str).to_numpy(), contrib=pd.DataFrame(contrib, columns=cols),
                       z=pd.DataFrame(z, columns=cols), raw=X.reset_index(drop=True), intercept=intercept)


def chip(feature: str, contrib: float, z: float, raw: float | None = None) -> str:
    """Readable chip: the phrase for the account's side, then the signed contribution.

    For the text score, raw (its logit) is required: the phrase follows p >= 0.5, and when that
    disagrees with the contribution's sign the chip reads TEXT_NEUTRAL instead.
    """
    above, below = PHRASES[feature]
    if feature == TEXT_FEATURE:
        if raw is None:
            raise ValueError("the text chip needs raw (the text logit), not only z")
        teen = raw >= 0
        if teen != (contrib >= 0):
            return f"{TEXT_NEUTRAL} {contrib:+.2f}"
        return f"{above if teen else below} {contrib:+.2f}"
    return f"{above if z >= 0 else below} {contrib:+.2f}"


def top_terms(exp: Explanation, n: int = N_TOP) -> list[list[tuple[str, float, str]]]:
    """Per account, (feature, contribution, chip) for its n largest contributions by size."""
    names = list(exp.contrib.columns)
    C, Z, R = exp.contrib.to_numpy(), exp.z.to_numpy(), exp.raw.to_numpy(dtype=float)
    out = []
    for c_row, z_row, r_row in zip(C, Z, R):
        top = np.argsort(-np.abs(c_row), kind="stable")[:n]
        out.append([(names[j], float(c_row[j]), chip(names[j], c_row[j], z_row[j], r_row[j])) for j in top])
    return out


def explain_frame(model: Stack, df: pd.DataFrame, n_words: int = 3, exp: Explanation | None = None) -> pd.DataFrame:
    """Band-free explanations in df order (FRAME_COLS). Labels are never read."""
    exp = exp if exp is not None else contributions(model, df)
    terms = top_terms(exp)
    words = (account_top_words(model.text_model, model.tm, exp.ids, n=n_words) if model.use_text
             else [[] for _ in exp.ids])
    out = pd.DataFrame({ID_COL: exp.ids, "score": exp.score, "words": [", ".join(w) for w in words]})
    for i in range(N_TOP):
        out[f"c{i + 1}"] = [t[i][2] if len(t) > i else "" for t in terms]
        out[f"f{i + 1}"] = [t[i][0] if len(t) > i else "" for t in terms]
        out[f"v{i + 1}"] = [t[i][1] if len(t) > i else np.nan for t in terms]
    return out[FRAME_COLS]


def bands(score, t_soft: float, t_verify: float) -> np.ndarray:
    """verify if score >= t_verify, soft if t_soft <= score < t_verify, else none.

    If t_soft >= t_verify the soft band is empty (every account at or above t_verify verifies).
    """
    if not (np.isfinite(t_soft) and np.isfinite(t_verify)):
        raise ValueError(f"thresholds must be finite, got t_soft={t_soft}, t_verify={t_verify}")
    score = np.asarray(score, dtype=float)
    if np.isnan(score).any():
        raise ValueError("scores contain NaN")
    return np.where(score >= t_verify, "verify", np.where(score >= t_soft, "soft", "none"))


def reason(score: float, band: str, chips: list[str], words: str) -> str:
    """One plain sentence per account; the A4 fallback. Words only where the account is flagged."""
    text = f"{ACTIONS[band]} (score {score:.2f}): " + "; ".join(c for c in chips if c) + "."
    if words and band != "none":
        text += f" Teen-leaning words: {words}."
    return text


def apply_bands(frame: pd.DataFrame, t_soft: float, t_verify: float) -> pd.DataFrame:
    """Add band, action and reason to an explain_frame; cheap enough for every slider move."""
    out = frame.copy()
    out["band"] = bands(out["score"], t_soft, t_verify)
    out["action"] = out["band"].map(ACTIONS)
    out["reason"] = [reason(s, b, [c1, c2, c3], w) for s, b, c1, c2, c3, w
                     in zip(out["score"], out["band"], out["c1"], out["c2"], out["c3"], out["words"])]
    return out


def rank_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Highest score first; ties broken by account id, so the order never depends on input order."""
    return frame.sort_values(["score", ID_COL], ascending=[False, True], kind="stable").reset_index(drop=True)


def ranked(model: Stack, df: pd.DataFrame, t_soft: float, t_verify: float, n_words: int = 3,
           exp: Explanation | None = None) -> pd.DataFrame:
    """The ranked list for these accounts, highest score first, in RANKED_COLS."""
    out = rank_order(apply_bands(explain_frame(model, df, n_words, exp), t_soft, t_verify))
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[RANKED_COLS]


def contrib_table(model: Stack, df: pd.DataFrame, exp: Explanation | None = None) -> pd.DataFrame:
    """Long format (CONTRIB_COLS): one row per (account, feature), in df order then feature order."""
    exp = exp if exp is not None else contributions(model, df)
    feats = list(exp.contrib.columns)
    n, k = len(exp.ids), len(feats)
    out = pd.DataFrame({
        ID_COL: np.repeat(exp.ids, k),
        "score": np.repeat(exp.score, k),
        "intercept": exp.intercept,
        "feature": np.tile(feats, n),
        "raw": exp.raw.to_numpy(dtype=float).ravel(),
        "z": exp.z.to_numpy().ravel(),
        "contrib": exp.contrib.to_numpy().ravel(),
    })
    return out[CONTRIB_COLS]


def _readable_mode() -> int:
    """0o666 minus the process umask: what open() would give a new file (mkstemp gives 0o600)."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file and rename, so the app never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            df.to_csv(f, index=False)
        os.chmod(tmp, _readable_mode())  # other readers (a container, CI, the demo account) need access
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_ranked(df: pd.DataFrame, path: Path = RANKED) -> None:
    if list(df.columns) != RANKED_COLS:
        raise ValueError(f"ranked list must have columns {RANKED_COLS}")
    _write_csv(df, path)


def write_contrib(df: pd.DataFrame, path: Path = CONTRIB) -> None:
    """Check the full schema: CONTRIB_COLS and every account having the same feature set."""
    if list(df.columns) != CONTRIB_COLS:
        raise ValueError(f"contrib table must have columns {CONTRIB_COLS}")
    per_account = df.groupby(ID_COL, sort=False)["feature"].apply(frozenset)
    if per_account.nunique() != 1 or df.duplicated([ID_COL, "feature"]).any():
        raise ValueError("every account must have each level-2 feature exactly once")
    _write_csv(df, path)


def contrarian_candidates(model: Stack, df: pd.DataFrame, t_verify: float, n: int = 5,
                          exp: Explanation | None = None) -> pd.DataFrame:
    """Verify-band accounts caught only because of what they wrote: activity points adult, the
    text score points teen, and with the text term at its training average the score would fall
    below t_verify. (Style columns such as slang come from the posts too, so the catch is "from
    the writing", not "from TF-IDF alone".)

    These are the demo pin candidates. No labels are used, so a person must still check one
    before showing it as a correct catch. Pass the final (budget-capped) t_verify.
    """
    cols = [ID_COL, "score", "text_contrib", "activity_contrib"]
    if not model.use_text:
        return pd.DataFrame(columns=cols)
    exp = exp if exp is not None else contributions(model, df)
    text = exp.contrib[TEXT_FEATURE].to_numpy()
    act = exp.contrib[ACTIVITY_COLS].sum(axis=1).to_numpy()
    t_logit = np.log(t_verify / (1 - t_verify))
    keep = (exp.score >= t_verify) & (text > 0) & (act < 0) & (exp.logit - text < t_logit)  # exact, unclipped
    out = pd.DataFrame({ID_COL: exp.ids, "score": exp.score, "text_contrib": text, "activity_contrib": act})[keep]
    out = out.sort_values(["activity_contrib", ID_COL], kind="stable")
    return out.head(n).reset_index(drop=True)[cols]


def interim_thresholds(oof, y, scores, cap: float = DEFAULT_CAP,
                       budget: float = REVIEW_BUDGET) -> tuple[float, float]:
    """(t_soft, t_verify) until policy.py lands. oof/y: nested OOF train scores and labels; scores:
    the accounts being banded (one batch, here the test set).

    t_cap is the cap cutoff on OOF scores; the plan's t_soft is the 10% quantile of OOF teen
    scores. The review budget wins: t_verify is raised to the batch's top-budget cutoff, so the
    verify band is budget-bound BY DESIGN (its size is set here, it is not a result); with fewer
    than 1 / budget accounts nobody verifies. Every account over t_cap is at least soft. policy.py
    should keep the budget a per-batch rule when it takes this over.
    """
    oof, y = np.asarray(oof, dtype=float), np.asarray(y)
    t_cap = cap_threshold(oof, y, cap)
    t_budget = top_share_cutoff(scores, budget)
    plan_soft = float(np.quantile(oof[y == 1], 0.10))
    return min(plan_soft, t_cap), max(t_cap, t_budget)


def main() -> None:
    train, test = load_data(on_param_mismatch="error")
    res = stack(train, test)
    exp = contributions(res.stack, test)  # one model pass, reused below
    t_soft, t_verify = interim_thresholds(res.oof, train[TARGET], exp.score)
    out = ranked(res.stack, test, t_soft, t_verify, exp=exp)
    write_ranked(out)
    write_contrib(contrib_table(res.stack, test, exp=exp))
    counts = out["band"].value_counts().reindex(BANDS, fill_value=0)
    print(f"wrote {len(out)} held-out test accounts to {RANKED} and {CONTRIB.name}")
    print(f"interim thresholds (until policy.py): t_soft {t_soft:.3f}, t_verify {t_verify:.3f} "
          f"(cap cutoff {res.threshold:.3f}; verify is capped at the {REVIEW_BUDGET:.0%} review budget by design)")
    print("bands:", ", ".join(f"{b} {k}" for b, k in counts.items()))
    by_id = test.set_index(ID_COL)[TARGET].reindex(out[ID_COL])
    print("top-k precision (test labels, report only):",
          ", ".join(f"top-{k} {top_k_precision(by_id, out['score'], k):.3f}" for k in (100, 200, 300)))
    print("\n" + out.head(5)[["rank", ID_COL, "score", "band", "c1", "words"]].round(3).to_string(index=False))
    cands = contrarian_candidates(res.stack, test, t_verify, exp=exp)
    print(f"\ncontrarian candidates (caught only because of the writing): {len(cands)} shown")
    if len(cands):
        print(cands.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
