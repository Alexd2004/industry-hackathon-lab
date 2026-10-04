"""Explanations and the ranked "likely teen" list (Tier 2, step 10).

Every score is explained exactly: level 2 is a logistic regression on standardized inputs,
so logit(score) = intercept + sum(coef x standardized value). Each term is one feature's
signed contribution (positive pushes toward teen). The top 3 by size become readable chips
("quiet in school hours +0.82"), with their feature keys (f1..f3) and values (v1..v3) kept as
data for A3/A4. Top words are the account's own teen-leaning words (text_model.account_top_words).

Two steps, so the cap slider can re-band without refitting or stale text:
- explain_frame(model, df): score, chips, f/v columns and words. No band; built once.
- apply_bands(frame, t_soft, t_verify): band, action and the one-sentence reason (also A4's
  deterministic fallback). Cheap; rerun on every slider move.
ranked() does both and sorts. ranked.csv (metrics.RANKED_COLS) holds the held-out test
accounts and never a label; contrib.csv holds all contributions for the account detail card.

Thresholds belong to policy.py (step 9). Until it lands, main() uses interim_thresholds(): the
plan's rules plus the 25% review budget, so the demo list is never 55% "verify".

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
from softsignal.metrics import DEFAULT_CAP, RANKED_COLS, cap_threshold, top_k_precision
from softsignal.oracle import REVIEW_BUDGET
from softsignal.stack import TEXT_FEATURE, Stack, logit, stack
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
BANDS = ("none", "soft", "verify")
ACTIONS = {"none": "No action", "soft": "Teen-safe defaults", "verify": "Request verification"}
FRAME_COLS = [ID_COL, "score"] + [f"{p}{i}" for p in ("c", "f", "v") for i in range(1, N_TOP + 1)] + ["words"]


@dataclass
class Explanation:
    """Per account (rows in df order) and level-2 feature: contribution, standardized and raw value."""

    contrib: pd.DataFrame
    z: pd.DataFrame
    raw: pd.DataFrame
    intercept: float


def contributions(model: Stack, df: pd.DataFrame) -> Explanation:
    """coef x standardized value per feature. logit(model.score(df)) == intercept + contrib.sum(axis=1)."""
    if len(df) == 0:
        raise ValueError("no accounts to explain")
    X = model._features(df)  # the level-2 input, exactly as Stack.score builds it
    z = model.level2[0].transform(X)
    lr = model.level2[-1]
    cols = list(X.columns)
    return Explanation(contrib=pd.DataFrame(z * lr.coef_[0], columns=cols), z=pd.DataFrame(z, columns=cols),
                       raw=X.reset_index(drop=True), intercept=float(lr.intercept_[0]))


def chip(feature: str, contrib: float, z: float, raw: float | None = None) -> str:
    """Readable chip: the phrase for the account's side, then the signed contribution."""
    above, below = PHRASES[feature]
    high = raw >= 0 if feature == TEXT_FEATURE and raw is not None else z >= 0
    return f"{above if high else below} {contrib:+.2f}"


def top_terms(exp: Explanation, n: int = N_TOP) -> list[list[tuple[str, float, str]]]:
    """Per account, (feature, contribution, chip) for its n largest contributions by size."""
    names = list(exp.contrib.columns)
    C, Z, R = exp.contrib.to_numpy(), exp.z.to_numpy(), exp.raw.to_numpy(dtype=float)
    out = []
    for c_row, z_row, r_row in zip(C, Z, R):
        top = np.argsort(-np.abs(c_row), kind="stable")[:n]
        out.append([(names[j], float(c_row[j]), chip(names[j], c_row[j], z_row[j], r_row[j])) for j in top])
    return out


def explain_frame(model: Stack, df: pd.DataFrame, n_words: int = 3) -> pd.DataFrame:
    """Band-free explanations in df order (FRAME_COLS). Labels are never read."""
    exp = contributions(model, df)
    terms = top_terms(exp)
    ids = df[ID_COL].astype(str).to_numpy()
    words = (account_top_words(model.text_model, model.tm, ids, n=n_words) if model.use_text
             else [[] for _ in ids])
    out = pd.DataFrame({ID_COL: ids, "score": model.score(df), "words": [", ".join(w) for w in words]})
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


def ranked(model: Stack, df: pd.DataFrame, t_soft: float, t_verify: float, n_words: int = 3) -> pd.DataFrame:
    """The ranked list for these accounts, highest score first, in RANKED_COLS."""
    out = apply_bands(explain_frame(model, df, n_words), t_soft, t_verify)
    out = out.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[RANKED_COLS]


def contrib_table(model: Stack, df: pd.DataFrame) -> pd.DataFrame:
    """All contributions per account (for the detail card): blogger_id, intercept, one column per feature."""
    exp = contributions(model, df)
    out = exp.contrib.copy()
    out.insert(0, "intercept", exp.intercept)
    out.insert(0, ID_COL, df[ID_COL].astype(str).to_numpy())
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file and rename, so the app never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            df.to_csv(f, index=False)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_ranked(df: pd.DataFrame, path: Path = RANKED) -> None:
    if list(df.columns) != RANKED_COLS:
        raise ValueError(f"ranked list must have columns {RANKED_COLS}")
    _write_csv(df, path)


def write_contrib(df: pd.DataFrame, path: Path = CONTRIB) -> None:
    if list(df.columns[:2]) != [ID_COL, "intercept"]:
        raise ValueError(f"contrib table must start with {ID_COL}, intercept")
    _write_csv(df, path)


def contrarian_candidates(model: Stack, df: pd.DataFrame, t_verify: float, n: int = 5) -> pd.DataFrame:
    """Verify-band accounts caught only because of the text: activity points adult, writing points
    teen, and without the text term the score would fall below t_verify.

    These are the demo pin candidates. No labels are used, so a person must still check one
    before showing it as a correct catch. Pass the final (budget-capped) t_verify.
    """
    cols = [ID_COL, "score", "text_contrib", "activity_contrib"]
    if not model.use_text:
        return pd.DataFrame(columns=cols)
    exp = contributions(model, df)
    score = model.score(df)
    text = exp.contrib[TEXT_FEATURE].to_numpy()
    act = exp.contrib[ACTIVITY_COLS].sum(axis=1).to_numpy()
    without_text = logit(score) - text
    keep = (score >= t_verify) & (text > 0) & (act < 0) & (without_text < logit(t_verify))
    out = pd.DataFrame({ID_COL: df[ID_COL].astype(str).to_numpy(), "score": score,
                        "text_contrib": text, "activity_contrib": act})[keep]
    return out.sort_values("activity_contrib", kind="stable").head(n).reset_index(drop=True)[cols]


def interim_thresholds(oof, y, scores, cap: float = DEFAULT_CAP,
                       budget: float = REVIEW_BUDGET) -> tuple[float, float]:
    """(t_soft, t_verify) until policy.py lands. oof/y: nested OOF train scores and labels; scores:
    the accounts being ranked.

    t_cap is the cap cutoff on OOF scores; the plan's t_soft is the 10% quantile of OOF teen
    scores. The review budget wins: t_verify is raised until at most budget x len(scores)
    accounts verify, and every account over t_cap is at least soft (t_soft <= t_cap).
    """
    oof, y, scores = np.asarray(oof, dtype=float), np.asarray(y), np.asarray(scores, dtype=float)
    t_cap = cap_threshold(oof, y, cap)
    t_budget = cap_threshold(scores, np.zeros(len(scores), dtype=int), budget)  # top budget share
    plan_soft = float(np.quantile(oof[y == 1], 0.10))
    return min(plan_soft, t_cap), max(t_cap, t_budget)


def main() -> None:
    train, test = load_data(on_param_mismatch="error")
    res = stack(train, test)
    t_soft, t_verify = interim_thresholds(res.oof, train[TARGET], res.test_score)
    out = ranked(res.stack, test, t_soft, t_verify)
    write_ranked(out)
    write_contrib(contrib_table(res.stack, test))
    counts = out["band"].value_counts().reindex(BANDS, fill_value=0)
    print(f"wrote {len(out)} held-out test accounts to {RANKED} and {CONTRIB.name}")
    print(f"interim thresholds (until policy.py): t_soft {t_soft:.3f}, t_verify {t_verify:.3f} "
          f"(cap cutoff {res.threshold:.3f}, {REVIEW_BUDGET:.0%} review budget)")
    print("bands:", ", ".join(f"{b} {k}" for b, k in counts.items()))
    by_id = test.set_index(ID_COL)[TARGET].reindex(out[ID_COL])
    print("top-k precision (test labels, report only):",
          ", ".join(f"top-{k} {top_k_precision(by_id, out['score'], k):.3f}" for k in (100, 200, 300)))
    print("\n" + out.head(5)[["rank", ID_COL, "score", "band", "c1", "words"]].round(3).to_string(index=False))
    cands = contrarian_candidates(res.stack, test, t_verify)
    print(f"\ncontrarian candidates (caught only because of the text): {len(cands)} shown")
    if len(cands):
        print(cands.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
