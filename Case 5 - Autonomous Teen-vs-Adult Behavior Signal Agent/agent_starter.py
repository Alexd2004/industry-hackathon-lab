"""Teen vs adult: keyword baseline -> style score -> blend activity once."""
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.data import load_data
from softsignal.metrics import prf
from softsignal.tier1 import (
    EVAL_TIER1, K_FOLDS, activity_score, blend, evaluate_blend, is_current, load_best, style_score, style_sweep_report, tune,
)

JOINED = Path(__file__).parent / "data" / "teen_adult_joined.csv"


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    prec, rec, ft, mt = prf(y_true, y_pred)
    acc = float((y_true == y_pred).mean())
    print(
        f"{name:28} acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  "
        f"false_teen={ft:.3f}  missed_teen={mt:.3f}"
    )


def main() -> None:
    df = pd.read_csv(JOINED)
    y = df["label_teen"].to_numpy()

    # Starter rows below use all rows (train + test), so they are not held-out results.
    # --- Baseline: school/birthday keyword flag from posts ---
    baseline = df["keyword_teen_flag"].to_numpy()
    report("Baseline keywords", y, baseline)

    # --- v1: writing-style score (no activity yet) ---
    # Teens in this corpus tend toward shorter words, more first-person, more bangs/slang.
    style = style_score(df)
    v1 = (style >= 0.55).astype(int).to_numpy()
    report("v1 style score>=0.55", y, v1)

    # --- Revise: blend style with Meta-like activity soft signals ---
    # Higher evening / short-video / night opens, lower school-hour activity -> more teen-like.
    activity = activity_score(df)
    blend_w = 0.45  # weight on activity; change this and re-run
    blended = blend(style, activity, blend_w)
    v2 = (blended >= 0.50).astype(int).to_numpy()
    report(f"Revise blend w={blend_w}", y, v2)

    # --- Step 3: sweep the style cutoff on train, evaluate the picks once on test ---
    train, test = load_data()
    rep = style_sweep_report(train, test)
    print(f"\nStyle cutoff sweep on train ({len(train)} rows):")
    print(rep["train_sweep"].round(3).to_string(index=False))
    print(f"Picks (train): {rep['picks']}")
    print(f"\nPicked cutoffs on test ({len(test)} rows):")
    print(rep["test"].round(3).to_string(index=False))
    print(f"Flag-everyone F1 on test (reference): {rep['flag_all_f1']:.3f}")

    # --- Step 4: grid blend weight x cutoff on train under the 15% cap, picks scored once on test ---
    prior = load_best()
    stale = "" if prior is None or is_current(prior, train) else " (stale: made under a different cap, rules or data)"
    print(f"\nStored best before tuning: {prior if prior else 'none'}{stale}")
    tuned = tune(train, test)
    print(f"Tune picks (train, {len(tuned['cv'])} grid points, {K_FOLDS}-fold): {tuned['picks']}")
    fold = tuned["cv_cap_best"]
    if fold is not None:
        print(f"cap_best over {K_FOLDS} train folds: mean rec {fold['rec']:.3f}, mean ft {fold['ft']:.3f}, worst-fold ft {fold['ft_max']:.3f}")
    print(f"Picked points on test ({len(test)} rows):")
    print(tuned["test"].round(3).to_string(index=False))
    print(f"Checkpoint written: {tuned['checkpoint_written']}")

    # --- Step 5: starter vs one alternate blend_w vs the tuned point, on test, in the eval.csv schema ---
    ladder = evaluate_blend(test, tuned["picks"]["cap_best"])
    EVAL_TIER1.parent.mkdir(parents=True, exist_ok=True)
    ladder.to_csv(EVAL_TIER1, index=False)
    print(f"\nBlend weight comparison on test ({len(test)} rows), written to {EVAL_TIER1.name}:")
    print(ladder.round(3).to_string(index=False))

    flipped = int((v1 != v2).sum())
    print(f"\nAccounts that flipped v1->revise: {flipped}/{len(df)}")
    print("Next: train LogisticRegression on the numeric columns.")
    print("Honesty: posts are 2004 blogs; activity columns are synthetic_calibrated_demo.")


if __name__ == "__main__":
    main()
