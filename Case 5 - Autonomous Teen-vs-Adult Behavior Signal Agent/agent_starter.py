"""Teen vs adult: keyword baseline -> style score -> blend activity once."""
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.data import load_data
from softsignal.metrics import prf
from softsignal.tier1 import activity_score, blend, load_best, style_score, style_sweep_report, tune

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
    print(f"\nStored best before tuning: {prior if prior else 'none'}")
    tuned = tune(train, test)
    print(f"\nTune picks (train, {len(tuned['grid'])} grid points): {tuned['picks']}")
    pt = tuned["picks"]["cap_best"]
    if pt is not None:
        cv = tuned["cv"]
        fold = cv[(cv["w"] == pt["w"]) & (cv["cutoff"] == pt["cutoff"])].iloc[0]
        print(f"cap_best over 5 train folds: mean rec {fold['rec']:.3f}, mean ft {fold['ft']:.3f}, worst-fold ft {fold['ft_max']:.3f}")
    print(f"Picked points on test ({len(test)} rows):")
    print(tuned["test"].round(3).to_string(index=False))
    print(f"Checkpoint written: {tuned['checkpoint_written']}")

    flipped = int((v1 != v2).sum())
    print(f"\nAccounts that flipped v1->revise: {flipped}/{len(df)}")
    print("Next: train LogisticRegression on the numeric columns.")
    print("Honesty: posts are 2004 blogs; activity columns are synthetic_calibrated_demo.")


if __name__ == "__main__":
    main()
