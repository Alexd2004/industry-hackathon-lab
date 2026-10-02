"""Weak-pump flag: always-normal vs an EF rule, then one revise."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "cardiac_mri_volumes_seed.csv"
EF_CUTOFF = 40.0  # flag hearts with EF below this percent - change me to 50 and re-run
EF_REVISE = 50.0


def prf(flag, weak):
    tp = int(((flag == 1) & (weak == 1)).sum())
    fp = int(((flag == 1) & (weak == 0)).sum())
    fn = int(((flag == 0) & (weak == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, prec, rec


def main():
    df = pd.read_csv(DATA)
    missing = int(df[["edv_ml", "esv_ml"]].isna().sum().sum())
    df = df.dropna(subset=["edv_ml", "esv_ml"]).reset_index(drop=True)
    df["ef_check"] = (df["edv_ml"] - df["esv_ml"]) / df["edv_ml"] * 100
    weak = (df["group"].isin(["HF", "HF-I"])).astype(int)
    base_acc = float((weak == 0).mean())

    print(f"Dropped {missing} rows with missing volumes. Scoring {len(df)} hearts ({int(weak.sum())} weak).")
    print(f"Baseline always-normal accuracy: {base_acc:.3f} (looks fine, catches nobody)")
    for name, cut in [(f"v1 EF<{EF_CUTOFF:.0f}%", EF_CUTOFF), (f"revise EF<{EF_REVISE:.0f}%", EF_REVISE)]:
        tp, fp, fn, prec, rec = prf((df["ef_check"] < cut).astype(int), weak)
        alarm = "false alarm" if fp == 1 else "false alarms"
        print(f"{name}: caught {tp}/{tp + fn} weak hearts, {fp} {alarm} "
              f"(precision {prec:.2f}, recall {rec:.2f})")
    print("Next: also flag enlarged hearts (high lv_mass_g) and re-score.")


if __name__ == "__main__":
    main()
