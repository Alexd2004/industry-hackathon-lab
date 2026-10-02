"""Well event flag: always-normal vs a pressure rule, then one revise."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "well_hydrate_seed.csv"
PCTL = 5.0  # flag if pressure below this train percentile - change me to 10 and re-run
PCTL_REVISE = 10.0


def prf(flag, label):
    tp = int(((flag == 1) & (label == 1)).sum())
    fp = int(((flag == 1) & (label == 0)).sum())
    fn = int(((flag == 0) & (label == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, prec, rec


def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")
    missing = int(df["pressure_bar"].isna().sum())
    df = df.dropna(subset=["pressure_bar"]).reset_index(drop=True)
    cut = df["timestamp"].min() + pd.Timedelta(20, "D")
    train, test = df[df["timestamp"] < cut], df[df["timestamp"] >= cut]
    base_acc = float((test["label"] == 0).mean())

    t1 = float(train["pressure_bar"].quantile(PCTL / 100))
    v1 = (test["pressure_bar"] < t1).astype(int)
    t2 = float(train["pressure_bar"].quantile(PCTL_REVISE / 100))
    v2 = (test["pressure_bar"] < t2).astype(int)

    print(f"Dropped {missing} rows with missing pressure. Train {len(train)}h, test {len(test)}h.")
    print(f"Baseline always-normal accuracy: {base_acc:.3f} (looks good, catches nothing)")
    for name, flag in [(f"v1 pressure<p{PCTL:.0f}", v1), (f"revise pressure<p{PCTL_REVISE:.0f}", v2)]:
        tp, fp, fn, prec, rec = prf(flag, test["label"])
        alarm = "false alarm" if fp == 1 else "false alarms"
        print(f"{name}: caught {tp}/{tp + fn} events, {fp} {alarm} "
              f"(precision {prec:.2f}, recall {rec:.2f})")
    print("Next: add temp (flag only if temp also low) and re-score.")


if __name__ == "__main__":
    main()
