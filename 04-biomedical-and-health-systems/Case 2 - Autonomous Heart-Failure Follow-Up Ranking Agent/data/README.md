# Data Guide - Heart-Failure Follow-Up Ranking (Case 2)

The real patient file is already in this folder - all 299 rows, no download needed.

---

## Bundled seed

| File | What it is |
|---|---|
| `heart_failure_clinical_records.csv` | 299 heart-failure patients, 13 clinical features each |

Columns: `age`, `anaemia` (0/1), `creatinine_phosphokinase`, `diabetes` (0/1), `ejection_fraction` (%), `high_blood_pressure` (0/1), `platelets`, `serum_creatinine` (mg/dL), `serum_sodium`, `sex` (0/1), `smoking` (0/1), `time` (follow-up days), `DEATH_EVENT` (0/1 - whether the patient died in follow-up).

No values are missing. `DEATH_EVENT` is your answer key for scoring - a good call list should contain many of the 96 patients who later died. Do not train a black box on it and call it a day; the task is a ranked list a nurse can read.

---

## Primary source

**UCI Machine Learning Repository - Heart Failure Clinical Records** (DOI 10.24432/C5Z89R)

- **Page:** https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
- **Paper:** Chicco, D. & Jurman, G. *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.* BMC Med Inform Decis Mak 20, 16 (2020).
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) - sharing and adaptation allowed with credit.

---

## Loading example

```python
import pandas as pd

df = pd.read_csv("data/heart_failure_clinical_records.csv")
print(df["DEATH_EVENT"].value_counts())
print(df.groupby("DEATH_EVENT")[["ejection_fraction", "serum_creatinine"]].mean().round(2))
```

---

## Citation

Chicco, D. & Jurman, G. Heart Failure Clinical Records [Dataset]. UCI Machine Learning Repository (2020). https://doi.org/10.24432/C5Z89R (CC BY 4.0). File bundled verbatim.
