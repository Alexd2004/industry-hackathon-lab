# Data Guide - Cardiac Weak-Pump Flag (Case 1)

A seed file is already in this folder. You can start without downloading the full MRI archive.

---

## Bundled seed

| File | What it is |
|---|---|
| `cardiac_mri_volumes_seed.csv` | 45 hearts, one row each: MRI volumes plus pathology group |

Columns: `patient_id`, `edv_ml` (full volume), `esv_ml` (empty volume), `ef_pct` (ejection fraction), `lv_mass_g` (muscle mass), `group` (`N` = healthy, `HYP` = thickened muscle, `HF` / `HF-I` = failing pump).

This seed is **synthetic, modelled on published Sunnybrook group statistics** (means and spreads below). EF is derived from EDV and ESV, so recompute it yourself as a check - it matches `ef_pct`.

| Group | n | EDV (ml) | EF (%) | LV mass (g) |
|---|---|---|---|---|
| N (healthy) | 9 | 115.7 ± 36.9 | 62.9 ± 3.7 | 130.3 ± 32.7 |
| HYP (thick muscle) | 12 | 114.4 ± 50.5 | 62.7 ± 9.2 | 175.9 ± 85.7 |
| HF (failing) | 12 | 233.7 ± 63.2 | 33.1 ± 13.1 | 193.7 ± 39.0 |
| HF-I (failing + scar) | 12 | 244.9 ± 86.0 | 32.0 ± 12.3 | 201.3 ± 45.2 |

Generation: `numpy` seed 42 - EDV and EF drawn per group, ESV derived as `EDV × (1 − EF/100)`, mass drawn per group. Failing hearts (`HF`, `HF-I`) are the ones to catch; the `group` column is your answer key, not an input.

---

## Primary source (full real data)

**Sunnybrook Cardiac Data** - 45 real cine-MRI studies (healthy, hypertrophy, heart failure with and without infarction), public-domain licence, via the Cardiac Atlas Project.

- **Info:** https://www.cardiacatlas.org/sunnybrook-cardiac-data/
- **Challenge paper:** Radau et al., *Evaluation Framework for Algorithms Segmenting Short Axis Cardiac MRI*, MIDAS Journal - http://hdl.handle.net/10380/3070
- **Access:** free CAP account required for the NIfTI images - request it early if you want pixels.

Do **not** commit the full image archive to GitHub. Extra files belong in `data/raw/` (gitignored).

---

## Loading example

```python
import pandas as pd

df = pd.read_csv("data/cardiac_mri_volumes_seed.csv")
df["ef_check"] = (df["edv_ml"] - df["esv_ml"]) / df["edv_ml"] * 100
print(df["group"].value_counts())
print(df.groupby("group")["ef_pct"].mean().round(1))
```

---

## Citation

Radau, P. et al. Sunnybrook Cardiac Data (2009 Cardiac MR Left Ventricle Segmentation Challenge), via the Cardiac Atlas Project, https://www.cardiacatlas.org/sunnybrook-cardiac-data/ (public domain). Seed file in this folder is synthetic, modelled on published Sunnybrook group statistics (numpy seed 42).
