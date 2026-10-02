# Data Guide - Offshore Well Event Flag (Case 9)

A seed file is already in this folder. You can start without downloading the full research dataset.

---

## Bundled seed

| File | What it is |
|---|---|
| `well_hydrate_seed.csv` | 720 hourly sensor readings (30 days), one well, one event type (hydrate) |

Columns: `timestamp`, `pressure_bar`, `temp_C`, `flow_Lps`, `label` (`0` = normal, `1` = needs help).

This seed is **synthetic, modelled on real hydrate signatures** from the 3W Dataset below (pressure ramps down ~30 bar over 12–30 hours, temperature dips ~3°C, flow drops ~12%). Three events are planted (hours 150–174, 400–430, 600–612). 16 pressure readings (about 2%) are missing outside events - drop or impute them and say so.

Generation: `numpy` seed 42, baseline pressure 280 ± 2.5 bar, temp 85 ± 1°C, flow 12 ± 0.5 L/s.

---

## Primary source (full real data)

**Petrobras 3W Dataset 2.0.0** - real offshore well multivariate time series with expert-labelled undesirable events (hydrate, leakage, scaling, and more).

- **Repo:** https://github.com/petrobras/3W (Apache-2.0 toolkit)
- **Data:** https://doi.org/10.6084/m9.figshare.29205836.v1 (CC BY 4.0)
- **Paper:** Vargas et al., *Scientific Data* 13, 949 (2026) - https://doi.org/10.1038/s41597-026-07225-z
- **UCI mirror (v1, plain CSVs):** search "3W dataset" at https://archive.ics.uci.edu/

Do **not** commit the full dataset to GitHub. Extra files belong in `data/raw/` (gitignored).

---

## Loading example

```python
import pandas as pd

df = pd.read_csv("data/well_hydrate_seed.csv", parse_dates=["timestamp"])
print(df["label"].value_counts())
print(df.describe())
```

---

## Citation

Vaz Vargas, R.E. et al. 3W Dataset 2.0.0: a realistic and public dataset with rare undesirable real events in oil wells. *Sci Data* 13, 949 (2026). Seed file in this folder is synthetic, modelled on 3W hydrate signatures (numpy seed 42).
