# Data Guide - Pipeline Incident Ranking (Case 10)

A 2015-through-August-2026 Alberta subset of CER Pipeline Incident Data is already in this folder.

---

## Bundled seed

| File | What it is |
|---|---|
| `cer_pipeline_incidents_alberta_2015.csv` | 313 Alberta incidents, 2015 through August 2026, 128 corridors |

Columns: `date`, `company` (short name), `corridor` (nearest town), `substance`, `release_m3`, `incident_type`, `cause`, `latitude`, `longitude`, `consequence`.

- **`company`:** shortened for readability (`NOVA Gas` covers NOVA Gas Transmission Ltd. and NGTL; `Keystone` is TransCanada Keystone; `Spectra` is Westcoast/Spectra). Full names are in the source file.
- **`corridor`:** nearest populated centre with the province suffix stripped (`Grand Prairie` fixed to `Grande Prairie`). Good enough to score hotspots; not a surveyed address.
- **`consequence`:** a **lab label**, not a regulator risk model.
  - `high` - sour gas or crude substance, or release of 100 m³ or more, or explosion / fatality
  - `medium` - other releases (sweet gas, NGLs, smaller spills)
  - `low` - fires or limit breaches with no substance released
- **Dropped rows:** 141 Alberta records with no usable occurrence date were excluded (mostly serious-injury filings with no timestamp). `release_m3` is empty where nothing was released.

---

## Primary source

**Canada Energy Regulator - Pipeline Incident Data (2008 to current, updated quarterly)**

- **CSV:** https://www.cer-rec.gc.ca/open/incident/pipeline-incidents-comprehensive-data.csv
- **Dictionary:** https://www.cer-rec.gc.ca/open/incident/pipeline-incidents-data-dictionary.csv
- **Portal:** https://open.canada.ca/data/en/dataset/7dffedc4-23fa-440c-a36d-adf5a6cc09f1
- **Licence:** [Open Government Licence - Canada](https://open.canada.ca/en/open-government-licence-canada)

Do not commit the full national file if you re-download it; this seed is Alberta, 2015 through August 2026.

---

## Loading example

```python
import pandas as pd

df = pd.read_csv("data/cer_pipeline_incidents_alberta_2015.csv", parse_dates=["date"])
print(df["consequence"].value_counts())
print(df["date"].dt.year.value_counts().sort_index())
```

---

## Citation

Canada Energy Regulator. Pipeline Incident Data. https://www.cer-rec.gc.ca/open/incident/pipeline-incidents-comprehensive-data.csv (Alberta slice, 2015 through August 2026, pulled September 2026).
