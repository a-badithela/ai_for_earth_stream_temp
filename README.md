# DRB Stream Temperature LSTM — Student Project

Forecast daily stream water temperature at 5 monitoring sites in the **Delaware River Basin (DRB)** using LSTM models.  
You will build two variants:
- **Local LSTM** — one model trained per site
- **Global LSTM** — one model trained across all 5 sites simultaneously

---

## Sites

| Site ID | Associated Reservoir(s) | Notes |
|---------|------------------------|-------|
| 1573    | Cannonsville + Pepacton | Downstream of two reservoirs |
| 1571    | Cannonsville            | |
| 1565    | Cannonsville            | |
| 1450    | Pepacton                | |
| 1641    | Neversink               | |

The three reservoirs (Cannonsville, Pepacton, Neversink) supply drinking water to New York City. Their release schedules influence downstream stream temperatures, which in turn affect aquatic habitat and water quality.

---

## Data

Data files are **not included in this repository**. Download them from Google Drive and place them in the `data/` folder:

> **Google Drive link**: 

Each site has its own `.npz` file named `{site_id}.npz` (e.g., `data/1573.npz`).

### NPZ file contents

| Key | Shape | Description |
|-----|-------|-------------|
| `pretrain_X` | (12754, 13) | Input features, 1985-05-01 → 2020-04-01 |
| `pretrain_Y` | (12754,) | Target: next-day stream temp (°C), process-based model |
| `finetune_X` | (~13132, 13) | Input features, 1985-05-01 → 2021-04-14 |
| `finetune_Y` | (~13132,) | Target: next-day stream temp (°C), observed + infilled |
| `finetune_Y_mask` | (~13132,) | `True` where target is dwallin-infilled (not a real observation) |
| `forecast_E{0..30}_X` | (708, 13) | Forecast period inputs — 31 NOAA GEFS weather ensembles |
| `forecast_E{0..30}_Y` | (708,) | Forecast period targets |
| `forecast_dates` | (708,) | Datetime index for forecast rows |
| `forecast_infill_mask` | (708,) | `True` where forecast context used infilled stream temp |
| `forecast_pb_temp` | (708,) | Fully process-based temperature (no observed data) |

### Input features (15 columns in X)

| Col | Name | Description |
|-----|------|-------------|
| 0 | `tmin` | Daily min air temperature (°C, GridMET) |
| 1 | `tmax` | Daily max air temperature (°C, GridMET) |
| 2 | `srad` | Solar radiation (W/m², GridMET) |
| 3 | `pr` | Precipitation (mm, GridMET) |
| 4 | `vs` | Wind speed (m/s, GridMET) |
| 5 | `rmax` | Relative humidity max (%, GridMET) |
| 6 | `rmin` | Relative humidity min (%, GridMET) |
| 7 | `rmean` | Relative humidity mean (%, GridMET) |
| 8 | `spillway` | Spillway release from the site's closest reservoir (cms) |
| 9 | `releases` | Controlled (operational) release from the site's closest reservoir (cms) |
| 10 | `seg_slope` | Stream segment slope — **static** |
| 11 | `seg_elev` | Segment elevation — **static** |
| 12 | `seg_width` | Segment width — **static** |
| 13 | `seg_length` | Segment length — **static** |
| 14 | `stream_temp` | Previous day's stream temperature — **ar1 feature, MASKED at forecast steps** |

**Target Y**: next-day stream temperature (°C), scalar per row.

---

## Forecasting windows and data leakage

This is a **time-series forecasting** task, which requires careful handling of what information the model is allowed to see.

### Lookback window
The LSTM receives 360 days of historical context before making a prediction. Each training sample is a sliding window:
```
days 0–359: historical context → LSTM input
day 360:    prediction target  → model output
```

### The ar1 feature and masking
Column 14 (`stream_temp`) is the **autoregressive feature (ar1)** — the previous day's observed stream temperature. During the 360-day context window the model can see real past temperatures, which is legitimate. However, at the **forecast step** (day 360), the true temperature is what we are trying to predict — it must not be passed as input.

`data_loader.py` automatically zeros column 14 at all forecast-step positions before returning a batch. If you build your own data loading, you must do the same.

### Train/validation split
Always split time-series data **chronologically**, not randomly. The training set must contain only earlier dates; the validation set must contain only later dates. Random splits allow future information to leak into training, artificially inflating metrics.

---

## Repository structure

```
drb-stream-temp-student/
├── README.md               this file
├── config.yaml             site IDs, time splits, feature names, default parameters
├── data_loader.py          Dataset and DataLoader utilities (provided)
├── 01_explore_data.ipynb   Walk through .npz file structure and statistics
├── 02_visualize_sites.ipynb Time series plots, seasonal patterns, correlations
├── requirements.txt        Python dependencies
├── .gitignore
└── data/                   Place .npz files here (not committed to git)
```

---

## Setup

**Option A — conda:**
```bash
conda create -n drb-lstm python=3.10
conda activate drb-lstm
pip install -r requirements.txt
```

**Option B — pip (existing environment):**
```bash
pip install -r requirements.txt
```

---

## Tasks

### Task 1 — Local LSTM
Train a separate LSTM for each of the 5 sites.

1. Use `StreamTempDataset` and `get_dataloaders` from `data_loader.py` to create training and validation sets for one site.
2. Implement your LSTM model (number of layers, hidden size, different architectures!). 
3. Normalize your features: fit a z-score scaler on the **training set only**, then apply it to train, val, and any test data.
4. Train with MSE or RMSE loss. Evaluate on the validation set.
5. Report RMSE for each forecast horizon (1-day, 4-day, 8-day ahead) per site on the forecast data. (You will need to prepare the forecast similar to how the train set is prepared).
6. Compare performance across sites — does the model perform better at some sites? Why?

### Task 2 — Global LSTM
Train a single LSTM that generalizes across all 5 sites.

2. Design a strategy for site identity: options include a random vector per site, one-hot encoding concatenated to each timestep, or ignoring it entirely.
3. Train and evaluate using the same protocol as Task 1.
4. Compare global vs local LSTM performance. When does the global model win?

### Deliverable
- Model architecture description and training setup
- RMSE table for each site × forecast horizon (1, 4, 8 days) for both local and global LSTM

Discussion: 
- How do meteorological drivers and reservoir releases relate to model errors?
- Are all features necessary for this task? 
- Explore calculating loss on all targets vs observed targets only.
- Explore providing a flag to indicate if a given target is observed to the model.

---

## Data sources

- **Stream temperature observations**: USGS streamgauges in the Delaware River Basin
- **Meteorological drivers**: [GridMET](https://www.climatologylab.org/gridmet.html) — daily gridded surface meteorology for the contiguous US
- **Reservoir operations**: USGS reservoir release records (Cannonsville, Pepacton, Neversink)
- **Process-based baseline**: Dwallin stream temperature model (Zwart et al.)
- **Forecasted Weather Drivers**: [Global Ensemble Forecast System](https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast) (NOAA)
