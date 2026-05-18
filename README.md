# Agricultural Spray Drone Flight Duration Dataset
### Sequential Machine Learning for Pre-Mission Flight Duration Prediction in Tropical Highland Operations

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)
[![Pipeline: v9](https://img.shields.io/badge/Pipeline-v9-green.svg)]()

> **Thesis:** *Sequential Machine Learning for Pre-Mission Flight Duration Prediction of Agricultural Spray Drones in Tropical Highland Operations*
> **Author:** Mc Sergel G. Cardaño — BS Computer Science, South Philippine Adventist College
> **Adviser:** Ms. Neila M. Paglinawan-Muñez, MSCS
> **Submitted:** May 2026

---

## Overview

This repository contains the dataset, supplementary output files, and trained model artifacts produced by the v9 pipeline for the thesis above. The study predicts `flight_duration_min` — the elapsed time in minutes from drone takeoff to landing — as a **pre-mission observable** using real telemetry from a single DJI AGRAS T40 agricultural spray drone (unit MKAVI1, serial RPU028B) operating in Bukidnon Province, Philippines from January to November 2024.

A key design principle is **mission-driven stopping**: the T40 lands when its assigned spray polygon is complete, not when the battery is exhausted. This makes `sprayed_area` the primary determinant of flight duration and makes prediction tractable from pre-mission inputs alone.

---

## Repository Structure

```
.
├── data/
│   ├── raw/
│   │   ├── auto_flights_raw.csv          # Raw DJI Smart Farm CSV export — AUTO mode
│   │   └── manual_flights_raw.csv        # Raw DJI Smart Farm CSV export — MANUAL mode
│   └── processed/
│       ├── final_dataset_auto.csv        # Cleaned AUTO dataset (6,446 flights, 12 features)
│       ├── final_dataset_manual.csv      # Cleaned MANUAL dataset (1,090 flights)
│       └── filtering_audit.csv           # Three-step data cleaning audit log
│
├── outputs_v9/
│   ├── baseline_results.csv             # Sequential model performance — GRU, LSTM, Seq-RF
│   ├── computational_efficiency.csv     # Training time, inference latency, memory footprint
│   ├── repetition_results.csv           # Stability across 5 random seeds
│   ├── ground_truth_and_predictions.csv # y_true + all model predictions (578 sequences)
│   ├── ablation_results.csv             # Environmental δMAE by season × feature
│   ├── ablation_baseline_control.csv    # Ablation baseline (fixed hyperparameters)
│   ├── sequential_permutation_importance.csv  # Permutation importance rankings
│   ├── flat_vs_seq_rf_comparison.csv    # Flat RF vs Seq-RF comparison
│   ├── manual_baseline_results.csv      # MANUAL mode flat RF baseline
│   ├── prediction_comparison_models.csv # All models on same 578 test sequences
│   ├── threshold_comparison.csv         # 1 min vs 2 min exclusion threshold experiment
│   ├── weather_alignment_audit.csv      # ERA5 merge quality report
│   ├── session_length_bias_auto.csv     # Session-length distribution audit (AUTO)
│   ├── session_length_bias_manual.csv   # Session-length distribution audit (MANUAL)
│   ├── hp_search_GRU.csv               # GRU hyperparameter search log
│   ├── hp_search_LSTM.csv              # LSTM hyperparameter search log
│   ├── baseline_pred_vs_actual.png     # Predicted vs actual scatter (all 3 models)
│   ├── delta_mae_heatmap.png           # δMAE heatmap — ablation (season × feature)
│   ├── mae_by_season.png               # GRU MAE stratified by season
│   └── feature_importance.png          # Environmental δMAE bar chart
│
├── saved_models/
│   ├── gru_model.pth                   # Trained GRU weights (PyTorch state_dict)
│   ├── lstm_model.pth                  # Trained LSTM weights (PyTorch state_dict)
│   ├── rf_model.pkl                    # Trained Seq-RF (scikit-learn, 500 trees)
│   ├── x_scaler.pkl                    # Feature StandardScaler (fitted on train only)
│   ├── y_scaler.pkl                    # Target StandardScaler (fitted on train only)
│   └── feature_cols.json               # Ordered 12-feature list for inference
│
├── notebooks/
│   └── v9_pipeline.ipynb               # Canonical pipeline notebook (all phases)
│
└── README.md
```

---

## Dataset

### Primary Dataset — MKAVI1 (Jan–Nov 2024)

| Attribute | Value |
|---|---|
| Drone model | DJI AGRAS T40 |
| Unit ID | MKAVI1 (serial RPU028B) |
| Location | Bukidnon Province, Philippines |
| Period | January 1 – November 30, 2024 |
| Raw records | 8,675 |
| Valid records (after cleaning) | 7,536 total → **6,446 AUTO / 1,090 MANUAL** |
| Sessions (AUTO) | 533 |
| Sequences produced (N=5, AUTO) | 3,848 |
| Target variable | `flight_duration_min` (minutes) |

### Data Source

Flight telemetry was exported from the **DJI Smart Farm** cloud platform as CSV files. ERA5 reanalysis weather variables were retrieved via the **Open-Meteo API** (no API key required, CC BY 4.0) for grid points covering Lantapan and Valencia municipalities, Bukidnon Province, matched to the floor-hour of each flight's start timestamp.

### ⚠️ Critical Unit Note — `sprayed_area`

DJI Smart Farm **CSV exports record `sprayed_area` in 亩 (mu), not hectares**. The web interface silently converts to hectares; the raw CSV does not. The v9 pipeline applies the correction immediately after CSV load:

```python
df["sprayed_area_mu_raw"] = df["sprayed_area"].copy()  # preserved for audit
df["sprayed_area"]        = df["sprayed_area"] / 15.0  # mu → ha
```

This is validated against the confirmed operational spray rate of 20 L/ha. All v8.x results are based on uncorrected (15× inflated) area values and must not be cited.

### Data Cleaning

Three sequential filters were applied:

| Filter Step | Removed | % | Remaining |
|---|---|---|---|
| Null required fields | 10 | 0.12% | 8,665 |
| Duration < 1.0 min (aborted/artifacts) | 1,128 | 13.02% | 7,537 |
| IQR outlier filter (bounds: −3.54 to 10.26 min) | 1 | 0.01% | 7,536 |
| **TOTAL** | **1,139** | **13.13%** | **7,536** |

MANUAL mode flights (1,090 records) were excluded from sequential model training because they cannot form the N=5 session windows required by GRU, LSTM, and Seq-RF. They are retained in `final_dataset_manual.csv` and used for a flat RF baseline.

---

## Feature Set (12 Features — v9 Canonical)

All features are observable **before** the flight begins.

| # | Feature | Category | Unit | Description |
|---|---|---|---|---|
| 1 | `starting_battery` | Operational | % | Battery charge at flight start |
| 2 | `sprayed_area` | Operational | ha | Assigned field polygon area (converted from mu) |
| 3 | `total_amount_l` | Operational | L | Liquid payload loaded for the flight |
| 4 | `hour_sin` | Temporal | — | sin(2π × hour / 24) |
| 5 | `hour_cos` | Temporal | — | cos(2π × hour / 24) |
| 6 | `month_sin` | Temporal | — | sin(2π × month / 12) |
| 7 | `month_cos` | Temporal | — | cos(2π × month / 12) |
| 8 | `flight_index_within_day` | Temporal | int ≥ 0 | Ordinal position of flight in the operational day |
| 9 | `rolling_duration_3` | Sequential | min | Rolling mean of the last 3 flight durations in the session |
| 10 | `temperature_2m` | Environmental | °C | ERA5 2-meter air temperature at flight start hour |
| 11 | `wind_speed_10m` | Environmental | km/h | ERA5 10-meter wind speed at flight start hour |
| 12 | `precipitation` | Environmental | mm/hr | ERA5 hourly precipitation at flight start hour |

**Dropped features (v9):** `relative_humidity_2m` (indirect mechanism), `day_of_week` (no physical link), `time_since_last_flight` (redundant with `starting_battery`). Wind direction decomposition (`wind_u`, `wind_v`) requires drone heading from DJI binary `.DAT` logs — reserved for future work.

---

## Models

### Sequential Models (AUTO flights only)

Each model receives a sliding window of **N=5 consecutive flights** from the same session.

| Model | Input Shape | MAE (min) | R² | Parameters | Inference |
|---|---|---|---|---|---|
| **GRU** | (n, 5, 12) | 0.847 | 0.640 | 15,041 | 0.6 ms |
| **LSTM** | (n, 5, 12) | 0.808 | 0.699 | 803,073 | 1.3 ms |
| **Seq-RF** | (n, 60) | 0.808 | 0.692 | 500 trees | 128.4 ms |
| Naive baseline | — | 2.078 | 0.000 | — | — |

All three models represent a 59–61% reduction in MAE over the naive mean baseline. Pairwise Wilcoxon signed-rank tests (n=578) confirm LSTM ≥ Seq-RF > GRU with statistical significance (α = 0.05). **GRU is recommended for deployment** due to its 0.6 ms inference latency and 54.4 MB memory footprint.

### Temporal Split

| Split | Period | Sequences |
|---|---|---|
| Train (70%) | Jan – Jul 2024 | — |
| Validation (15%) | Aug – Sep 2024 | — |
| Test (15%) | Oct – Nov 2024 | 578 |

Splits are strictly chronological to prevent data leakage. All reported metrics are from the held-out test set only.

---

## Key Findings

**Seasonal accuracy (GRU):**

| Season | Months | MAE (min) |
|---|---|---|
| Dry | Jan – Apr | 0.936 |
| Hot | May – Aug | 1.481 |
| Wet | Sep – Nov | 0.747 |

**Environmental ablation (δMAE = MAE after removal − baseline):**

| Context | −Precipitation | −Temperature | −Wind speed |
|---|---|---|---|
| AUTO-Dry | +0.056 ✓ | +0.025 ✓ | +0.029 ✓ |
| AUTO-Hot | −0.051 ✗ | −0.039 ✗ | −0.065 ✗ |
| AUTO-Wet | +0.006 ✓ | −0.020 ✗ | −0.006 ✗ |

✓ = feature is useful (removal hurts accuracy) &nbsp; ✗ = feature acts as noise in this context

**Top permutation importance (sequential models):**

| Rank | Feature | Mean δMAE (min) |
|---|---|---|
| 1 | `rolling_duration_3` | 0.062 |
| 2 | `sprayed_area` | 0.039 |
| 3 | `total_amount_l` | 0.021 |

---

## Reproducing the Results

### Requirements

```
python==3.10
pytorch==2.1.0
scikit-learn==1.3.0
pandas==2.1.0
numpy==1.24.0
xgboost
lightgbm
catboost
jupyterlab==4.0
requests  # for Open-Meteo ERA5 retrieval
```

### Running the Pipeline

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/drone-flight-duration-prediction.git
cd drone-flight-duration-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw DJI Smart Farm CSV exports in data/raw/
#    auto_flights_raw.csv and manual_flights_raw.csv

# 4. Open and run the pipeline notebook (cells in order)
jupyter lab notebooks/v9_pipeline.ipynb
```

The pipeline runs in seven sequential phases:

| Phase | Description | Key Output |
|---|---|---|
| 1 | Load, clean, assign sessions | `filtering_audit.csv`, `weather_alignment_audit.csv` |
| 2 | Context labels (season, mode) | — |
| 3 | Feature engineering + cyclical encoding | `final_dataset_auto.csv`, `final_dataset_manual.csv` |
| 4 | Session-aware window construction + temporal split | 3,848 sequences |
| 5 | Model training + hyperparameter search | `baseline_results.csv`, `saved_models/` |
| 6 | Ablation study (3 seasons × 3 features) | `ablation_results.csv` |
| 7 | Permutation importance + seasonal analysis | `sequential_permutation_importance.csv`, figures |

### Loading Trained Models for Inference

```python
import torch, joblib, json
import numpy as np

# Load artifacts
feature_cols = json.load(open("saved_models/feature_cols.json"))
x_scaler     = joblib.load("saved_models/x_scaler.pkl")
y_scaler     = joblib.load("saved_models/y_scaler.pkl")

# GRU inference example
# Input: numpy array of shape (5, 12) — one N=5 window, 12 features
window = np.array(...)   # shape (5, 12), features in feature_cols order
x_norm = x_scaler.transform(window)           # standardize
tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)  # (1, 5, 12)

gru_model = torch.load("saved_models/gru_model.pth")
gru_model.eval()
with torch.no_grad():
    pred_norm = gru_model(tensor).item()

pred_minutes = y_scaler.inverse_transform([[pred_norm]])[0][0]
print(f"Predicted flight duration: {pred_minutes:.2f} min")
```

---

## Hardware

All training and evaluation was performed on:

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 5 7000 Series (8 cores, 16 threads) |
| GPU | NVIDIA GeForce RTX 2050 (4 GB GDDR6) |
| RAM | 8 GB DDR5 |
| CUDA | 11.8 + cuDNN 8.9.0 |

Seq-RF runs on CPU only (no GPU required).

---

## Citation

If you use this dataset or pipeline in your research, please cite:

```bibtex
@thesis{cardano2026drone,
  title   = {Sequential Machine Learning for Pre-Mission Flight Duration
             Prediction of Agricultural Spray Drones in Tropical Highland Operations},
  author  = {Cardaño, Mc Sergel G.},
  school  = {South Philippine Adventist College},
  year    = {2026},
  address = {Camanchiles, Matanao, Davao del Sur, Philippines},
  type    = {Undergraduate Thesis}
}
```

---

## License

The dataset and supplementary files in this repository are released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The pipeline code (`v9_pipeline.ipynb`) is released under the [MIT License](https://opensource.org/licenses/MIT).

The ERA5 weather data accessed via Open-Meteo is subject to the [Copernicus Climate Change Service Terms](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf). Proper attribution: Hersbach et al. (2020), *The ERA5 global reanalysis*, QJRMS. DOI: [10.1002/qj.3803](https://doi.org/10.1002/qj.3803).

---

## Contact

**Mc Sergel G. Cardaño**
BS Computer Science — South Philippine Adventist College
Km. 68 Camanchiles, Matanao, Davao del Sur 8003, Philippines
📧 kentgats@gmail.com

**Research Adviser:** Ms. Neila M. Paglinawan-Muñez, MSCS
South Philippine Adventist College, Computer Science Department
