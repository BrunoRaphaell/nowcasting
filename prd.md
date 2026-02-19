# PRD: Nowcasting de Inadimplência no Brasil

## 1. Overview & Motivation

Credit default rates (inadimplência) in Brazil are published with a lag relative to the macroeconomic indicators that drive them. By the time the BCB releases the official default rate for month *t*, several leading macro variables for that same period — or even *t+1* — are already available.

**Nowcasting** exploits this information gap: we use timely macro releases (employment, activity, exchange rate, monetary policy) to predict the current or next-month default rate before it is officially published.

### Target Variable

| Variable | BCB/SGS Code | Frequency | Description |
|---|---|---|---|
| Inadimplência PF total | **21084** | Monthly | Primary target — pessoa física, carteira total |
| Inadimplência PF recursos livres | **21112** | Monthly | Secondary target — pessoa física, recursos livres |

### Goals

- Build a reproducible pipeline that ingests public macro data and produces a nowcast for series 21084 (and optionally 21112).
- Compare baseline models (persistence, ARIMA) against macro-augmented models (SARIMAX, Elastic Net, XGBoost).
- Identify which macro variables and lag structures carry the most predictive power.

---

## 2. Data Sources & Series Codes

### 2.1 BCB/SGS API

**Base URL**: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json`

| Variable | Code | Native Frequency | Notes |
|---|---|---|---|
| Inadimplência PF total | 21084 | Monthly | Target (primary) |
| Inadimplência PF recursos livres | 21112 | Monthly | Target (secondary) |
| Inadimplência carteira total | 21082 | Monthly | Broader credit default rate |
| Selic acumulada no mês | 4390 | Monthly | Monetary policy rate |
| IBC-Br dessazonalizado | 24364 | Monthly | Economic activity proxy |
| Câmbio USD/BRL (PTAX venda) | 1 | Daily | Aggregated to monthly mean |

**Python library**: [`python-bcb`](https://github.com/wilsonfreitas/python-bcb) — no authentication required.

### 2.2 IBGE/SIDRA API

**Base URL**: `https://apisidra.ibge.gov.br/values/t/{table}/...`

| Variable | Table | Variable Code | Native Frequency | Notes |
|---|---|---|---|---|
| Taxa de desocupação (%) | 6381 | 4099 | Trimestral móvel | Forward-fill to monthly |
| Rendimento médio real habitual | 6390 | — | Trimestral móvel | Forward-fill to monthly |

**Python library**: [`sidrapy`](https://github.com/AlanJhonworker/sidrapy) — no authentication required.

### 2.3 Frequency Alignment

| Native Frequency | Alignment Strategy |
|---|---|
| Daily | Monthly mean |
| Trimestral móvel (PNAD) | Forward-fill: assign the quarter value to each of its 3 constituent months |
| Monthly | No transformation needed |

---

## 3. Tech Stack & Setup

### Python Version

- **Python ≥ 3.11**

### pyproject.toml Dependencies

```toml
[project]
name = "nowcasting"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "python-bcb>=0.2",
    "sidrapy>=0.1",
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "xgboost>=2.0",
    "statsmodels>=0.14",
    "pyyaml>=6.0",
    "matplotlib>=3.7",
    "seaborn>=0.13",
    "jupyterlab>=4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.4",
]
```

### Project Setup

```bash
# Clone / create the project directory
cd nowcasting

# Create virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 4. Project Structure

```
nowcasting/
├── prd.md                        # This document
├── pyproject.toml                # Project metadata & dependencies
├── configs/
│   └── series.yaml               # All series codes, names, frequencies, source
├── data/
│   ├── raw/                      # Immutable API responses (parquet/csv)
│   └── processed/                # Monthly panel ready for modeling
├── src/
│   ├── __init__.py
│   ├── ingestion.py              # Fetch BCB + IBGE raw data
│   ├── processing.py             # Frequency alignment, cleaning, merge into monthly panel
│   ├── features.py               # Lags, rolling means, deltas
│   ├── models.py                 # Baseline + ML model definitions
│   └── evaluation.py             # Walk-forward backtest + metrics
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_modeling.ipynb         # Model training & comparison
│   └── 03_results.ipynb          # Final results & visualizations
└── tests/
    ├── test_ingestion.py
    ├── test_processing.py
    └── test_features.py
```

---

## 5. Pipeline Steps

### Stage 1 — Ingestion (`src/ingestion.py`)

- Read series metadata from `configs/series.yaml`.
- For each series, call the appropriate API (`python-bcb` or `sidrapy`).
- Save raw responses to `data/raw/` as parquet files, one per series.
- Support date-range parameters to enable incremental updates.

### Stage 2 — Processing (`src/processing.py`)

- Load raw parquet files from `data/raw/`.
- Parse dates and set a `DatetimeIndex`.
- Align all series to **monthly** frequency:
  - Daily series → resample to monthly mean.
  - Trimestral móvel (PNAD) → forward-fill to monthly.
- Merge all series into a single `DataFrame` indexed by `year-month`.
- Save the result to `data/processed/panel.parquet`.

### Stage 3 — Feature Engineering (`src/features.py`)

For each explanatory variable, generate:

| Feature Type | Parameters |
|---|---|
| Lags | 1, 2, 3, 6, 12 months |
| Rolling means | 3, 6, 12-month windows |
| Deltas (Δ) | 1, 3, 6, 12-month differences |

**Leakage rule**: all features use `.shift(1)` at minimum — we never use contemporaneous values of explanatory variables to predict the target at time *t*.

### Stage 4 — Modeling (`src/models.py`)

| Model | Type | Description |
|---|---|---|
| Persistence | Baseline | ŷ_t = y_{t−1} |
| ARIMA(p,d,q) | Baseline | Univariate, auto-selected order |
| SARIMAX | Macro-augmented | ARIMA + exogenous macro features |
| Elastic Net | Macro-augmented | Linear model with L1+L2 regularization on lag/feature matrix |
| XGBoost | Macro-augmented | Gradient-boosted trees on lag/feature matrix |

### Stage 5 — Evaluation (`src/evaluation.py`)

- **Walk-forward expanding window**:
  - Minimum initial training window: **36 months**.
  - Step size: **1 month**.
  - At each step, fit on all data up to *t−1*, predict *t*.
- **Metrics** (computed over all out-of-sample predictions):
  - MAE
  - RMSE
  - Directional accuracy (% of months where predicted direction matches actual)
- **Scaler discipline**: any feature scaling (e.g., `StandardScaler`) is fit **only on the training fold** at each step.

---

## 6. Validation & Leakage Prevention

| Rule | Implementation |
|---|---|
| No contemporaneous features | All exogenous features use `.shift(1)` minimum |
| No future information in scaling | `StandardScaler.fit()` on train partition only |
| Expanding window only | No rolling/sliding window — always train from the start |
| Minimum history | First prediction requires ≥ 36 months of training data |
| Raw data immutability | `data/raw/` is never overwritten, only appended |

---

## 7. configs/series.yaml Schema

```yaml
targets:
  - name: inadimplencia_pf_total
    source: bcb
    code: 21084
    frequency: monthly
    description: "Inadimplência PF - carteira total"

  - name: inadimplencia_pf_livres
    source: bcb
    code: 21112
    frequency: monthly
    description: "Inadimplência PF - recursos livres"

features:
  - name: inadimplencia_carteira_total
    source: bcb
    code: 21082
    frequency: monthly

  - name: selic_acumulada_mes
    source: bcb
    code: 4390
    frequency: monthly

  - name: ibc_br_dessaz
    source: bcb
    code: 24364
    frequency: monthly

  - name: cambio_ptax_venda
    source: bcb
    code: 1
    frequency: daily

  - name: taxa_desocupacao
    source: ibge
    table: 6381
    variable: 4099
    frequency: quarterly

  - name: rendimento_medio_real
    source: ibge
    table: 6390
    frequency: quarterly
```

---

## 8. Deliverables & Success Criteria

### Deliverables

1. **Reproducible pipeline** — a single command (or notebook run-through) that goes from API fetch to evaluation results.
2. **Model comparison table** — MAE, RMSE, and directional accuracy for each model, over the walk-forward test set.
3. **Feature importance ranking** — which variables and lag structures contribute most (via XGBoost importance + Elastic Net coefficients).
4. **Notebooks** — EDA, modeling, and results notebooks for interactive exploration.

### Success Criteria

- At least one macro-augmented model achieves lower MAE than both baselines (persistence and ARIMA) on the walk-forward test set.
- Pipeline runs end-to-end without manual intervention.
- All API series codes are validated and pulling correct data.
