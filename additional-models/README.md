# Additional volatility models

Four extra volatility forecasters that mirror the methodology used by
[`lstm-model/lstm_volatility.py`](../lstm-model/lstm_volatility.py) so that the
results sit side-by-side with the LSTM baseline. Every script:

- Targets the same 12-cell grid (3 assets × 2 frequencies × 2 feature regimes)
  defined under [`data/splits/`](../data/splits/).
- Predicts one-step-ahead realized variance `(100 * ret)^2` (Prophet,
  SilverKite, and Orbit fit in **log** variance space and exponentiate, so
  predictions are always positive — Transformer learns it directly with a
  `Softplus` head, identical to the LSTM).
- Selects hyperparameters on the validation block (lowest QLIKE wins).
- Refits the chosen blueprint on `train + validation` and evaluates on the
  test block with **expanding cross-validation**, refitting every 20 daily
  steps / 4 weekly steps (same cadence as the LSTM run).
- Writes the same metrics (MSE, RMSE, MAE, QLIKE, 1% / 5% VaR exception
  rates), the same forecast-CSV columns, and the same four diagnostic PNGs
  per cell.

## Files in this directory

| File | What it does |
| --- | --- |
| [common.py](common.py) | Shared helpers: split loading, metric calculation, log/variance bookkeeping, residual columns, and the four diagnostic plots. Imported by the four model scripts. |
| [prophet_volatility.py](prophet_volatility.py) | **Meta Prophet** with optional lagged regressors, additive vs multiplicative seasonality, and a grid over `changepoint_prior_scale` / `seasonality_prior_scale`. Models log-realized-variance. |
| [silverkite_volatility.py](silverkite_volatility.py) | **LinkedIn SilverKite** via `greykite.sklearn.estimator.SimpleSilverkiteEstimator`. Grids over `fit_algorithm`, yearly Fourier order, and changepoint method. Daily uses freq `B`, weekly uses freq `W-FRI` (matches the data). |
| [orbit_volatility.py](orbit_volatility.py) | **Uber Orbit** Damped-Local-Trend (DLT) Bayesian state-space model with optional lagged regressors. Grids over estimator (`stan-map` / `stan-mcmc`), `damped_factor`, and global trend option. |
| [transformer_volatility.py](transformer_volatility.py) | **Encoder-only Transformer** (sinusoidal positional encoding, GELU FFN, `Softplus` head). Same lookback windowing and training loop as the LSTM, but with multi-head self-attention instead of recurrence. |
| [requirements.txt](requirements.txt) | Pip dependencies for everything in this directory. |
| `outputs/{prophet,silverkite,orbit,transformer}/` | Generated at runtime. Mirrors `lstm-model/outputs/`. |

## Setup

We recommend a fresh Python 3.10 environment (Greykite pins `pandas<2.0`,
which conflicts with the rest of the stack on Python 3.11+; the workaround
documented below works on 3.10).

```bash
conda create -n volmodels python=3.10 -y
conda activate volmodels
pip install -r additional-models/requirements.txt
# Greykite reinstalls pandas 1.5 — restore a 2.x build for the others.
pip install -U "pandas>=2.1"
```

`prophet` will compile its Stan model on first import (one-off, ~30 s).
`orbit-ml` likewise compiles its Stan models on first run.

GPU support for the Transformer through PyTorch:
- **CUDA** is auto-detected when available.
- **CPU** is the default fallback (and the recommended choice on Apple Silicon).
- **MPS** is opt-in via `--mps`. We benchmarked it on an M-series Mac with
  `torch==2.2.2`: MPS was 2–4× slower than CPU because PyTorch saturates all
  performance cores on CPU (~900% utilization) while MPS pays a kernel-launch
  overhead per op that small models / small batches can't amortize. Try
  `--mps` only after benchmarking it for your specific grid.

Pass `--cpu` to force CPU even on a CUDA box.

## Running

Each script accepts a `--targets / --freqs / --exogs` selection identical to
the LSTM CLI, plus model-specific tuning grids. Outputs land under
`additional-models/outputs/<model>/`.

### One-cell smoke tests (fast)

```bash
# Prophet, SPY weekly, target-only feature set
python additional-models/prophet_volatility.py \
    --targets SPY --freqs weekly --exogs no_exog \
    --seasonality-modes additive --changepoint-priors 0.05

# SilverKite, SPY weekly
python additional-models/silverkite_volatility.py \
    --targets SPY --freqs weekly --exogs no_exog \
    --fit-algorithms ridge --yearly-seasonality 0 --changepoint-methods none

# Orbit DLT, SPY weekly
python additional-models/orbit_volatility.py \
    --targets SPY --freqs weekly --exogs no_exog \
    --estimators stan-map --damped-factors 0.8 --trends linear

# Transformer, SPY daily, with-exog feature set
python additional-models/transformer_volatility.py \
    --targets SPY --freqs daily --exogs with_exog \
    --epochs 20 --lookbacks 22 --d-models 32 --nheads 4 \
    --num-layers 1 --dropouts 0.0 --learning-rates 0.001 --cpu
```

### Full 12-cell grid (default tuning grid)

```bash
python additional-models/prophet_volatility.py
python additional-models/silverkite_volatility.py
python additional-models/orbit_volatility.py
python additional-models/transformer_volatility.py
```

Add `--no-plots` to skip the four-PNG diagnostic bundle for each cell.

## What each run produces

For every model `<m>` in `{prophet, silverkite, orbit, transformer}`:

```
additional-models/outputs/<m>/
├── <m>_validation_results.csv   # one row per cell — metrics on val + chosen hyperparams
├── <m>_test_results.csv         # one row per cell — final test metrics + chosen hyperparams
├── run_config.json              # the CLI args used for the run
├── forecasts/
│   └── {TARGET}_{freq}_{exog}_test_forecasts.csv
│       # columns: date, ret_pct, realized_var, pred_var, pred_vol,
│       #          VaR_1, VaR_5, std_resid, squared_std_resid
└── plots/{TARGET}/{freq}/{exog}/
    ├── volatility_forecast_timeseries.png
    ├── standardized_residuals.png
    ├── acf_standardized_residuals.png
    └── acf_squared_standardized_residuals.png
```

These columns and plot names are byte-identical to the LSTM outputs, so the
two runs can be diffed directly (e.g. with `pandas.read_csv` + `.merge`).

### Metric columns (same set as LSTM)

`mse`, `rmse`, `mae`, `qlike`, `var_1_hit_rate`, `var_1_exceptions`,
`var_5_hit_rate`, `var_5_exceptions`, plus the chosen-config columns
(model-specific) and the cell identifiers (`target`, `freq`, `exog`).

### Plot semantics (same as LSTM)

- **`volatility_forecast_timeseries.png`** — full historical observed
  volatility (grey), test observed volatility (blue), test predicted
  volatility (red). Title is prefixed with the model name so screenshots are
  self-identifying.
- **`standardized_residuals.png`** — `ret / pred_vol` over the test block,
  with a zero line. Diagnostic for whether the model captures the magnitude
  of returns.
- **`acf_standardized_residuals.png`** — ACF of standardized residuals. Bars
  inside the blue confidence band ⇒ model captured the temporal dynamics.
- **`acf_squared_standardized_residuals.png`** — ACF of squared
  standardized residuals. Bars inside the band ⇒ model captured the
  conditional-variance dynamics (no remaining ARCH effects).

## Expected runtime (M-class CPU, full 12-cell grid, default grids)

| Script | Approx. runtime | Notes |
| --- | --- | --- |
| `prophet_volatility.py` | 30–60 min | cmdstanpy compiles once, then ~12 refits per cell. |
| `silverkite_volatility.py` | 60–120 min | Heaviest of the four; rich feature engineering. |
| `orbit_volatility.py` | 30–90 min | Stan MAP is fast; switch to `stan-mcmc` for posterior bands at higher cost. |
| `transformer_volatility.py` | 20–40 min CPU, <10 min on GPU | Same training loop as the LSTM; CUDA used automatically when available. |

For quicker iteration during development, restrict `--targets`, `--freqs`,
or `--exogs`; or shrink the model-specific grids (`--seasonality-modes`,
`--fit-algorithms`, `--estimators`, `--lookbacks`, etc.).

## Reproducibility

All scripts honor `--seed` (default `42`). Prophet, SilverKite, and Orbit
results are otherwise deterministic given fixed inputs; the Transformer
matches the LSTM seeding strategy (Python / NumPy / PyTorch).

## Comparing against the LSTM baseline

The forecast CSV schema is shared. A quick comparison:

```python
import pandas as pd
def load(model):
    return pd.read_csv(f"{model}/outputs/{m}_test_results.csv" if model == "lstm-model"
                       else f"additional-models/outputs/{m}/{m}_test_results.csv")
# … merge on (target, freq, exog) and diff the metric columns.
```

Because every model writes the same realized-variance target, the same
predicted-variance column, and the same VaR thresholds, downstream analysis
notebooks can stack all five model outputs into one tidy frame.
