# LSTM volatility model

`lstm_volatility.py` implements the project-plan LSTM approach on the prepared
`data/splits` grid:

- 3 targets: `SPY`, `OIL`, `GOLD`
- 2 frequencies: `daily`, `weekly`
- 2 feature regimes: `no_exog`, `with_exog`

The model predicts one-step realized variance, `(100 * ret)^2`, using only the
lagged feature columns in each split CSV. It tunes hyperparameters on validation
data, refits the selected blueprint on train + validation, then evaluates test
data with expanding cross-validation. ML refits follow the split guide:
20 steps for daily data and 4 steps for weekly data.

## Run one cell quickly

```powershell
python .\lstm-model\lstm_volatility.py --targets SPY --freqs daily --exogs with_exog --epochs 20 --lookbacks 22 --hidden-sizes 32 --dropouts 0.0
```

## Run the full 12-cell grid

```powershell
python .\lstm-model\lstm_volatility.py
```

Outputs are written to `lstm-model/outputs/`:

- `lstm_validation_results.csv`: validation metrics and selected parameters
- `lstm_test_results.csv`: final expanding-test metrics
- `forecasts/*_test_forecasts.csv`: date-level realized variance, predicted
  variance, volatility, standardized residuals, and 1%/5% VaR forecasts
- `plots/{TARGET}/{freq}/{exog}/`: diagnostic plots for each model cell

Metrics include MSE, RMSE, MAE, QLIKE, and empirical 1%/5% VaR exception rates.

The script automatically uses CUDA when PyTorch can see a GPU. Pass `--cpu` to
force CPU execution.

Each plot directory contains:

- `volatility_forecast_timeseries.png`: historical observed volatility, test
  observed volatility, and test predicted volatility
- `standardized_residuals.png`: test return divided by predicted volatility
- `acf_standardized_residuals.png`: ACF of standardized residuals
- `acf_squared_standardized_residuals.png`: ACF of squared standardized residuals

To skip plots:

```powershell
python .\lstm-model\lstm_volatility.py --no-plots
```

## RFP (Random Forecast Periods) evaluation

`lstm_rfp.py` evaluates the validated LSTM blueprints on the RFP windows
defined in `data/splits/rfp/`. For each window the model is trained from
scratch on all data up to `fit_end` (using hyperparameters from
`lstm_validation_results.csv`), then predicts the full forecast window
without refitting.

### Quick example — one cell, one regime

```bash
python lstm-model/lstm_rfp.py --targets SPY --freqs daily --exogs no_exog --regimes GFC --cpu
```

### Run all 12 cells across all regimes

```bash
python lstm-model/lstm_rfp.py
```

### Data-config filtering

Three optional flags control which cells to evaluate. If a flag is omitted
(or set to `all`), all values for that dimension are included:

| Flag | Values | Default |
|------|--------|---------|
| `--targets` | `SPY,OIL,GOLD` | all |
| `--freqs` | `daily,weekly` | all |
| `--exogs` | `no_exog,with_exog` | all |

Use `--regimes` to restrict to specific RFP regimes (e.g. `GFC,COVID`).

### Outputs

Written to `lstm-model/outputs/rfp/`:

- `lstm_rfp_results.csv`: one row per (cell × window) with metrics and hyperparams
- `lstm_rfp_summary.csv`: per-cell per-regime mean/median aggregates
- `forecasts/{target}_{freq}_{exog}_{window_id}.csv`: per-day predictions
- `plots/per_window/`: predicted vs actual volatility per window
- `plots/regime_bars/`: mean QLIKE by regime per cell
- `plots/ablation/`: no_exog vs with_exog grouped bars per target×freq
- `plots/heatmap_qlike.png`: cell × regime QLIKE heatmap
- `plots/boxplot_regime_qlike.png`: QLIKE distribution by regime
