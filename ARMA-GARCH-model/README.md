# ARMA-GJR-GARCH / GARCH-X Volatility Modeling

## Methodology

Models and forecasts volatility of SPY, OIL, and GOLD across the 12-cell
grid (3 targets × 2 frequencies × 2 feature regimes).

- **`no_exog`**: Constant-mean GJR-GARCH — univariate, uses only the target's
  own return history.
- **`with_exog`**: ARX-mean GJR-GARCH (GARCH-X) — cross-asset returns, DXY,
  and VIX enter the mean equation as exogenous regressors.

### Pipeline

1. **Grid search** (`garch_grid_search.py`): searches GARCH (p,o,q) on
   training data, selects by AIC.
2. **Validation + Test ECV** (`garch_validation_and_analysis.py`):
   walk-forward 1-step-ahead on validation (sanity check) and test
   (expanding cross-validation with every-step refitting).
3. **RFP evaluation** (`garch_rfp.py`): evaluates on RFP windows using
   `src/rfp_generator.py`.

## Quick start

### 1. Grid search (one cell)

```bash
python ARMA-GARCH-model/garch_grid_search.py --targets SPY --freqs daily --exogs no_exog
```

### 2. Full 12-cell grid search

```bash
python ARMA-GARCH-model/garch_grid_search.py
```

### 3. Validation + test evaluation

```bash
python ARMA-GARCH-model/garch_validation_and_analysis.py
```

### 4. RFP evaluation

```bash
python ARMA-GARCH-model/garch_rfp.py
```

### Data-config filtering

All three scripts support the same CLI filtering pattern:

| Flag | Values | Default |
|------|--------|---------|
| `--targets` | `SPY,OIL,GOLD` | all |
| `--freqs` | `daily,weekly` | all |
| `--exogs` | `no_exog,with_exog` | all |

RFP also supports `--regimes` (e.g. `GFC,COVID`).

## Outputs

Written to `ARMA-GARCH-model/outputs/`:

- `garch_grid_search_results.csv`: best (p,o,q) and AIC per cell
- `garch_validation_results.csv`: validation walk-forward metrics
- `garch_test_results.csv`: test ECV metrics (MSE, RMSE, MAE, QLIKE, VaR)
- `forecasts/{target}_{freq}_{exog}_test_forecasts.csv`: per-date predictions
- `plots/{target}/{freq}/{exog}/`: diagnostic plots per cell
- `rfp/garch_rfp_results.csv`: per-window RFP metrics
- `rfp/garch_rfp_summary.csv`: per-regime aggregates
- `rfp/forecasts/`: per-window forecast CSVs
- `rfp/plots/`: RFP diagnostic plots (per-window, regime bars, ablation,
  heatmap, boxplot)

Metrics: MSE, RMSE, MAE, QLIKE, 1%/5% VaR hit rates and exception counts.
