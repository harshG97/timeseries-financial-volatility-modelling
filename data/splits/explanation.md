# Train / Validation / Test Split Strategy

This document describes how the volatility-modeling dataset is partitioned for
model fitting, selection, and evaluation, and the layout of files under
`data/splits/`.

## 1. Data overview

Five aligned time series at two frequencies:

| Series | Role          | Notes                                  |
|--------|---------------|----------------------------------------|
| SPY    | target / exog | S&P 500 ETF, equity proxy              |
| OIL    | target / exog | WTI crude futures                      |
| GOLD   | target / exog | Gold (GLD/GC=F)                        |
| VIX    | exog only     | Implied volatility — never a target    |
| DXY    | exog only     | US Dollar Index                        |

Daily and weekly data span 2000-01-03 to 2026-04-29. We use 2004-01-01 onward
to ensure all five series are populated.

## 2. The 12-cell analysis grid

We evaluate every combination of:

- **Frequency**: daily, weekly
- **Feature regime**: no exogenous variables, with exogenous variables
- **Target**: SPY, OIL, GOLD

→ 2 × 2 × 3 = **12 modeling cells.** Calendar boundaries are identical across
all cells so cross-cell comparisons isolate one factor at a time
(frequency / feature regime / asset).

When a series is the target, it is excluded from its own exogenous set:
- target = SPY  → exog = {OIL, GOLD, VIX, DXY}
- target = OIL  → exog = {SPY, GOLD, VIX, DXY}
- target = GOLD → exog = {SPY, OIL, VIX, DXY}

## 3. Three-stage split

Each cell is sliced into three contiguous, non-overlapping blocks:

| Stage | Daily              | Weekly             | Purpose                             |
|-------|--------------------|--------------------|-------------------------------------|
| train | 2004-01-01 → 2021-12-31 | 2004-01-02 → 2018-12-31 | Fit candidate models, order/IC selection |
| val   | 2022-01-01 → 2023-12-31 | 2019-01-01 → 2019-12-31 | Out-of-sample model/feature/HP selection |
| test  | 2024-01-01 → 2026-04-29 | 2020-01-01 → 2026-04-29 | Final evaluation — touched **once** |

The weekly hold-out starts in 2020 to give VaR backtests adequate observation
count (~330 weeks). The daily hold-out is shorter in calendar terms (~580 days)
but has equivalent statistical power.

### How val is used by model class

| Model class       | Selection on train | Use of val                          |
|-------------------|--------------------|-------------------------------------|
| GARCH / GJR       | AIC/BIC            | Out-of-sample sanity check, then refit on train ∪ val |
| MS-GARCH          | AIC/BIC + log-lik  | Confirm regime count, then refit on train ∪ val |
| GARCH-X           | IC                 | Feature subset / regularisation tuning, refit on train ∪ val |
| ML models         | —                  | Full hyperparameter search, refit on train ∪ val |

The final model in every cell is refit on **train ∪ val** before evaluation on
test.

## 4. Test-set evaluation: Expanding Cross-Validation (ECV)

On the test slice we produce 1-step-ahead forecasts via walk-forward:

1. Fit model on `[start, t-1]`.
2. Forecast σ²(t) (and VaR(t) at 1% / 5% levels).
3. Append the actual observation r(t) to the training set.
4. Step forward; refit per the cadence below.

| Model class | Refit cadence (daily) | Refit cadence (weekly) |
|-------------|------------------------|-------------------------|
| GARCH / GJR | Every step             | Every step              |
| MS-GARCH    | Every 20 steps         | Every 4 steps           |
| GARCH-X     | Every 20 steps         | Every 4 steps           |
| ML models   | Every 20 steps         | Every 4 steps           |

ECV produces a continuous forecast track suitable for QLIKE / MSE scoring and
Kupiec / Christoffersen VaR backtests.

## 5. Robustness: Random Forecast Periods (RFP)

To verify performance is not driven by a single regime, we evaluate each cell
on multiple non-overlapping forecast windows drawn from **pre-test data only**
(strictly before the frequency's `test_start`), stratified across regimes:

| Regime      | Span                       |
|-------------|----------------------------|
| GFC         | 2007-07-01 → 2009-06-30    |
| OIL_CRASH   | 2014-06-01 → 2016-02-29    |
| COVID       | 2020-02-01 → 2020-06-30    |
| ENERGY_22   | 2022-01-01 → 2022-12-31    |
| CALM_17_19  | 2017-01-01 → 2019-12-31    |

A regime is usable for a frequency only if its full span lies before that
frequency's `test_start`. The build code attempts to draw `n_per_regime`
non-overlapping windows; if a regime is too short to fit that many, fewer are
produced and a notice is printed.

| Frequency | Usable regimes                                          | Target windows / regime | Window length | Actual total |
|-----------|---------------------------------------------------------|--------------------------|----------------|---------------|
| Daily     | GFC, OIL_CRASH, COVID, ENERGY_22, CALM_17_19 (all 5)    | 5                        | 60 trading days| 23 (COVID fits 3 non-overlapping) |
| Weekly    | GFC, OIL_CRASH, CALM_17_19 (3 — COVID and ENERGY_22 are in the weekly test set) | 3 | 26 weeks       | 9 |

For each window: fit on all data strictly before `forecast_start`, forecast the
window, score. Per-regime metrics (mean, median, worst-case) are reported
alongside the ECV headline numbers.

For weekly models, COVID and ENERGY_22 are evaluated *out-of-sample as part of
the ECV test set* (which begins 2020-01-01) rather than via RFP — this is
actually a stronger evaluation for those regimes since they appear in the
continuous walk-forward forecast track.

### Applying RFP across the 12 cells

RFP is evaluated for **every one of the 12 cells**. Window boundaries are
shared across cells — defined once in `rfp/daily_windows.csv` and
`rfp/weekly_windows.csv` — and what differs per cell is the feature set used
to fit and forecast:

- **no_exog cells**: at each window, fit on rows `[start, fit_end]` of
  `{freq}/no_exog/{TARGET}/train.csv` (target return + lagged target).
- **with_exog cells**: at each window, fit on rows `[start, fit_end]` of
  `{freq}/with_exog/{TARGET}/train.csv` (target return + lagged target +
  lagged exogenous features).

This produces a (cell × window) results matrix. The primary comparison drawn
from RFP is **no_exog vs with_exog within each regime**, holding target and
frequency fixed — this directly answers whether exogenous information improves
volatility forecasts and whether the improvement is regime-dependent.

If compute becomes a bottleneck (MS-GARCH and ML cells in particular), reduce
**windows per regime** before dropping cells. Preserving the no_exog ↔
with_exog ablation is the analytical priority.

## 6. Leakage and embargo

- All exogenous variables enter with `.shift(1)` minimum (no contemporaneous
  features).
- A **1-period embargo** separates `fit_end` from `forecast_start` in every
  RFP and ECV step. This guards against rolling-window features
  (e.g. 22-day realised volatility) inadvertently using day-of-forecast data.
- VIX is treated as a feature only. Using it as a target would be circular
  (it *is* the market's implied-volatility forecast).
- Daily and weekly panels are constructed independently from raw data — no
  cross-frequency leakage.

## 7. File structure

```
data/splits/
├── manifest.yaml                    # split boundaries, RFP config
├── explanation.md                   # this document
├── rfp/
│   ├── daily_windows.csv            # 23 rows
│   └── weekly_windows.csv           # 9 rows
├── daily/
│   ├── no_exog/
│   │   ├── SPY/   {train,val,test}.csv
│   │   ├── OIL/   {train,val,test}.csv
│   │   └── GOLD/  {train,val,test}.csv
│   └── with_exog/
│       ├── SPY/   {train,val,test}.csv
│       ├── OIL/   {train,val,test}.csv
│       └── GOLD/  {train,val,test}.csv
└── weekly/
    ├── no_exog/   (same structure as daily/no_exog)
    └── with_exog/ (same structure as daily/with_exog)
```

Total: 36 split CSVs + 2 RFP CSVs + 1 manifest + 1 explanation.

### CSV schemas

**No-exog cell** — `data/splits/{freq}/no_exog/{TARGET}/{stage}.csv`:

| column   | description                                 |
|----------|---------------------------------------------|
| date     | ISO date (trading day or week-ending date)  |
| ret      | log-return of target (the y variable)       |
| ret_lag1 | one-period-lagged log-return of target      |

**With-exog cell** — `data/splits/{freq}/with_exog/{TARGET}/{stage}.csv`:

| column           | description                                 |
|------------------|---------------------------------------------|
| date             | ISO date                                    |
| ret              | log-return of target (y)                    |
| ret_lag1         | lagged target return                        |
| {exog}_ret_lag1  | lagged log-returns of each exog price series|
| vix_log_lag1     | lagged log-VIX level                        |
| dxy_ret_lag1     | lagged DXY log-return                       |

Exog columns vary by target (the target is excluded from its own exog set).

**RFP window file** — `data/splits/rfp/{freq}_windows.csv`:

| column          | description                                     |
|-----------------|-------------------------------------------------|
| window_id       | unique window label, e.g. `d_GFC_1`             |
| regime          | one of GFC / OIL_CRASH / COVID / ENERGY_22 / CALM_17_19 |
| fit_end         | last date used for fitting                      |
| forecast_start  | first forecast date (= fit_end + 1 + embargo)   |
| forecast_end    | last forecast date                              |

## 8. Regenerating the splits

All split files are deterministic outputs of:

1. Raw CSVs in `data/daily/` and `data/weekly/`.
2. The boundaries in `data/splits/manifest.yaml`.
3. `02_build_splits.ipynb` (run end-to-end).

To regenerate after a change to the manifest or feature engineering, open and
run `02_build_splits.ipynb` from the repository root. This recomputes the
aligned panels, applies the manifest, and writes all 38 files under
`data/splits/`. Do not hand-edit split CSVs — they will be overwritten.
