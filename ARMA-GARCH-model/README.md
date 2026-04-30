# ARMA-GJR-GARCH Volatility Modeling

This directory contains the scripts and results for fitting, validating, and analyzing ARMA-GJR-GARCH models on financial time series data.

## Methodology

The goal is to model and forecast the volatility of three key financial assets: SPY (S&P 500 ETF), OIL (WTI Crude Futures), and GOLD. The modeling approach follows the guidelines outlined in `data/splits/explanation.md`.

The core of the methodology is a two-stage process for each of the 12 modeling cells (3 targets × 2 frequencies × 2 feature sets):

1.  **Grid Search on Training Data**: An ARMA(p,q) - GJR-GARCH(p,o,q) model is fitted on the **training set**. A grid search is performed over a range of `p`, `o`, and `q` parameters. The best parameter combination is selected based on the Akaike Information Criterion (AIC).
2.  **Validation and Analysis**: The best model from the grid search is then used for an out-of-sample "sanity check" on the **validation set**. This involves forecasting on the validation data and calculating the Mean Squared Error (MSE). Residual analysis is also performed to check for model adequacy.

The final model is intended to be refit on the combined training and validation data before final evaluation on the test set.

## Data and Features

-   **Targets**: SPY, OIL, GOLD
-   **Frequencies**: `daily`, `weekly`
-   **Data Splits**: The data is split into `train`, `val`, and `test` sets. The splits are located in the `data/splits` directory.
-   **Feature Regimes**:
    -   `no_exog`: The model only uses the target's own return history.
    -   `with_exog`: The model uses the target's return history plus returns from the other assets and market indicators (VIX, DXY).

## Scripts

-   `garch_grid_search.py`: This script performs the initial grid search on the training data for each of the 12 modeling cells. It iterates through different `p`, `o`, and `q` values and saves the best parameters based on AIC.
-   `garch_validation_and_analysis.py`: This script takes the best parameters from the grid search, fits the model on the training data, evaluates its performance on the validation set, and performs residual analysis.

## Results

-   `garch_grid_search_results.csv`: This file contains the best `p`, `o`, and `q` parameters found for each modeling cell during the grid search on the training data.
-   `garch_validation_results.csv`: This file contains the out-of-sample performance metrics (MSE, Log-Likelihood, AIC, BIC) for each model when evaluated on the validation set.
-   `validation_analysis/`: This directory contains the residual analysis plots for each model, including:
    -   Standardized Residuals over time.
    -   ACF of Standardized Residuals.
    -   ACF of Squared Standardized Residuals.
