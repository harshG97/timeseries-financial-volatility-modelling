# XGBoost Volatility Model

This folder contains the complete implementation for the XGBoost volatility forecasting pipeline. It is designed to act as a Machine Learning benchmark against the statistical ARMA-GARCH models and the deep learning LSTM models.

## Pipeline Architecture
This code is structured exactly like the GARCH and LSTM models so we can easily compare the results. It includes the same hyperparameter tuning, walk-forward testing, and Random Forecast Period (RFP) evaluations.

### 1. `xgboost_volatility.py`
*   **Purpose:** The main script for tuning and testing the XGBoost model.
*   **Methodology:**
    *   Loads the engineered datasets (daily/weekly, exogenous/non-exogenous) for SPY, OIL, and GOLD.
    *   Performs a Grid Search over `max_depth` and `learning_rate` to optimize for Q-LIKE.
    *   Executes an expanding window (walk-forward) 1-step-ahead forecast on the test set.
*   **Outputs:** `xgboost_validation_results.csv`, `xgboost_test_results.csv`, and individual forecast CSVs in the `outputs/forecasts/` directory.

### 2. `xgboost_rfp.py`
*   **Purpose:** Evaluates the best XGBoost models on completely random historical windows (market regimes) to test out-of-sample generalization.
*   **Outputs:** `xgboost_rfp_summary.csv` and individual RFP forecasts.

### 3. `xgboost_plots.py`
*   **Purpose:** Generates a standardized suite of diagnostic plots identical to those used by the GARCH and LSTM models.
*   **Outputs:** 
    *   Timeseries Volatility Forecasts
    *   Standardized Residuals and ACF plots (max 40 lags)
    *   Regime Performance Bars
    *   Exogenous Feature Ablation Comparisons

### 4. `final_report_xgboost.ipynb`
*   **Purpose:** The final consolidated code deliverable. This single Jupyter Notebook contains the end-to-end pipeline required by the grading rubric.

## Additional Contributions (Root Directory)
*   **`src/plot_news_impact.py`**: Fits GJR-GARCH models across SPY, OIL, and GOLD to visually prove the existence of the asymmetrical "Leverage Effect" (justifying the inclusion of the `neg_ret_sq` feature in the ML models).
*   **`src/analyze_daily_weekly.py`**: Aggregates RFP results to compare forecasting performance between Daily and Weekly frequencies.
*   **`environment.yaml`**: Updated to include `xgboost` and `arch` dependencies.
