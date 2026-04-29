### 1. The Core Strategy: The 12-Cell Grid

We don't have just one dataset; we evaluate models across 12 isolated scenarios to see exactly what drives performance.

* **Targets (3):** SPY, OIL, GOLD.
* **Frequencies (2):** Daily, Weekly.
* **Feature Sets (2):** `no_exog` (target only) vs. `with_exog` (target + lagged features like VIX and DXY).

---

### 2. The Normal Split: Train-Val-Test

This is the primary workflow for building and tuning the models.

* **Train:** The sandbox. Use this to fit base models and calculate initial selection metrics (like AIC/BIC for econometric models) or train the raw weights for machine learning architectures.
* **Validation:** The tuning ground. Use this strictly out-of-sample block to run hyperparameter grid searches, select feature subsets, and finalize the model architecture.
* **The "Train + Val" Refit:** *Critical step.* Once a model is finalized on the Validation set, you must refit it on the combined Train and Validation data before touching the Test set.
* **Test:** The vault. Evaluate the final models here using Expanding Cross-Validation (ECV). The model predicts one step ahead, actual data is revealed and appended to the training set, the model steps forward, and it periodically refits (e.g., every 20 days).

---

### 3. The RFP Split: Random Forecast Periods

This acts as a specialized stress test for historical market shocks (like the 2008 GFC or 2020 COVID crash).

* **The Standard Rule:** RFP windows are for **evaluation only**. Take the finalized model blueprint from the Normal Split, initialize a blank version, train it on all data strictly prior to the specific crash, and forecast the crash window.
* **The Goal:** Compare the `no_exog` model against the `with_exog` model within these windows to prove if features like the VIX actually act as an early-warning radar during structural breaks.

#### Optional Alternative Approach: Per-Window Selection

Instead of using one universal blueprint, the team can carve a mini Train-Val split strictly from the pre-RFP data to run hyperparameter searches and select a unique model architecture for each specific crash window.

> **⚠️ Disclaimer & Risks of Per-Window Selection:**
> 
> * **The "Oracle Problem":** In live trading, you don't know which regime is coming next, so you wouldn't know which specialized model to deploy tomorrow. The standard approach forces you to find one universally robust model.
> * **Data Starvation & Overfitting:** Pre-RFP data (especially for early windows like the 2008 GFC) is very small. Chopping 3 years of data into Train/Val to tune complex ML models will likely result in overfitting to a tiny, non-representative sample.
> * **Ruins the Ablation Study:** If the `no_exog` cell picks a Random Forest, and the `with_exog` cell picks Gradient Boosting for the same window, you can no longer compare them fairly. You won't know if performance differences came from the exogenous features or simply the change in model architecture.