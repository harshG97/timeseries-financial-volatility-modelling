"""
MS-GARCH Walk-forward validation and test Expanding Cross-Validation (ECV).

Produces identical metrics (QLIKE, VaR, RMSE) and plots to the LSTM pipeline.
Refit cadence: 20 steps (daily), 4 steps (weekly).
"""

import argparse
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from tqdm.auto import tqdm

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

warnings.filterwarnings("ignore")
msgarch = importr("MSGARCH")
stats = importr("stats")

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

ENDO_COLS = {"ret", "date", "ret_lag1", "ret_sq_lag1", "neg_ret_sq_lag1",
             "RV_5_lag1", "RV_10_lag1", "RV_22_lag1"}


def exog_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ENDO_COLS]


def metrics(y_true: np.ndarray, pred_var: np.ndarray, returns_pct: np.ndarray) -> dict:
    pred_var = np.maximum(pred_var, 1e-8)
    errors = pred_var - y_true
    out = {
        "mse": float(np.mean(np.square(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "qlike": float(np.mean(np.log(pred_var) + y_true / pred_var)),
    }
    sigma = np.sqrt(pred_var)
    for level in [0.01, 0.05]:
        var_forecast = norm.ppf(level) * sigma
        hits = returns_pct < var_forecast
        out[f"var_{int(level * 100)}_hit_rate"] = float(np.mean(hits))
        out[f"var_{int(level * 100)}_exceptions"] = int(np.sum(hits))
    return out


def walk_forward_msgarch(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    spec_params: dict,
    refit_every: int
) -> pd.DataFrame:
    """Expanding window Walk-Forward for MS-GARCH."""
    exog = spec_params["exog"]
    use_exog = (exog == "with_exog")
    exog_cols = exog_columns(train_df) if use_exog else []

    all_data = pd.concat([train_df, eval_df])
    returns = all_data["ret"] * 100
    exog_data = all_data[exog_cols] if use_exog else None

    k = spec_params["k"]
    model_type = spec_params["model"]
    dist = spec_params["dist"]

    spec = msgarch.CreateSpec(
        variance_spec=robjects.ListVector({"model": robjects.StrVector([model_type] * k)}),
        distribution_spec=robjects.ListVector({"distribution": robjects.StrVector([dist] * k)}),
        switch_spec=robjects.ListVector({"do.mix": False})
    )

    preds = []
    regime_probs = []

    eval_start_idx = len(train_df)
    n_eval = len(eval_df)
    
    current_fit = None
    arx_model = None

    # We iterate over the evaluation period
    for i in tqdm(range(n_eval), leave=False):
        current_t = eval_start_idx + i
        
        # Need to refit?
        if i % refit_every == 0 or current_fit is None:
            train_y = returns.iloc[:current_t]
            
            if use_exog:
                train_x = exog_data.iloc[:current_t]
                arx_model = arch_model(train_y, x=train_x, mean="ARX", lags=0, vol="Constant").fit(disp="off")
                resids = arx_model.resid.values
            else:
                resids = train_y.values

            try:
                current_fit = msgarch.FitML(spec=spec, data=robjects.FloatVector(resids))
            except Exception as e:
                # If fit fails, fallback to previous fit if available
                if current_fit is None:
                    raise RuntimeError("Initial MSGARCH fit failed!") from e

        # Forecast step
        if use_exog:
            # Predict mean for step t
            # arch_model forecast API is brittle for multiple exogenous features.
            # Since it's a simple ARX with lags=0 (OLS), we can compute manually:
            x_oos = exog_data.iloc[current_t:current_t+1]
            mean_pred = arx_model.params['Const'] + sum(arx_model.params[col] * x_oos[col].iloc[0] for col in exog_cols)
            
            # Predict variance using MSGARCH (predicts based on the residuals seen during fit)
            # Actually, to predict step t, MSGARCH needs the residuals up to t-1.
            # But wait, MSGARCH::predict function predicts based on `data`.
            # We must pass the residuals up to t-1
            y_hist = returns.iloc[:current_t]
            x_hist = exog_data.iloc[:current_t]
            curr_arx = arch_model(y_hist, x=x_hist, mean="ARX", lags=0, vol="Constant").fit(disp="off")
            r_resids = robjects.FloatVector(curr_arx.resid.values)
        else:
            mean_pred = 0.0
            r_resids = robjects.FloatVector(returns.iloc[:current_t].values)

        # 1-step ahead prediction
        fc = stats.predict(object=current_fit, newdata=r_resids, nahead=1)
        # fc$vol is the volatility forecast
        vol_pred = fc.rx2("vol")[0]
        var_pred = vol_pred ** 2

        # Inferred state probabilities for step t-1 (filtered probabilities)
        state_probs = msgarch.State(object=current_fit, data=r_resids)
        # state_probs$FiltProb is a matrix: [T, k]
        filt_prob = np.array(state_probs.rx2("FiltProb"))
        last_prob = filt_prob[-1, :] # Probabilities at t-1
        
        # Pad with zeros if k=1 so we always return an array of length 2 for consistency in plotting
        p_regime1 = last_prob[0] if len(last_prob) > 0 else 1.0
        p_regime2 = last_prob[1] if len(last_prob) > 1 else 0.0

        preds.append({
            "date": returns.index[current_t],
            "ret": returns.iloc[current_t] / 100.0,
            "realized_variance": (returns.iloc[current_t])**2,
            "pred_variance": var_pred,
            "pred_volatility": vol_pred,
            "p_regime1": p_regime1,
            "p_regime2": p_regime2
        })

    df_preds = pd.DataFrame(preds)
    
    # Calculate VaR
    df_preds["std_resid"] = df_preds["ret"] * 100.0 / df_preds["pred_volatility"]
    df_preds["var_1"] = norm.ppf(0.01) * df_preds["pred_volatility"]
    df_preds["var_5"] = norm.ppf(0.05) * df_preds["pred_volatility"]
    
    return df_preds


def process_cell(target: str, freq: str, exog: str, spec_params: dict):
    print(f"\nEvaluating {target}/{freq}/{exog}")
    train_df = pd.read_csv(SPLIT_DIR / freq / exog / target / "train.csv", index_col="date", parse_dates=True)
    val_df = pd.read_csv(SPLIT_DIR / freq / exog / target / "val.csv", index_col="date", parse_dates=True)
    test_df = pd.read_csv(SPLIT_DIR / freq / exog / target / "test.csv", index_col="date", parse_dates=True)

    refit_every = 20 if freq == "daily" else 4

    print("  Val walk-forward...")
    val_preds = walk_forward_msgarch(train_df, val_df, spec_params, refit_every)
    val_met = metrics(val_preds["realized_variance"], val_preds["pred_variance"], val_preds["ret"] * 100)
    
    print("  Test walk-forward...")
    train_val_df = pd.concat([train_df, val_df])
    test_preds = walk_forward_msgarch(train_val_df, test_df, spec_params, refit_every)
    test_met = metrics(test_preds["realized_variance"], test_preds["pred_variance"], test_preds["ret"] * 100)

    # Save forecasts
    fc_dir = OUT_DIR / "forecasts"
    fc_dir.mkdir(parents=True, exist_ok=True)
    test_preds.to_csv(fc_dir / f"{target}_{freq}_{exog}_test_forecasts.csv", index=False)
    
    # Diagnostic plot
    plot_dir = OUT_DIR / "plots" / target / freq / exog
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_preds["date"], test_preds["pred_volatility"], label="Predicted Volatility")
    plt.plot(test_preds["date"], np.abs(test_preds["ret"] * 100), alpha=0.5, label="Observed Absolute Return")
    if spec_params["k"] > 1:
        ax2 = plt.gca().twinx()
        ax2.plot(test_preds["date"], test_preds["p_regime2"], color="red", alpha=0.3, label="P(Regime 2)")
        ax2.set_ylabel("Probability")
    plt.title(f"MS-GARCH Volatility Forecast - {target} ({freq}, {exog})")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plot_dir / "volatility_forecast_timeseries.png")
    plt.close()

    return {
        "target": target, "freq": freq, "exog": exog,
        "k": spec_params["k"], "model": spec_params["model"], "dist": spec_params["dist"],
        **val_met
    }, {
        "target": target, "freq": freq, "exog": exog,
        "k": spec_params["k"], "model": spec_params["model"], "dist": spec_params["dist"],
        **test_met
    }


def run():
    grid_res = pd.read_csv(OUT_DIR / "msgarch_grid_search_results.csv")
    
    val_results = []
    test_results = []

    for _, row in grid_res.iterrows():
        spec = {
            "target": row["target"],
            "freq": row["freq"],
            "exog": row["exog"],
            "k": int(row["k"]),
            "model": row["model"],
            "dist": row["dist"]
        }
        val_met, test_met = process_cell(row["target"], row["freq"], row["exog"], spec)
        val_results.append(val_met)
        test_results.append(test_met)

    pd.DataFrame(val_results).to_csv(OUT_DIR / "msgarch_validation_results.csv", index=False)
    pd.DataFrame(test_results).to_csv(OUT_DIR / "msgarch_test_results.csv", index=False)
    print("Done. Saved results and plots.")


if __name__ == "__main__":
    run()
