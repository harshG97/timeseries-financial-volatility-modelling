"""
XGBoost RFP (Random Forecast Periods) evaluation.

Reads the selected XGBoost hyperparameters from ``xgboost_validation_results.csv``
(one blueprint per cell), then evaluates each blueprint on every RFP window
using ``src/rfp_generator.py``.

For each window, the model is trained from scratch on all data up to ``fit_end``,
then predicts every day/week in the forecast window.

Outputs
-------
xgboost-model/outputs/rfp/
    xgboost_rfp_results.csv
    xgboost_rfp_summary.csv
    forecasts/{target}_{freq}_{exog}_{window_id}.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rfp_generator import RFPGenerator

# Import helper functions from xgboost_volatility
from importlib.util import spec_from_file_location, module_from_spec
_xgb_path = Path(__file__).resolve().parent / "xgboost_volatility.py"
_spec = spec_from_file_location("xgboost_volatility", _xgb_path)
_xgb_mod = module_from_spec(_spec)
sys.modules["xgboost_volatility"] = _xgb_mod
_spec.loader.exec_module(_xgb_mod)

XGBConfig = _xgb_mod.XGBConfig
train_model = _xgb_mod.train_model
predict = _xgb_mod.predict
metrics = _xgb_mod.metrics
feature_columns = _xgb_mod.feature_columns
make_dataset = _xgb_mod.make_dataset

OUT_DIR = Path(__file__).resolve().parent / "outputs"
RFP_OUT = OUT_DIR / "rfp"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]

def evaluate_window(window, config: XGBConfig, use_exog: bool, seed: int) -> tuple[dict, pd.DataFrame]:
    train_df = window.train
    forecast_df = window.forecast

    # 1. Determine features based on exog
    if use_exog:
        cols = feature_columns(train_df) # Uses all columns except date/ret
    else:
        # If no_exog, strictly exclude any external columns if they accidentally leaked in
        # (Though rfp_generator handles this, it's good to be safe)
        cols = ["ret_lag1", "ret_sq_lag1", "neg_ret_sq_lag1", "RV_5_lag1", "RV_10_lag1", "RV_22_lag1"]
        cols = [c for c in cols if c in train_df.columns]

    # 2. Extract arrays
    train_x, train_y, _, _ = make_dataset(train_df, cols)
    test_x, test_y, test_dates, test_ret = make_dataset(forecast_df, cols)

    # 3. Train Model
    model = train_model(train_x, train_y, config, seed)

    # 4. Forecast
    pred_var_path = predict(model, test_x)
    
    # 5. Assemble forecast frame
    rows = []
    n_forecast = len(forecast_df)
    for step in range(n_forecast):
        pred_var = float(pred_var_path[step])
        sigma = float(np.sqrt(pred_var))
        actual_ret_pct = float(test_ret[step])
        actual_rv = float(test_y[step])

        rows.append({
            "date": test_dates[step],
            "ret_pct": actual_ret_pct,
            "realized_var": actual_rv,
            "pred_var": pred_var,
            "pred_vol": sigma,
            "VaR_1": float(NormalDist().inv_cdf(0.01) * sigma),
            "VaR_5": float(NormalDist().inv_cdf(0.05) * sigma),
        })

    fc_frame = pd.DataFrame(rows)
    m = metrics(test_y, pred_var_path, test_ret)
    
    m.update({
        "target": window.target,
        "freq": window.freq,
        "exog": "with_exog" if use_exog else "no_exog",
        "window_id": window.window_id,
        "regime": window.regime,
        "n_train": window.n_train,
        "n_forecast": n_forecast,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "n_estimators": config.n_estimators
    })

    return m, fc_frame

def run(args: argparse.Namespace) -> None:
    val_path = OUT_DIR / "xgboost_validation_results.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation results not found at {val_path}. Run xgboost_volatility.py first.")

    val_df = pd.read_csv(val_path)
    gen = RFPGenerator()

    (RFP_OUT / "forecasts").mkdir(parents=True, exist_ok=True)
    all_results = []
    
    cells = [(t, f, e) for t in TARGETS for f in FREQS for e in EXOGS]

    for target, freq, exog in tqdm(cells, desc="RFP cells"):
        match = val_df[(val_df["target"] == target) & (val_df["freq"] == freq) & (val_df["exog"] == exog)]
        if match.empty:
            continue

        row = match.iloc[0]
        config = XGBConfig(
            max_depth=int(row["max_depth"]),
            learning_rate=float(row["learning_rate"]),
            n_estimators=int(row["n_estimators"])
        )
        use_exog = (exog == "with_exog")

        windows = list(gen.iter_windows(freq=freq, target=target, use_exog=use_exog))

        for w in tqdm(windows, desc=f"  windows {target}/{freq}/{exog}", leave=False):
            m, fc_frame = evaluate_window(w, config, use_exog, args.seed)
            all_results.append(m)
            
            fc_path = RFP_OUT / "forecasts" / f"{target}_{freq}_{exog}_{w.window_id}.csv"
            fc_frame.to_csv(fc_path, index=False)

    if not all_results:
        print("No windows evaluated.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RFP_OUT / "xgboost_rfp_results.csv", index=False)

    summary = results_df.groupby(["target", "freq", "exog", "regime"])[["mse", "rmse", "mae", "qlike", "var_1_hit_rate"]].mean().reset_index()
    summary.to_csv(RFP_OUT / "xgboost_rfp_summary.csv", index=False)
    
    print(f"Done. Saved to {RFP_OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
