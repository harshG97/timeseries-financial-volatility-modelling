"""
XGBoost volatility forecasting for the 12 split cells in data/splits.

The model predicts one-period-ahead realized variance, defined as
100 * ret squared, from the lagged feature columns already built in the split
CSVs. Hyperparameters are selected on validation data, then the selected
blueprint is evaluated on test with expanding cross-validation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]
REFIT_CADENCE = {"daily": 20, "weekly": 4}
VAR_LEVELS = (0.01, 0.05)


@dataclass(frozen=True)
class XGBConfig:
    max_depth: int = 3
    learning_rate: float = 0.05
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def parse_selection(raw: str, allowed: list[str]) -> list[str]:
    """Parse a comma-separated selection or 'all' against an allowed list."""
    if raw.lower() == "all":
        return list(allowed)
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    bad = sorted(set(selected) - set(allowed))
    if bad:
        raise ValueError(f"Invalid values {bad}; allowed values are {allowed}")
    return selected

def load_cell(freq: str, exog: str, target: str) -> dict[str, pd.DataFrame]:
    base = SPLIT_DIR / freq / exog / target
    frames = {}
    for stage in ("train", "val", "test"):
        path = base / f"{stage}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        frames[stage] = df.reset_index(drop=True)
    return frames

def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"date", "ret"}]

def realized_variance(ret: pd.Series) -> np.ndarray:
    returns_pct = ret.to_numpy(dtype=np.float32) * 100.0
    return np.square(returns_pct).astype(np.float32)

def make_dataset(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = df[columns].to_numpy(dtype=np.float32)
    y = realized_variance(df["ret"])
    dates = df["date"].to_numpy()
    returns_pct = df["ret"].to_numpy(dtype=np.float32) * 100.0
    return x, y, dates, returns_pct

def train_model(train_x: np.ndarray, train_y: np.ndarray, config: XGBConfig, seed: int) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1
    )
    model.fit(train_x, train_y)
    return model

def predict(model: xgb.XGBRegressor, x: np.ndarray) -> np.ndarray:
    pred = model.predict(x)
    return np.maximum(pred, 1e-8)

def metrics(y_true: np.ndarray, pred_var: np.ndarray, returns_pct: np.ndarray) -> dict[str, float]:
    pred_var = np.maximum(pred_var, 1e-8)
    errors = pred_var - y_true
    out = {
        "mse": float(np.mean(np.square(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "qlike": float(np.mean(np.log(pred_var) + y_true / pred_var)),
    }
    sigma = np.sqrt(pred_var)
    for level in VAR_LEVELS:
        z = NormalDist().inv_cdf(level)
        var_forecast = z * sigma
        hits = returns_pct < var_forecast
        out[f"var_{int(level * 100)}_hit_rate"] = float(np.mean(hits))
        out[f"var_{int(level * 100)}_exceptions"] = int(np.sum(hits))
    return out

def tune_cell(freq: str, exog: str, target: str, grid: list[XGBConfig], seed: int) -> tuple[XGBConfig, dict[str, float]]:
    frames = load_cell(freq, exog, target)
    if "train" not in frames or "val" not in frames:
        raise RuntimeError(f"Missing data for {target}/{freq}/{exog}")
        
    cols = feature_columns(frames["train"])
    train_x, train_y, _, _ = make_dataset(frames["train"], cols)
    val_x, val_y, _, val_ret = make_dataset(frames["val"], cols)
    
    best_config = None
    best_row = None

    for config in grid:
        model = train_model(train_x, train_y, config, seed)
        pred = predict(model, val_x)
        row = metrics(val_y, pred, val_ret)
        row.update(asdict(config))
        if best_row is None or row["qlike"] < best_row["qlike"]:
            best_row = row
            best_config = config

    return best_config, best_row

def expanding_test_forecast(frames: dict[str, pd.DataFrame], columns: list[str], config: XGBConfig, freq: str, seed: int) -> pd.DataFrame:
    history = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    test = frames["test"].reset_index(drop=True)
    cadence = REFIT_CADENCE[freq]
    
    rows = []
    model = None

    for step in tqdm(range(len(test)), desc=f"ECV {freq}", leave=False):
        if model is None or step % cadence == 0:
            x_hist, y_hist, _, _ = make_dataset(history, columns)
            model = train_model(x_hist, y_hist, config, seed + step)

        forecast_context = test.iloc[[step]]
        x_step, y_step, dates, ret_step = make_dataset(forecast_context, columns)
        
        pred_var = predict(model, x_step)[0]
        rows.append({
            "date": pd.Timestamp(dates[0]).date().isoformat(),
            "ret_pct": float(ret_step[0]),
            "realized_var": float(y_step[0]),
            "pred_var": float(pred_var),
            "pred_vol": float(np.sqrt(pred_var)),
            "VaR_1": float(NormalDist().inv_cdf(0.01) * np.sqrt(pred_var)),
            "VaR_5": float(NormalDist().inv_cdf(0.05) * np.sqrt(pred_var)),
        })
        history = pd.concat([history, test.iloc[[step]]], ignore_index=True)

    return pd.DataFrame(rows)

def default_grid() -> list[XGBConfig]:
    depths = [2, 3, 5]
    lrs = [0.01, 0.05, 0.1]
    ests = [50, 100, 200]
    
    grid = []
    for d, lr, n in itertools.product(depths, lrs, ests):
        grid.append(XGBConfig(max_depth=d, learning_rate=lr, n_estimators=n))
    return grid

def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "forecasts").mkdir(parents=True, exist_ok=True)

    targets = parse_selection(args.targets, TARGETS)
    freqs = parse_selection(args.freqs, FREQS)
    exogs = parse_selection(args.exogs, EXOGS)
    grid = default_grid()

    selection_rows = []
    test_rows = []

    cells = list(itertools.product(targets, freqs, exogs))
    for target, freq, exog in tqdm(cells, desc="XGBoost cells"):
        print(f"Tuning {target}/{freq}/{exog} on validation...")
        try:
            best_config, val_row = tune_cell(freq, exog, target, grid, args.seed)
        except RuntimeError:
            continue
            
        val_row.update({"target": target, "freq": freq, "exog": exog})
        selection_rows.append(val_row)

        print(f"Refitting {target}/{freq}/{exog} and running expanding test forecast...")
        frames = load_cell(freq, exog, target)
        columns = feature_columns(frames["train"])
        forecast_df = expanding_test_forecast(frames, columns, best_config, freq, args.seed + 10_000)
        
        forecast_path = OUT_DIR / "forecasts" / f"{target}_{freq}_{exog}_test_forecasts.csv"
        forecast_df.to_csv(forecast_path, index=False)

        test_metric = metrics(
            forecast_df["realized_var"].to_numpy(),
            forecast_df["pred_var"].to_numpy(),
            forecast_df["ret_pct"].to_numpy(),
        )
        test_metric.update({
            "target": target,
            "freq": freq,
            "exog": exog,
            "forecast_file": str(forecast_path.relative_to(ROOT)),
            **asdict(best_config),
        })
        test_rows.append(test_metric)

    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "xgboost_validation_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT_DIR / "xgboost_test_results.csv", index=False)
    print(f"Done. Results saved under {OUT_DIR.relative_to(ROOT)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="all",
                        help="Comma list or 'all': SPY,OIL,GOLD (default: all)")
    parser.add_argument("--freqs", default="all",
                        help="Comma list or 'all': daily,weekly (default: all)")
    parser.add_argument("--exogs", default="all",
                        help="Comma list or 'all': no_exog,with_exog (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
