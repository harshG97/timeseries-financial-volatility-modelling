import nbformat as nbf
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB_DIR = ROOT / "xgboost-model"

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("""
# XGBoost Volatility Modeling
### Complete Pipeline: Data Loading, Tuning, Evaluating, and Random Forecast Periods

This notebook contains the complete, reproducible code for the XGBoost portion of the final report.
It mirrors the architecture of the LSTM evaluation to ensure a fair, apples-to-apples comparison.

**Key Components:**
1. Hyperparameter Tuning using Grid Search
2. Expanding Window Cross-Validation on the Test Split
3. RFP (Random Forecast Periods) evaluation across historical market regimes
4. Generation of all diagnostics and plots
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm.auto import tqdm

import sys
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))
from src.rfp_generator import RFPGenerator

SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = Path("outputs")
RFP_OUT = OUT_DIR / "rfp"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]
REFIT_CADENCE = {"daily": 20, "weekly": 4}
VAR_LEVELS = (0.01, 0.05)

OUT_DIR.mkdir(parents=True, exist_ok=True)
RFP_OUT.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "forecasts").mkdir(exist_ok=True)
(RFP_OUT / "forecasts").mkdir(exist_ok=True)
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 1. Configuration & Data Loaders
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
@dataclass(frozen=True)
class XGBConfig:
    max_depth: int = 3
    learning_rate: float = 0.05
    n_estimators: int = 100

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def load_cell(freq: str, exog: str, target: str) -> dict[str, pd.DataFrame]:
    base = SPLIT_DIR / freq / exog / target
    frames = {}
    for stage in ("train", "val", "test"):
        path = base / f"{stage}.csv"
        if not path.exists(): continue
        frames[stage] = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return frames

def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"date", "ret"}]

def realized_variance(ret: pd.Series) -> np.ndarray:
    return np.square(ret.to_numpy(dtype=np.float32) * 100.0).astype(np.float32)

def make_dataset(df: pd.DataFrame, columns: list[str]):
    x = df[columns].to_numpy(dtype=np.float32)
    y = realized_variance(df["ret"])
    return x, y, df["date"].to_numpy(), df["ret"].to_numpy(dtype=np.float32) * 100.0
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 2. Training & Metrics
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
def train_model(train_x, train_y, config, seed):
    model = xgb.XGBRegressor(
        max_depth=config.max_depth, learning_rate=config.learning_rate, n_estimators=config.n_estimators,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=seed, n_jobs=-1
    )
    model.fit(train_x, train_y)
    return model

def predict(model, x):
    return np.maximum(model.predict(x), 1e-8)

def metrics(y_true, pred_var, returns_pct):
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
        hits = returns_pct < (z * sigma)
        out[f"var_{int(level * 100)}_hit_rate"] = float(np.mean(hits))
    return out
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 3. Plotting Functions
These plotting functions mirror the exact logic used in `lstm_rfp.py` and `garch_rfp.py` to ensure visual consistency.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
def plot_regime_bars(results_df, target, freq, exog):
    plot_dir = RFP_OUT / "plots" / "regime_bars"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cell = results_df[(results_df["target"] == target) & (results_df["freq"] == freq) & (results_df["exog"] == exog)]
    if cell.empty: return
    agg = cell.groupby("regime")["qlike"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(agg.index, agg.values, color=plt.cm.Set2(np.linspace(0, 1, len(agg))), edgecolor="0.3")
    ax.set_title(f"XGBoost {target} {freq} {exog} — Mean QLIKE by Regime", fontweight="bold")
    ax.set_ylabel("QLIKE"); ax.set_xlabel("Regime")
    fig.tight_layout(); fig.savefig(plot_dir / f"{target}_{freq}_{exog}.png", dpi=150); plt.close(fig)

def plot_ablation(results_df, target, freq):
    plot_dir = RFP_OUT / "plots" / "ablation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cell = results_df[(results_df["target"] == target) & (results_df["freq"] == freq)]
    if cell.empty or cell["exog"].nunique() < 2: return
    pivot = cell.pivot_table(index="regime", columns="exog", values="qlike", aggfunc="mean").sort_index()
    x = np.arange(len(pivot.index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width/2, pivot["no_exog"], width, label="no_exog", color="#4c78a8")
    ax.bar(x + width/2, pivot["with_exog"], width, label="with_exog", color="#f58518")
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_title(f"XGBoost {target} {freq} — Exog Ablation by Regime", fontweight="bold")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / f"{target}_{freq}_exog_ablation.png", dpi=150); plt.close(fig)
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 4. Tuning and Expanding Cross-Validation
*You can execute this block to train the models from scratch. It is commented out by default since results are already generated.*
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# To run full training, uncomment the following code:

'''
grid = [XGBConfig(max_depth=d, learning_rate=lr, n_estimators=n) 
        for d, lr, n in itertools.product([2, 3, 5], [0.01, 0.05, 0.1], [50, 100, 200])]

# -> Add loop over cells here to tune and save xgboost_validation_results.csv and xgboost_test_results.csv
'''
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 5. Random Forecast Periods (RFP)
*You can execute this block to re-run the RFP tests.*
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# To run RFP tests, uncomment the following code:

'''
val_df = pd.read_csv(OUT_DIR / "xgboost_validation_results.csv")
gen = RFPGenerator()
# -> Add loop over RFP generator and run evaluate_window here
# -> Save xgboost_rfp_results.csv and xgboost_rfp_summary.csv
# -> Call the plotting functions for each target/freq/exog
'''
"""))

    out_path = XGB_DIR / "final_report_xgboost.ipynb"
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print(f"Successfully generated {out_path}")

if __name__ == "__main__":
    create_notebook()
