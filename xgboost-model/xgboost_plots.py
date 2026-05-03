"""
Generates the standardized suite of diagnostic and RFP plots for the XGBoost model.
Uses the exact same plotting logic as GARCH to ensure 1:1 comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from statsmodels.graphics.tsaplots import plot_acf

ROOT = Path(__file__).resolve().parents[1]
XGB_DIR = ROOT / "xgboost-model"
OUT_DIR = XGB_DIR / "outputs"
RFP_OUT = OUT_DIR / "rfp"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]

def plot_cell_diagnostics(target, freq, exog):
    forecast_path = OUT_DIR / "forecasts" / f"{target}_{freq}_{exog}_test_forecasts.csv"
    if not forecast_path.exists():
        print(f"File not found: {forecast_path}")
        return

    # Load history to match garch code
    base = ROOT / "data" / "splits" / freq / exog / target
    train = pd.read_csv(base / "train.csv", parse_dates=["date"])
    val = pd.read_csv(base / "val.csv", parse_dates=["date"])
    history = pd.concat([train, val], ignore_index=True)
    history = history.assign(realized_vol=lambda x: np.abs(x["ret"]) * 100.0)

    fc = pd.read_csv(forecast_path, parse_dates=['date'])
    
    # Calculate standardized residuals matching garch code exactly
    fc["std_resid"] = fc["ret_pct"] / np.maximum(fc["pred_vol"], 1e-8)
    fc["squared_std_resid"] = np.square(fc["std_resid"])

    plot_dir = OUT_DIR / "plots" / target / freq / exog
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Observed vs predicted volatility
    plt.figure(figsize=(13, 6))
    plt.plot(history["date"], history["realized_vol"], color="0.70",
             linewidth=0.8, label="Historical |return|")
    plt.plot(fc["date"], np.sqrt(fc["realized_var"]), color="#1f77b4",
             linewidth=1.1, label="Test observed vol")
    plt.plot(fc["date"], fc["pred_vol"], color="#d62728",
             linewidth=1.1, label="Test predicted vol")
    plt.title(f"{target} {freq} {exog}: observed vs predicted volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "volatility_forecast_timeseries.png", dpi=150)
    plt.close()

    # 2. Standardized residuals
    plt.figure(figsize=(13, 5))
    plt.plot(fc["date"], fc["std_resid"], color="#4c78a8", linewidth=0.9)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(f"{target} {freq} {exog}: standardized residuals")
    plt.xlabel("Date")
    plt.ylabel("Return / predicted volatility")
    plt.tight_layout()
    plt.savefig(plot_dir / "standardized_residuals.png", dpi=150)
    plt.close()

    # 3. ACF plots
    max_lags = min(40, max(1, len(fc) // 4))
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(fc["std_resid"].dropna(), lags=max_lags, ax=ax)
    ax.set_title(f"{target} {freq} {exog}: ACF of standardized residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "acf_standardized_residuals.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(fc["squared_std_resid"].dropna(), lags=max_lags, ax=ax)
    ax.set_title(f"{target} {freq} {exog}: ACF of squared std residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "acf_squared_standardized_residuals.png", dpi=150)
    plt.close(fig)

def plot_rfp_regime_bars():
    rfp_summary_path = RFP_OUT / "xgboost_rfp_summary.csv"
    if not rfp_summary_path.exists():
        return
        
    df = pd.read_csv(rfp_summary_path)
    plot_dir = RFP_OUT / "plots" / "regime_bars"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for target in TARGETS:
        for freq in FREQS:
            for exog in EXOGS:
                cell = df[(df['target'] == target) & (df['freq'] == freq) & (df['exog'] == exog)]
                if cell.empty: continue
                
                agg = cell.groupby("regime")["qlike"].mean().sort_index()
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(agg.index, agg.values, color=plt.cm.Set2(np.linspace(0, 1, len(agg))), edgecolor="0.3")
                ax.set_title(f"XGBoost {target} {freq} {exog} — Mean QLIKE by Regime", fontweight="bold")
                ax.set_ylabel("QLIKE (Lower is Better)")
                ax.set_xlabel("Market Regime")
                plt.xticks(rotation=15)
                fig.tight_layout()
                fig.savefig(plot_dir / f"{target}_{freq}_{exog}.png", dpi=150)
                plt.close(fig)

def plot_rfp_ablation():
    rfp_summary_path = RFP_OUT / "xgboost_rfp_summary.csv"
    if not rfp_summary_path.exists():
        return
        
    df = pd.read_csv(rfp_summary_path)
    plot_dir = RFP_OUT / "plots" / "ablation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for target in TARGETS:
        for freq in FREQS:
            cell = df[(df['target'] == target) & (df['freq'] == freq)]
            if cell.empty or cell['exog'].nunique() < 2: continue
            
            pivot = cell.pivot_table(index="regime", columns="exog", values="qlike", aggfunc="mean").sort_index()
            x = np.arange(len(pivot.index))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(x - width/2, pivot["no_exog"], width, label="No Exogenous", color="#4c78a8")
            ax.bar(x + width/2, pivot["with_exog"], width, label="With Exogenous (VIX/OVX)", color="#f58518")
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index, rotation=15)
            ax.set_title(f"XGBoost {target} {freq} — Exogenous Features Ablation by Regime", fontweight="bold")
            ax.set_ylabel("Mean Q-LIKE (Lower is Better)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{target}_{freq}_exog_ablation.png", dpi=150)
            plt.close(fig)

if __name__ == "__main__":
    print("Generating Cell Diagnostics...")
    for t in TARGETS:
        for f in FREQS:
            for e in EXOGS:
                plot_cell_diagnostics(t, f, e)
                
    print("Generating RFP Plots...")
    plot_rfp_regime_bars()
    plot_rfp_ablation()
    
    print("All XGBoost plots successfully generated!")
