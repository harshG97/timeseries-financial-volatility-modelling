"""
Shared utilities used by every volatility-forecasting script in
``additional-models``.

The helpers here mirror the conventions established in
``lstm-model/lstm_volatility.py`` so that results and plots produced by the
Prophet / SilverKite / Orbit / Transformer scripts are *directly* comparable to
the LSTM baseline.

Key conventions (kept identical to the LSTM script):

- Target series is one-step-ahead realized variance ``(100 * ret)^2``.
- The 12-cell evaluation grid is ``TARGETS x FREQS x EXOGS``.
- Hyperparameters are tuned on the validation block (lowest QLIKE wins).
- Final evaluation uses expanding cross-validation on the test block, refitting
  every ``REFIT_CADENCE[freq]`` steps.
- Each forecast row exposes the same columns
  ``date, ret_pct, realized_var, pred_var, pred_vol, VaR_1, VaR_5,
  std_resid, squared_std_resid``.
- Diagnostic plots (timeseries, std-residuals, ACF, ACF^2) follow the LSTM
  layout under ``<outputs>/plots/{TARGET}/{freq}/{exog}/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]
REFIT_CADENCE = {"daily": 20, "weekly": 4}
VAR_LEVELS = (0.01, 0.05)


def load_cell(freq: str, exog: str, target: str) -> dict[str, pd.DataFrame]:
    """Load train/val/test CSVs for one cell of the 12-cell grid."""
    base = SPLIT_DIR / freq / exog / target
    frames: dict[str, pd.DataFrame] = {}
    for stage in ("train", "val", "test"):
        path = base / f"{stage}.csv"
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        frames[stage] = df.reset_index(drop=True)
    return frames


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All lagged regressor columns (i.e. everything but ``date`` and ``ret``)."""
    return [c for c in df.columns if c not in {"date", "ret"}]


def realized_variance(ret: pd.Series) -> np.ndarray:
    """Realized variance target: ``(100 * ret)^2`` in float32."""
    returns_pct = ret.to_numpy(dtype=np.float64) * 100.0
    return np.square(returns_pct).astype(np.float64)


def returns_pct(ret: pd.Series) -> np.ndarray:
    return (ret.to_numpy(dtype=np.float64) * 100.0)


def metrics(
    y_true: np.ndarray, pred_var: np.ndarray, returns_pct: np.ndarray
) -> dict[str, float]:
    """Identical metric set to the LSTM script."""
    pred_var = np.maximum(pred_var, 1e-8)
    errors = pred_var - y_true
    out: dict[str, float] = {
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


def add_residual_columns(forecast_df: pd.DataFrame) -> pd.DataFrame:
    out = forecast_df.copy()
    out["std_resid"] = out["ret_pct"] / np.maximum(out["pred_vol"], 1e-8)
    out["squared_std_resid"] = np.square(out["std_resid"])
    return out


def build_forecast_row(date: pd.Timestamp, ret_pct: float, realized_var: float, pred_var: float) -> dict:
    pred_var = float(max(pred_var, 1e-8))
    pred_vol = float(np.sqrt(pred_var))
    return {
        "date": pd.Timestamp(date).date().isoformat(),
        "ret_pct": float(ret_pct),
        "realized_var": float(realized_var),
        "pred_var": pred_var,
        "pred_vol": pred_vol,
        "VaR_1": float(NormalDist().inv_cdf(0.01) * pred_vol),
        "VaR_5": float(NormalDist().inv_cdf(0.05) * pred_vol),
    }


def plot_cell_diagnostics(
    frames: dict[str, pd.DataFrame],
    forecast_df: pd.DataFrame,
    out_dir: Path,
    target: str,
    freq: str,
    exog: str,
    model_label: str,
) -> None:
    """Mirror of ``lstm_volatility.plot_cell_diagnostics`` so plots line up."""
    plot_dir = out_dir / "plots" / target / freq / exog
    plot_dir.mkdir(parents=True, exist_ok=True)

    forecast_df = add_residual_columns(forecast_df)
    history = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    history = history.assign(
        realized_var=realized_variance(history["ret"]),
        realized_vol=lambda x: np.sqrt(x["realized_var"]),
    )
    forecast_plot = forecast_df.assign(
        date=pd.to_datetime(forecast_df["date"]),
        observed_vol=lambda x: np.sqrt(x["realized_var"]),
    )

    plt.figure(figsize=(13, 6))
    plt.plot(history["date"], history["realized_vol"], color="0.70", linewidth=0.8, label="Historical observed vol")
    plt.plot(
        forecast_plot["date"], forecast_plot["observed_vol"],
        color="#1f77b4", linewidth=1.1, label="Test observed vol",
    )
    plt.plot(
        forecast_plot["date"], forecast_plot["pred_vol"],
        color="#d62728", linewidth=1.1, label="Test predicted vol",
    )
    plt.title(f"{model_label} | {target} {freq} {exog}: observed vs predicted volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "volatility_forecast_timeseries.png", dpi=150)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(forecast_plot["date"], forecast_plot["std_resid"], color="#4c78a8", linewidth=0.9)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(f"{model_label} | {target} {freq} {exog}: standardized residuals")
    plt.xlabel("Date")
    plt.ylabel("Return / predicted volatility")
    plt.tight_layout()
    plt.savefig(plot_dir / "standardized_residuals.png", dpi=150)
    plt.close()

    max_lags = min(40, max(1, len(forecast_plot) // 4))
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(forecast_plot["std_resid"].dropna(), lags=max_lags, ax=ax)
    ax.set_title(f"{model_label} | {target} {freq} {exog}: ACF of standardized residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "acf_standardized_residuals.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(forecast_plot["squared_std_resid"].dropna(), lags=max_lags, ax=ax)
    ax.set_title(f"{model_label} | {target} {freq} {exog}: ACF of squared standardized residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "acf_squared_standardized_residuals.png", dpi=150)
    plt.close(fig)


def parse_selection(raw: str, allowed: list[str]) -> list[str]:
    if raw.lower() == "all":
        return allowed
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    bad = sorted(set(selected) - set(allowed))
    if bad:
        raise ValueError(f"Invalid values {bad}; allowed values are {allowed}")
    return selected


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    """Common selection-set arguments shared across all four scripts."""
    parser.add_argument("--targets", default="all", help="Comma list or 'all': SPY,OIL,GOLD")
    parser.add_argument("--freqs", default="all", help="Comma list or 'all': daily,weekly")
    parser.add_argument("--exogs", default="all", help="Comma list or 'all': no_exog,with_exog")
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic plots.")
    parser.add_argument("--seed", type=int, default=42)


def history_until(frames: dict[str, pd.DataFrame], step: int) -> pd.DataFrame:
    """Train + val + first ``step`` rows of test, used as 'history' at step ``step``."""
    pieces = [frames["train"], frames["val"]]
    if step > 0:
        pieces.append(frames["test"].iloc[:step])
    return pd.concat(pieces, ignore_index=True)


def cells_iterator(targets: Iterable[str], freqs: Iterable[str], exogs: Iterable[str]):
    for target in targets:
        for freq in freqs:
            for exog in exogs:
                yield target, freq, exog
