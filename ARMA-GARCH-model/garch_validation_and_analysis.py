"""
GJR-GARCH / GARCH-X validation and test evaluation.

Reads best (p,o,q) from ``garch_grid_search_results.csv``, then for each cell:

1. **Validation**: walk-forward 1-step-ahead on val set (sanity check).
2. **Refit** on train + val.
3. **Test ECV**: expanding walk-forward 1-step-ahead on test set with
   every-step refitting (GARCH is fast).
4. Compute metrics: MSE, RMSE, MAE, QLIKE, VaR 1%/5% hit rates.
5. Save per-date forecast CSVs and diagnostic plots.

For ``no_exog``: Constant-mean GJR-GARCH.
For ``with_exog``: ARX-mean GJR-GARCH with cross-asset / market regressors.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]
VAR_LEVELS = (0.01, 0.05)

ENDO_COLS = {"ret", "date", "ret_lag1", "ret_sq_lag1", "neg_ret_sq_lag1",
             "RV_5_lag1", "RV_10_lag1", "RV_22_lag1"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_selection(raw: str, allowed: list[str]) -> list[str]:
    if raw.lower() == "all":
        return allowed
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    bad = sorted(set(selected) - set(allowed))
    if bad:
        raise ValueError(f"Invalid values {bad}; allowed values are {allowed}")
    return selected


def exog_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ENDO_COLS]


def load_cell(freq: str, exog: str, target: str) -> dict[str, pd.DataFrame]:
    base = SPLIT_DIR / freq / exog / target
    frames = {}
    for stage in ("train", "val", "test"):
        path = base / f"{stage}.csv"
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        frames[stage] = df.reset_index(drop=True)
    return frames


def realized_variance(ret: pd.Series) -> np.ndarray:
    return np.square(ret.to_numpy(dtype=np.float64) * 100.0)


def metrics(y_true: np.ndarray, pred_var: np.ndarray,
            returns_pct: np.ndarray) -> dict[str, float]:
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


def fit_garch(returns: pd.Series, exog_df: pd.DataFrame | None,
              p: int, o: int, q: int, use_exog: bool,
              starting_values=None):
    """Fit GJR-GARCH (or ARX-GJR-GARCH). Returns fitted result or None."""
    try:
        if use_exog and exog_df is not None and len(exog_df.columns) > 0:
            model = arch_model(
                returns, x=exog_df, mean="ARX", lags=0,
                p=p, o=o, q=q, vol="Garch", dist="ged",
            )
        else:
            model = arch_model(
                returns, mean="Constant",
                p=p, o=o, q=q, vol="Garch", dist="ged",
            )
        kw = {"disp": "off"}
        if starting_values is not None:
            kw["starting_values"] = starting_values
        return model.fit(**kw)
    except Exception:
        return None


def walk_forward(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    p: int, o: int, q: int,
    use_exog: bool,
    desc: str = "walk-forward",
) -> pd.DataFrame:
    """1-step-ahead expanding walk-forward forecast. Refits every step."""
    rows = []
    combined = history_df.copy()
    prev_params = None

    for step in tqdm(range(len(test_df)), desc=desc, leave=False):
        returns = combined["ret"] * 100
        exog_df = combined[exog_columns(combined)] if use_exog else None

        res = fit_garch(returns, exog_df, p, o, q, use_exog, prev_params)
        if res is None:
            # fallback: use unconditional variance
            pred_var = float(np.var(returns))
        else:
            prev_params = res.params.values
            # forecast 1-step ahead
            try:
                if use_exog:
                    next_row = test_df.iloc[[step]]
                    x_out = next_row[exog_columns(next_row)].values.reshape(1, -1)
                    fc = res.forecast(horizon=1, x=x_out, reindex=False)
                else:
                    fc = res.forecast(horizon=1, reindex=False)
                pred_var = float(fc.variance.iloc[-1, 0])
            except Exception:
                pred_var = float(np.var(returns))

        pred_var = max(pred_var, 1e-8)
        actual_ret_pct = float(test_df.iloc[step]["ret"]) * 100.0
        actual_rv = actual_ret_pct ** 2
        sigma = np.sqrt(pred_var)

        rows.append({
            "date": test_df.iloc[step]["date"],
            "ret_pct": actual_ret_pct,
            "realized_var": actual_rv,
            "pred_var": pred_var,
            "pred_vol": sigma,
            "VaR_1": float(NormalDist().inv_cdf(0.01) * sigma),
            "VaR_5": float(NormalDist().inv_cdf(0.05) * sigma),
        })

        # expand history
        combined = pd.concat([combined, test_df.iloc[[step]]], ignore_index=True)

    fc_df = pd.DataFrame(rows)
    # add standardized residuals
    fc_df["std_resid"] = fc_df["ret_pct"] / np.maximum(fc_df["pred_vol"], 1e-8)
    fc_df["squared_std_resid"] = np.square(fc_df["std_resid"])
    return fc_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cell_diagnostics(
    frames: dict[str, pd.DataFrame],
    forecast_df: pd.DataFrame,
    target: str, freq: str, exog: str,
) -> None:
    plot_dir = OUT_DIR / "plots" / target / freq / exog
    plot_dir.mkdir(parents=True, exist_ok=True)

    history = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    history = history.assign(
        realized_vol=lambda x: np.abs(x["ret"]) * 100.0,
    )
    fc = forecast_df.copy()
    fc["date"] = pd.to_datetime(fc["date"])

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "forecasts").mkdir(parents=True, exist_ok=True)

    # Load grid search results
    gs_path = Path(args.grid_search_csv)
    if not gs_path.is_absolute():
        gs_path = OUT_DIR / gs_path
    if not gs_path.exists():
        raise FileNotFoundError(
            f"Grid search CSV not found: {gs_path}\n"
            "Run garch_grid_search.py first."
        )
    gs_df = pd.read_csv(gs_path)

    targets = parse_selection(args.targets, TARGETS)
    freqs = parse_selection(args.freqs, FREQS)
    exogs = parse_selection(args.exogs, EXOGS)

    val_rows = []
    test_rows = []

    cells = list(itertools.product(targets, freqs, exogs))
    for target, freq, exog in tqdm(cells, desc="GARCH cells"):
        # look up best params
        match = gs_df[
            (gs_df["target"] == target)
            & (gs_df["freq"] == freq)
            & (gs_df["exog"] == exog)
        ]
        if match.empty:
            print(f"  ⚠ No grid search result for {target}/{freq}/{exog}, skipping.")
            continue

        row = match.iloc[0]
        p, o, q = int(row["best_p"]), int(row["best_o"]), int(row["best_q"])
        use_exog = exog == "with_exog"

        print(f"\n{'='*60}")
        print(f"Cell: {target}/{freq}/{exog}  (p={p}, o={o}, q={q})")
        print(f"{'='*60}")

        frames = load_cell(freq, exog, target)

        # ---- Stage 1: Validation walk-forward ----
        print("  Validation walk-forward...")
        val_fc = walk_forward(
            frames["train"], frames["val"], p, o, q, use_exog,
            desc=f"val {target}/{freq}/{exog}",
        )
        val_m = metrics(
            val_fc["realized_var"].values,
            val_fc["pred_var"].values,
            val_fc["ret_pct"].values,
        )
        val_m.update({"target": target, "freq": freq, "exog": exog,
                      "p": p, "o": o, "q": q})
        val_rows.append(val_m)
        print(f"  Val QLIKE={val_m['qlike']:.4f}  RMSE={val_m['rmse']:.4f}")

        # ---- Stage 2: Test ECV ----
        print("  Test expanding cross-validation...")
        history = pd.concat([frames["train"], frames["val"]], ignore_index=True)
        test_fc = walk_forward(
            history, frames["test"], p, o, q, use_exog,
            desc=f"test {target}/{freq}/{exog}",
        )

        # save forecast CSV
        fc_path = OUT_DIR / "forecasts" / f"{target}_{freq}_{exog}_test_forecasts.csv"
        test_fc.to_csv(fc_path, index=False)

        test_m = metrics(
            test_fc["realized_var"].values,
            test_fc["pred_var"].values,
            test_fc["ret_pct"].values,
        )
        test_m.update({
            "target": target, "freq": freq, "exog": exog,
            "p": p, "o": o, "q": q,
            "forecast_file": str(fc_path.relative_to(ROOT)),
        })
        test_rows.append(test_m)
        print(f"  Test QLIKE={test_m['qlike']:.4f}  RMSE={test_m['rmse']:.4f}")

        # ---- Plots ----
        if not args.no_plots:
            plot_cell_diagnostics(frames, test_fc, target, freq, exog)

    # ---- Save results ----
    pd.DataFrame(val_rows).to_csv(
        OUT_DIR / "garch_validation_results.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(
        OUT_DIR / "garch_test_results.csv", index=False
    )

    print(f"\nDone. Results saved under {OUT_DIR.relative_to(ROOT)}/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GJR-GARCH / GARCH-X validation and test evaluation."
    )
    p.add_argument("--targets", default="all", help="SPY,OIL,GOLD or 'all'")
    p.add_argument("--freqs", default="all", help="daily,weekly or 'all'")
    p.add_argument("--exogs", default="all", help="no_exog,with_exog or 'all'")
    p.add_argument(
        "--grid-search-csv", default="garch_grid_search_results.csv",
        help="Path to grid search results CSV (default: outputs/garch_grid_search_results.csv)",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
