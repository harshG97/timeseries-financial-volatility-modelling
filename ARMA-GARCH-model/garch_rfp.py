"""
GJR-GARCH / GARCH-X RFP (Random Forecast Periods) evaluation.

Reads best (p,o,q) from ``garch_grid_search_results.csv``, then evaluates
each blueprint on every RFP window using ``src/rfp_generator.py``.

For each window: fit on all data up to ``fit_end`` (no refit within window),
then predict every day/week in the forecast period.

Outputs
-------
ARMA-GARCH-model/outputs/rfp/
    garch_rfp_results.csv
    garch_rfp_summary.csv
    forecasts/{target}_{freq}_{exog}_{window_id}.csv
    plots/  (per-window, regime bars, ablation, heatmap, boxplot)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rfp_generator import RFPGenerator

SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
RFP_OUT = OUT_DIR / "rfp"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]
VAR_LEVELS = (0.01, 0.05)

ENDO_COLS = {"ret", "date", "ret_lag1", "ret_sq_lag1", "neg_ret_sq_lag1",
             "RV_5_lag1", "RV_10_lag1", "RV_22_lag1"}


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


def evaluate_window(window, p: int, o: int, q: int,
                    use_exog: bool) -> tuple[dict, pd.DataFrame]:
    """Train GARCH on one RFP window, predict forecast period."""
    train_df = window.train
    forecast_df = window.forecast

    returns = train_df["ret"] * 100
    exog_df = train_df[exog_columns(train_df)] if use_exog else None

    # Fit on training data
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
        res = model.fit(disp="off")
    except Exception as e:
        # fallback: unconditional variance
        res = None

    # Predict each day in forecast window
    rows = []
    combined = train_df.copy()

    for step in range(len(forecast_df)):
        if res is not None:
            try:
                ret_hist = combined["ret"] * 100
                exog_hist = combined[exog_columns(combined)] if use_exog else None

                if use_exog and exog_hist is not None:
                    m = arch_model(
                        ret_hist, x=exog_hist, mean="ARX", lags=0,
                        p=p, o=o, q=q, vol="Garch", dist="ged",
                    )
                else:
                    m = arch_model(
                        ret_hist, mean="Constant",
                        p=p, o=o, q=q, vol="Garch", dist="ged",
                    )
                r = m.fit(disp="off", starting_values=res.params.values)

                if use_exog:
                    next_row = forecast_df.iloc[[step]]
                    x_out = next_row[exog_columns(next_row)].values.reshape(1, -1)
                    fc = r.forecast(horizon=1, x=x_out, reindex=False)
                else:
                    fc = r.forecast(horizon=1, reindex=False)
                pred_var = float(fc.variance.iloc[-1, 0])
                res = r  # update for warm start
            except Exception:
                pred_var = float(np.var(combined["ret"] * 100))
        else:
            pred_var = float(np.var(combined["ret"] * 100))

        pred_var = max(pred_var, 1e-8)
        actual_ret_pct = float(forecast_df.iloc[step]["ret"]) * 100.0
        actual_rv = actual_ret_pct ** 2
        sigma = np.sqrt(pred_var)

        rows.append({
            "date": forecast_df.iloc[step]["date"],
            "ret_pct": actual_ret_pct,
            "realized_var": actual_rv,
            "pred_var": pred_var,
            "pred_vol": sigma,
            "VaR_1": float(NormalDist().inv_cdf(0.01) * sigma),
            "VaR_5": float(NormalDist().inv_cdf(0.05) * sigma),
        })

        combined = pd.concat([combined, forecast_df.iloc[[step]]], ignore_index=True)

    fc_frame = pd.DataFrame(rows)
    fc_frame["std_resid"] = fc_frame["ret_pct"] / np.maximum(fc_frame["pred_vol"], 1e-8)
    fc_frame["squared_std_resid"] = np.square(fc_frame["std_resid"])

    m = metrics(
        fc_frame["realized_var"].values,
        fc_frame["pred_var"].values,
        fc_frame["ret_pct"].values,
    )
    m.update({
        "target": window.target,
        "freq": window.freq,
        "exog": "with_exog" if window.use_exog else "no_exog",
        "window_id": window.window_id,
        "regime": window.regime,
        "n_train": window.n_train,
        "n_forecast": len(fc_frame),
        "p": p, "o": o, "q": q,
    })

    return m, fc_frame


# ---------------------------------------------------------------------------
# Plotting (same structure as LSTM RFP)
# ---------------------------------------------------------------------------

def plot_per_window(fc_frame, target, freq, exog, window_id, regime):
    plot_dir = RFP_OUT / "plots" / "per_window" / target / freq / exog
    plot_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(fc_frame["date"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, np.sqrt(fc_frame["realized_var"]), color="#1f77b4",
            linewidth=1.2, label="Observed vol")
    ax.plot(dates, fc_frame["pred_vol"], color="#d62728",
            linewidth=1.2, label="Predicted vol")
    ax.set_title(f"{target} {freq} {exog} — {window_id} ({regime})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Volatility (%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{window_id}.png", dpi=150)
    plt.close(fig)


def plot_regime_bars(results_df, target, freq, exog):
    plot_dir = RFP_OUT / "plots" / "regime_bars"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cell = results_df[(results_df["target"] == target)
                      & (results_df["freq"] == freq)
                      & (results_df["exog"] == exog)]
    if cell.empty:
        return
    agg = cell.groupby("regime")["qlike"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(agg)))
    ax.bar(agg.index, agg.values, color=colors, edgecolor="0.3", linewidth=0.6)
    ax.set_title(f"{target} {freq} {exog} — Mean QLIKE by Regime",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("QLIKE"); ax.set_xlabel("Regime")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{target}_{freq}_{exog}.png", dpi=150)
    plt.close(fig)


def plot_ablation(results_df, target, freq):
    plot_dir = RFP_OUT / "plots" / "ablation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cell = results_df[(results_df["target"] == target) & (results_df["freq"] == freq)]
    if cell.empty or cell["exog"].nunique() < 2:
        return
    pivot = cell.pivot_table(index="regime", columns="exog",
                             values="qlike", aggfunc="mean").sort_index()
    regimes = pivot.index.tolist()
    x = np.arange(len(regimes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if "no_exog" in pivot.columns:
        ax.bar(x - width/2, pivot["no_exog"], width, label="no_exog",
               color="#4c78a8", edgecolor="0.3", linewidth=0.6)
    if "with_exog" in pivot.columns:
        ax.bar(x + width/2, pivot["with_exog"], width, label="with_exog",
               color="#f58518", edgecolor="0.3", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(regimes, rotation=30, ha="right")
    ax.set_ylabel("QLIKE (mean)")
    ax.set_title(f"{target} {freq} — Exog Ablation by Regime",
                 fontsize=11, fontweight="bold")
    ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir / f"{target}_{freq}_exog_ablation.png", dpi=150)
    plt.close(fig)


def plot_heatmap(results_df):
    plot_dir = RFP_OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = results_df.copy()
    df["cell"] = df["target"] + " / " + df["freq"] + " / " + df["exog"]
    pivot = df.pivot_table(index="cell", columns="regime",
                           values="qlike", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4),
                                    max(5, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                threshold = pivot.values[np.isfinite(pivot.values)].mean()
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if val > threshold else "0.15")
    ax.set_title("RFP Mean QLIKE — Cell × Regime", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="QLIKE")
    fig.tight_layout()
    fig.savefig(plot_dir / "heatmap_qlike.png", dpi=150)
    plt.close(fig)


def plot_boxplot_regime(results_df):
    plot_dir = RFP_OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    regimes = sorted(results_df["regime"].unique())
    data = [results_df[results_df["regime"] == r]["qlike"].dropna().values
            for r in regimes]
    fig, ax = plt.subplots(figsize=(max(7, len(regimes) * 1.2), 4.5))
    bp = ax.boxplot(data, tick_labels=regimes, patch_artist=True, notch=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(regimes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_edgecolor("0.3")
    ax.set_ylabel("QLIKE")
    ax.set_title("RFP QLIKE Distribution by Regime (all cells)",
                 fontsize=11, fontweight="bold")
    ax.set_xticklabels(regimes, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "boxplot_regime_qlike.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # Load grid search results
    gs_path = Path(args.grid_search_csv)
    if not gs_path.is_absolute():
        gs_path = OUT_DIR / gs_path
    if not gs_path.exists():
        raise FileNotFoundError(f"Grid search CSV not found: {gs_path}")

    gs_df = pd.read_csv(gs_path)
    gen = RFPGenerator()

    targets = parse_selection(args.targets, TARGETS)
    freqs = parse_selection(args.freqs, FREQS)
    exogs = parse_selection(args.exogs, EXOGS)
    regime_filter = (
        [r.strip() for r in args.regimes.split(",") if r.strip()]
        if args.regimes.lower() != "all"
        else None
    )

    (RFP_OUT / "forecasts").mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    cells = [(t, f, e) for t in targets for f in freqs for e in exogs]

    for target, freq, exog in tqdm(cells, desc="RFP cells"):
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

        windows = list(gen.iter_windows(
            freq=freq, target=target, use_exog=use_exog,
            regimes=regime_filter,
        ))

        for w in tqdm(windows, desc=f"  windows {target}/{freq}/{exog}", leave=False):
            m, fc_frame = evaluate_window(w, p, o, q, use_exog)
            all_results.append(m)

            fc_path = RFP_OUT / "forecasts" / f"{target}_{freq}_{exog}_{w.window_id}.csv"
            fc_frame.to_csv(fc_path, index=False)

            if not args.no_plots:
                plot_per_window(fc_frame, target, freq, exog, w.window_id, w.regime)

            print(f"    {w.window_id:20s}  QLIKE={m['qlike']:.4f}  "
                  f"RMSE={m['rmse']:.4f}  n_fc={m['n_forecast']}")

    if not all_results:
        print("No windows evaluated.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RFP_OUT / "garch_rfp_results.csv", index=False)

    # Summary
    group_cols = ["target", "freq", "exog", "regime"]
    metric_cols = ["mse", "rmse", "mae", "qlike", "var_1_hit_rate", "var_5_hit_rate"]
    available = [c for c in metric_cols if c in results_df.columns]
    summary = results_df.groupby(group_cols)[available].agg(["mean", "median"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RFP_OUT / "garch_rfp_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Results: {RFP_OUT.relative_to(ROOT)}/")
    print(f"  garch_rfp_results.csv ({len(results_df)} rows)")
    print(f"  garch_rfp_summary.csv ({len(summary)} rows)")

    if not args.no_plots:
        print("Generating aggregate plots...")
        for target in results_df["target"].unique():
            for freq in results_df["freq"].unique():
                for exog in results_df["exog"].unique():
                    plot_regime_bars(results_df, target, freq, exog)
                plot_ablation(results_df, target, freq)
        plot_heatmap(results_df)
        plot_boxplot_regime(results_df)
        print(f"  Plots: {(RFP_OUT / 'plots').relative_to(ROOT)}/")

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate GJR-GARCH / GARCH-X on RFP windows."
    )
    p.add_argument("--targets", default="all", help="SPY,OIL,GOLD or 'all'")
    p.add_argument("--freqs", default="all", help="daily,weekly or 'all'")
    p.add_argument("--exogs", default="all", help="no_exog,with_exog or 'all'")
    p.add_argument("--regimes", default="all",
                   help="GFC,OIL_CRASH,COVID,ENERGY_22,CALM_17_19 or 'all'")
    p.add_argument("--grid-search-csv", default="garch_grid_search_results.csv",
                   help="Path to grid search results CSV")
    p.add_argument("--no-plots", action="store_true", help="Skip plots.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
