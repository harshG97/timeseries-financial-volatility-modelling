"""
LSTM RFP (Random Forecast Periods) evaluation.

Reads the selected LSTM hyperparameters from ``lstm_validation_results.csv``
(one blueprint per cell), then evaluates each blueprint on every RFP window
using ``src/rfp_generator.py``.

For each window the model is trained from scratch on all data up to ``fit_end``
(no validation split, no early stopping), then predicts every day/week in the
forecast window without refitting. This isolates pure out-of-sample
generalization across market regimes.

Outputs
-------
lstm-model/outputs/rfp/
    lstm_rfp_results.csv        -- one row per (cell × window)
    lstm_rfp_summary.csv        -- per-cell and per-regime aggregates
    forecasts/{target}_{freq}_{exog}_{window_id}.csv
    plots/per_window/{target}/{freq}/{exog}/{window_id}.png
    plots/regime_bars/{target}_{freq}_{exog}.png
    plots/ablation/{target}_{freq}_exog_ablation.png
    plots/heatmap_qlike.png
    plots/boxplot_regime_qlike.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Resolve paths so imports work regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rfp_generator import RFPGenerator, VALID_FREQUENCIES, VALID_TARGETS

# Re-use building blocks from the main LSTM script
from importlib.util import spec_from_file_location, module_from_spec

_lstm_path = Path(__file__).resolve().parent / "lstm_volatility.py"
_spec = spec_from_file_location("lstm_volatility", _lstm_path)
_lstm_mod = module_from_spec(_spec)
sys.modules["lstm_volatility"] = _lstm_mod  # must register before exec for @dataclass
_spec.loader.exec_module(_lstm_mod)

LSTMConfig = _lstm_mod.LSTMConfig
VolatilityLSTM = _lstm_mod.VolatilityLSTM
train_model = _lstm_mod.train_model
predict = _lstm_mod.predict
metrics = _lstm_mod.metrics
set_seed = _lstm_mod.set_seed
get_device = _lstm_mod.get_device
feature_columns = _lstm_mod.feature_columns
realized_variance = _lstm_mod.realized_variance
make_sequences = _lstm_mod.make_sequences
fit_scaler = _lstm_mod.fit_scaler
add_residual_columns = _lstm_mod.add_residual_columns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
RFP_OUT = OUT_DIR / "rfp"
TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_selection(raw: str, allowed: list[str]) -> list[str]:
    """Parse a comma-separated selection or 'all'."""
    if raw.lower() == "all":
        return allowed
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    bad = sorted(set(selected) - set(allowed))
    if bad:
        raise ValueError(f"Invalid values {bad}; allowed values are {allowed}")
    return selected


def load_validation_configs(csv_path: Path) -> dict[tuple[str, str, str], LSTMConfig]:
    """Load per-cell best configs from the validation results CSV.

    Returns a dict keyed by (target, freq, exog) -> LSTMConfig.
    """
    df = pd.read_csv(csv_path)
    configs: dict[tuple[str, str, str], LSTMConfig] = {}
    for _, row in df.iterrows():
        key = (row["target"], row["freq"], row["exog"])
        configs[key] = LSTMConfig(
            lookback=int(row["lookback"]),
            hidden_size=int(row["hidden_size"]),
            num_layers=int(row["num_layers"]),
            dropout=float(row["dropout"]),
            learning_rate=float(row["learning_rate"]),
            weight_decay=float(row["weight_decay"]),
            batch_size=int(row["batch_size"]),
            epochs=int(row["epochs"]),
            patience=int(row["patience"]),
        )
    return configs


def evaluate_window(
    window,
    config: LSTMConfig,
    device,
    seed: int,
    show_epoch_progress: bool,
) -> tuple[dict, pd.DataFrame]:
    """Train LSTM on a single RFP window and return (metrics_dict, forecast_df).

    Training uses all data up to fit_end with no validation split.
    Prediction covers the full forecast period without refitting.
    """
    # ---- assemble training data ----
    train_df = window.train
    forecast_df = window.forecast
    columns = feature_columns(train_df)

    scaler = fit_scaler(train_df, columns)

    # Build training sequences
    train_x, train_y, _, _ = make_sequences(
        train_df, columns, config.lookback, scaler
    )

    # ---- train (no validation split) ----
    model, _ = train_model(
        train_x, train_y,
        None, None,           # no val
        config, device, seed,
        progress_label=f"RFP {window.window_id}",
        show_progress=show_epoch_progress,
    )

    # ---- predict on forecast period ----
    # Need lookback-1 rows of context before the first forecast row to form
    # complete sequences.  Concatenate the tail of training data with forecast.
    context_rows = train_df.tail(config.lookback - 1)
    combined = pd.concat([context_rows, forecast_df], ignore_index=True)
    fc_x, fc_y, fc_dates, fc_ret = make_sequences(
        combined, columns, config.lookback, scaler,
        start_output_idx=len(context_rows),  # only produce forecast outputs
    )

    pred_var = predict(model, fc_x, device)

    # ---- score ----
    m = metrics(fc_y, pred_var, fc_ret)
    m.update({
        "target": window.target,
        "freq": window.freq,
        "exog": "with_exog" if window.use_exog else "no_exog",
        "window_id": window.window_id,
        "regime": window.regime,
        "n_train": window.n_train,
        "n_forecast": len(fc_y),
        **asdict(config),
    })

    # ---- per-day forecast frame ----
    from statistics import NormalDist
    sigma = np.sqrt(pred_var)
    fc_frame = pd.DataFrame({
        "date": pd.to_datetime(fc_dates).strftime("%Y-%m-%d"),
        "ret_pct": fc_ret,
        "realized_var": fc_y,
        "pred_var": pred_var,
        "pred_vol": sigma,
        "VaR_1": NormalDist().inv_cdf(0.01) * sigma,
        "VaR_5": NormalDist().inv_cdf(0.05) * sigma,
    })
    fc_frame = add_residual_columns(fc_frame)

    return m, fc_frame


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_per_window(fc_frame: pd.DataFrame, target: str, freq: str,
                    exog: str, window_id: str, regime: str) -> None:
    """Plot predicted vs actual volatility for one window."""
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
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{window_id}.png", dpi=150)
    plt.close(fig)


def plot_regime_bars(results_df: pd.DataFrame, target: str, freq: str,
                     exog: str) -> None:
    """Bar chart of mean QLIKE by regime for one cell."""
    plot_dir = RFP_OUT / "plots" / "regime_bars"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cell = results_df[
        (results_df["target"] == target)
        & (results_df["freq"] == freq)
        & (results_df["exog"] == exog)
    ]
    if cell.empty:
        return

    agg = cell.groupby("regime")["qlike"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(agg)))
    ax.bar(agg.index, agg.values, color=colors, edgecolor="0.3", linewidth=0.6)
    ax.set_title(f"{target} {freq} {exog} — Mean QLIKE by Regime",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("QLIKE")
    ax.set_xlabel("Regime")
    for i, (regime, val) in enumerate(agg.items()):
        ax.text(i, val + 0.02 * agg.max(), f"{val:.3f}", ha="center",
                fontsize=8, color="0.2")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{target}_{freq}_{exog}.png", dpi=150)
    plt.close(fig)


def plot_ablation(results_df: pd.DataFrame, target: str, freq: str) -> None:
    """Grouped bar: no_exog vs with_exog by regime for one target×freq."""
    plot_dir = RFP_OUT / "plots" / "ablation"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cell = results_df[
        (results_df["target"] == target) & (results_df["freq"] == freq)
    ]
    if cell.empty or cell["exog"].nunique() < 2:
        return

    pivot = cell.pivot_table(
        index="regime", columns="exog", values="qlike", aggfunc="mean"
    ).sort_index()

    regimes = pivot.index.tolist()
    x = np.arange(len(regimes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if "no_exog" in pivot.columns:
        ax.bar(x - width / 2, pivot["no_exog"], width, label="no_exog",
               color="#4c78a8", edgecolor="0.3", linewidth=0.6)
    if "with_exog" in pivot.columns:
        ax.bar(x + width / 2, pivot["with_exog"], width, label="with_exog",
               color="#f58518", edgecolor="0.3", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=30, ha="right")
    ax.set_ylabel("QLIKE (mean)")
    ax.set_title(f"{target} {freq} — Exog Ablation by Regime",
                 fontsize=11, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{target}_{freq}_exog_ablation.png", dpi=150)
    plt.close(fig)


def plot_heatmap(results_df: pd.DataFrame) -> None:
    """Heatmap of mean QLIKE: rows = cell, columns = regime."""
    plot_dir = RFP_OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results_df = results_df.copy()
    results_df["cell"] = (
        results_df["target"] + " / " + results_df["freq"] + " / " + results_df["exog"]
    )
    pivot = results_df.pivot_table(
        index="cell", columns="regime", values="qlike", aggfunc="mean"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4),
                                    max(5, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > pivot.values[np.isfinite(pivot.values)].mean() else "0.15")

    ax.set_title("RFP Mean QLIKE — Cell × Regime", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="QLIKE")
    fig.tight_layout()
    fig.savefig(plot_dir / "heatmap_qlike.png", dpi=150)
    plt.close(fig)


def plot_boxplot_regime(results_df: pd.DataFrame) -> None:
    """Box plot of per-window QLIKE by regime across all cells."""
    plot_dir = RFP_OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    regimes = sorted(results_df["regime"].unique())
    data = [results_df[results_df["regime"] == r]["qlike"].dropna().values
            for r in regimes]

    fig, ax = plt.subplots(figsize=(max(7, len(regimes) * 1.2), 4.5))
    bp = ax.boxplot(data, tick_labels=regimes, patch_artist=True, notch=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(regimes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("0.3")
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
    set_seed(args.seed)
    device = get_device(args.cpu)
    print(f"Using device: {device}")

    # ---- resolve paths ----
    val_csv = Path(args.validation_csv)
    if not val_csv.is_absolute():
        val_csv = OUT_DIR / val_csv
    if not val_csv.exists():
        raise FileNotFoundError(
            f"Validation results CSV not found: {val_csv}\n"
            "Run lstm_volatility.py first to generate it."
        )

    configs = load_validation_configs(val_csv)
    gen = RFPGenerator()

    targets = parse_selection(args.targets, TARGETS)
    freqs = parse_selection(args.freqs, FREQS)
    exogs = parse_selection(args.exogs, EXOGS)
    regime_filter = (
        [r.strip() for r in args.regimes.split(",") if r.strip()]
        if args.regimes.lower() != "all"
        else None
    )

    # ---- prepare output dirs ----
    (RFP_OUT / "forecasts").mkdir(parents=True, exist_ok=True)

    # ---- iterate cells × windows ----
    all_results: list[dict] = []
    cells = [
        (t, f, e)
        for t in targets for f in freqs for e in exogs
    ]

    for target, freq, exog in tqdm(cells, desc="RFP cells"):
        key = (target, freq, exog)
        if key not in configs:
            print(f"  ⚠ No validation config for {key}, skipping.")
            continue

        config = configs[key]
        use_exog = exog == "with_exog"
        print(f"\n{'='*60}")
        print(f"Cell: {target}/{freq}/{exog}")
        print(f"Config: lookback={config.lookback}, hidden={config.hidden_size}, "
              f"layers={config.num_layers}, dropout={config.dropout}, "
              f"lr={config.learning_rate}")
        print(f"{'='*60}")

        windows = list(gen.iter_windows(
            freq=freq, target=target, use_exog=use_exog,
            regimes=regime_filter,
        ))

        for w in tqdm(windows, desc=f"  windows {target}/{freq}/{exog}", leave=False):
            m, fc_frame = evaluate_window(
                w, config, device, args.seed, args.show_epoch_progress,
            )
            all_results.append(m)

            # Save per-window forecast
            fc_path = RFP_OUT / "forecasts" / f"{target}_{freq}_{exog}_{w.window_id}.csv"
            fc_frame.to_csv(fc_path, index=False)

            # Per-window plot
            if not args.no_plots:
                plot_per_window(fc_frame, target, freq, exog,
                                w.window_id, w.regime)

            print(f"    {w.window_id:20s}  QLIKE={m['qlike']:.4f}  "
                  f"RMSE={m['rmse']:.4f}  n_fc={m['n_forecast']}")

    if not all_results:
        print("No windows evaluated — check your selection filters.")
        return

    # ---- save results ----
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RFP_OUT / "lstm_rfp_results.csv", index=False)

    # ---- build summary ----
    group_cols = ["target", "freq", "exog", "regime"]
    metric_cols = ["mse", "rmse", "mae", "qlike",
                   "var_1_hit_rate", "var_5_hit_rate"]
    available_metrics = [c for c in metric_cols if c in results_df.columns]

    summary = results_df.groupby(group_cols)[available_metrics].agg(
        ["mean", "median"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RFP_OUT / "lstm_rfp_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {RFP_OUT.relative_to(ROOT)}/")
    print(f"  lstm_rfp_results.csv   ({len(results_df)} rows)")
    print(f"  lstm_rfp_summary.csv   ({len(summary)} rows)")
    print(f"  forecasts/             ({len(results_df)} files)")

    # ---- aggregate plots ----
    if not args.no_plots:
        print("Generating aggregate plots...")

        # Per-cell regime bar charts
        for target in results_df["target"].unique():
            for freq in results_df["freq"].unique():
                for exog in results_df["exog"].unique():
                    plot_regime_bars(results_df, target, freq, exog)

        # Exog ablation charts (only if both exog conditions present)
        for target in results_df["target"].unique():
            for freq in results_df["freq"].unique():
                plot_ablation(results_df, target, freq)

        # Heatmap and boxplot
        plot_heatmap(results_df)
        plot_boxplot_regime(results_df)

        print(f"  Plots saved to {(RFP_OUT / 'plots').relative_to(ROOT)}/")

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate LSTM volatility models on RFP windows."
    )
    p.add_argument(
        "--targets", default="all",
        help="Comma list or 'all': SPY,OIL,GOLD (default: all)",
    )
    p.add_argument(
        "--freqs", default="all",
        help="Comma list or 'all': daily,weekly (default: all)",
    )
    p.add_argument(
        "--exogs", default="all",
        help="Comma list or 'all': no_exog,with_exog (default: all)",
    )
    p.add_argument(
        "--regimes", default="all",
        help="Comma list or 'all': GFC,OIL_CRASH,COVID,ENERGY_22,CALM_17_19 "
             "(default: all)",
    )
    p.add_argument(
        "--validation-csv",
        default="lstm_validation_results.csv",
        help="Path to validation results CSV (default: outputs/lstm_validation_results.csv)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation.")
    p.add_argument("--show-epoch-progress", action="store_true",
                   help="Show nested epoch progress bars.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
