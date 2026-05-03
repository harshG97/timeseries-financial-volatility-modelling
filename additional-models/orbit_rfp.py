"""
Orbit RFP (Random Forecast Periods) evaluation.

Reads the selected Orbit hyperparameters from ``orbit_validation_results.csv``
(one blueprint per cell), then evaluates each blueprint on every RFP window
using ``src/rfp_generator.py``.

Outputs
-------
additional-models/outputs/orbit/rfp/
    orbit_rfp_results.csv
    orbit_rfp_summary.csv
    forecasts/
    plots/
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
import warnings

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

warnings.filterwarnings("ignore")

# Re-use building blocks from the main Orbit script
from importlib.util import spec_from_file_location, module_from_spec

_path = Path(__file__).resolve().parent / "orbit_volatility.py"
_spec = spec_from_file_location("orbit_volatility", _path)
_mod = module_from_spec(_spec)
sys.modules["orbit_volatility"] = _mod  # must register before exec for @dataclass
_spec.loader.exec_module(_mod)

OrbitConfig = _mod.OrbitConfig
fit_orbit = _mod.fit_orbit
metrics = _mod.metrics
feature_columns = _mod.feature_columns
realized_variance = _mod.realized_variance
returns_pct = _mod.returns_pct
add_residual_columns = _mod.add_residual_columns
LOG_VAR_FLOOR = _mod.LOG_VAR_FLOOR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "orbit"
RFP_OUT = OUT_DIR / "rfp"
TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]


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


def load_validation_configs(csv_path: Path) -> dict[tuple[str, str, str], OrbitConfig]:
    df = pd.read_csv(csv_path)
    configs: dict[tuple[str, str, str], OrbitConfig] = {}
    for _, row in df.iterrows():
        key = (row["target"], row["freq"], row["exog"])
        configs[key] = OrbitConfig(
            estimator=row["estimator"],
            seasonality=int(row["seasonality"]) if pd.notna(row["seasonality"]) else None,
            global_trend_option=row["global_trend_option"],
            damped_factor=float(row["damped_factor"]),
            level_sm_input=float(row["level_sm_input"]) if pd.notna(row.get("level_sm_input")) else None,
            use_regressors=bool(row["use_regressors"]),
        )
    return configs


def evaluate_window(
    window,
    config: OrbitConfig,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Train Orbit on a single RFP window and return (metrics_dict, forecast_df)."""
    train_df = window.train
    forecast_df = window.forecast
    columns = feature_columns(train_df)

    model = fit_orbit(train_df, columns, config, seed)

    # Batch predict
    future = pd.DataFrame({"date": pd.to_datetime(forecast_df["date"]).values})
    for col in columns:
        future[col] = forecast_df[col].to_numpy(dtype=np.float64)
    
    pred_df = model.predict(df=future)
    candidate_cols = [c for c in ["prediction", "prediction_50", "prediction_5"] if c in pred_df.columns]
    if not candidate_cols:
        candidate_cols = [c for c in pred_df.columns if c != "date"]
    
    yhat_log = pred_df[candidate_cols[0]].values
    pred_var = np.maximum(np.exp(yhat_log), LOG_VAR_FLOOR)

    y_true = realized_variance(forecast_df["ret"])
    ret_pct = returns_pct(forecast_df["ret"])

    m = metrics(y_true, pred_var, ret_pct)
    m.update({
        "target": window.target,
        "freq": window.freq,
        "exog": "with_exog" if window.use_exog else "no_exog",
        "window_id": window.window_id,
        "regime": window.regime,
        "n_train": window.n_train,
        "n_forecast": len(y_true),
        **asdict(config),
    })

    from statistics import NormalDist
    sigma = np.sqrt(pred_var)
    fc_frame = pd.DataFrame({
        "date": pd.to_datetime(forecast_df["date"]).strftime("%Y-%m-%d"),
        "ret_pct": ret_pct,
        "realized_var": y_true,
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
    val_csv = Path(args.validation_csv)
    if not val_csv.is_absolute():
        val_csv = OUT_DIR / val_csv
    if not val_csv.exists():
        raise FileNotFoundError(
            f"Validation results CSV not found: {val_csv}\n"
            "Run orbit_volatility.py first to generate it."
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

    (RFP_OUT / "forecasts").mkdir(parents=True, exist_ok=True)

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
        print(f"{'='*60}")

        windows = list(gen.iter_windows(
            freq=freq, target=target, use_exog=use_exog,
            regimes=regime_filter,
        ))

        for w in tqdm(windows, desc=f"  windows {target}/{freq}/{exog}", leave=False):
            m, fc_frame = evaluate_window(w, config, seed=args.seed)
            all_results.append(m)

            fc_path = RFP_OUT / "forecasts" / f"{target}_{freq}_{exog}_{w.window_id}.csv"
            fc_frame.to_csv(fc_path, index=False)

            if not args.no_plots:
                plot_per_window(fc_frame, target, freq, exog,
                                w.window_id, w.regime)

            print(f"    {w.window_id:20s}  QLIKE={m['qlike']:.4f}  "
                  f"RMSE={m['rmse']:.4f}  n_fc={m['n_forecast']}")

    if not all_results:
        print("No windows evaluated — check your selection filters.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RFP_OUT / "orbit_rfp_results.csv", index=False)

    group_cols = ["target", "freq", "exog", "regime"]
    metric_cols = ["mse", "rmse", "mae", "qlike",
                   "var_1_hit_rate", "var_5_hit_rate"]
    available_metrics = [c for c in metric_cols if c in results_df.columns]

    summary = results_df.groupby(group_cols)[available_metrics].agg(
        ["mean", "median"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RFP_OUT / "orbit_rfp_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {RFP_OUT.relative_to(ROOT)}/")
    print(f"  orbit_rfp_results.csv   ({len(results_df)} rows)")
    print(f"  orbit_rfp_summary.csv   ({len(summary)} rows)")
    print(f"  forecasts/             ({len(results_df)} files)")

    if not args.no_plots:
        print("Generating aggregate plots...")
        for target in results_df["target"].unique():
            for freq in results_df["freq"].unique():
                for exog in results_df["exog"].unique():
                    plot_regime_bars(results_df, target, freq, exog)
                plot_ablation(results_df, target, freq)
        plot_heatmap(results_df)
        plot_boxplot_regime(results_df)

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate Orbit volatility models on RFP windows."
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
        default="orbit_validation_results.csv",
        help="Path to validation results CSV (default: outputs/orbit/orbit_validation_results.csv)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
