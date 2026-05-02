"""
LinkedIn SilverKite (via the ``greykite`` package) volatility forecasting on
the 12 split cells in ``data/splits``.

We use the ``SimpleSilverkiteEstimator`` directly because we need fine-grained
control over the expanding cross-validation refit cadence; the high-level
``Forecaster`` orchestrator assumes a single train/forecast split. The model
predicts one-step-ahead realized variance ``(100 * ret)^2`` from the lagged
feature columns built into each split CSV.

Outputs mirror ``lstm-model/outputs/`` so results are directly comparable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from common import (
    EXOGS,
    FREQS,
    REFIT_CADENCE,
    TARGETS,
    add_grid_arguments,
    add_residual_columns,
    build_forecast_row,
    cells_iterator,
    feature_columns,
    history_until,
    load_cell,
    metrics,
    parse_float_list,
    parse_selection,
    parse_str_list,
    plot_cell_diagnostics,
    realized_variance,
    returns_pct,
)

warnings.filterwarnings("ignore")

try:
    from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "greykite is not installed. Install it via `pip install greykite`."
    ) from e


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "silverkite"
MODEL_LABEL = "SilverKite"


@dataclass(frozen=True)
class SilverKiteConfig:
    fit_algorithm: str = "ridge"  # ridge / linear / lasso / sgd
    yearly_seasonality: int = 8
    weekly_seasonality: int = 0
    growth_term: str = "linear"
    changepoints_method: str = "auto"
    use_regressors: bool = True
    feature_sets_enabled: bool = False


LOG_VAR_FLOOR = 1e-8  # log-variance space; keeps predictions non-negative


def to_silverkite_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build SilverKite input frame in log-variance space (non-negative when exponentiated)."""
    out = pd.DataFrame({"ts": pd.to_datetime(df["date"]).values})
    rv = realized_variance(df["ret"])
    out["y"] = np.log(np.maximum(rv, LOG_VAR_FLOOR))
    for col in columns:
        out[col] = df[col].to_numpy(dtype=np.float64)
    return out


def build_silverkite(config: SilverKiteConfig, columns: list[str], freq: str) -> SimpleSilverkiteEstimator:
    if freq == "daily":
        weekly = config.weekly_seasonality
        yearly = config.yearly_seasonality
    else:
        weekly = 0
        yearly = config.yearly_seasonality

    extra_pred_cols = list(columns) if config.use_regressors else []

    # Daily series are business days (B); weekly series are Friday-anchored.
    # Mismatched frequency strings cause greykite to drop *every* training row.
    sk_freq = "B" if freq == "daily" else "W-FRI"
    return SimpleSilverkiteEstimator(
        freq=sk_freq,
        forecast_horizon=1,
        fit_algorithm_dict={"fit_algorithm": config.fit_algorithm, "fit_algorithm_params": None},
        auto_holiday=False,
        holidays_to_model_separately=[],
        holiday_lookup_countries=[],
        holiday_pre_num_days=0,
        holiday_post_num_days=0,
        changepoints_dict={"method": config.changepoints_method} if config.changepoints_method else None,
        yearly_seasonality=yearly,
        quarterly_seasonality=0,
        monthly_seasonality=0,
        weekly_seasonality=weekly,
        daily_seasonality=0,
        max_daily_seas_interaction_order=0,
        max_weekly_seas_interaction_order=0,
        growth_term=config.growth_term,
        regressor_cols=extra_pred_cols,
        feature_sets_enabled=config.feature_sets_enabled,
        extra_pred_cols=[],
    )


def fit_silverkite(history_df: pd.DataFrame, columns: list[str], config: SilverKiteConfig, freq: str) -> SimpleSilverkiteEstimator:
    estimator = build_silverkite(config, columns, freq)
    train_df = to_silverkite_df(history_df, columns)
    estimator.fit(X=train_df, time_col="ts", value_col="y")
    return estimator


def predict_one(model: SimpleSilverkiteEstimator, future_row: pd.DataFrame, columns: list[str]) -> float:
    future = pd.DataFrame({"ts": pd.to_datetime(future_row["date"]).values})
    for col in columns:
        future[col] = future_row[col].to_numpy(dtype=np.float64)
    pred_df = model.predict(X=future)
    yhat_col = "forecast" if "forecast" in pred_df.columns else pred_df.columns[-1]
    yhat_log = float(pred_df[yhat_col].iloc[0])
    return float(max(np.exp(yhat_log), LOG_VAR_FLOOR))


def evaluate_validation(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: SilverKiteConfig,
    freq: str,
) -> dict[str, float]:
    """Fit on train, score one-step ahead across the validation block.

    Like Prophet, we keep a single fit over the whole val block to keep tuning
    runtime tractable; the test block performs the proper expanding ECV.
    """
    val_df = frames["val"].reset_index(drop=True)
    model = fit_silverkite(frames["train"], columns, config, freq)

    preds = np.empty(len(val_df), dtype=np.float64)
    for i in range(len(val_df)):
        preds[i] = predict_one(model, val_df.iloc[[i]], columns)

    y_true = realized_variance(val_df["ret"])
    rets = returns_pct(val_df["ret"])
    return metrics(y_true, preds, rets)


def tune_cell(
    freq: str,
    exog: str,
    target: str,
    grid: list[SilverKiteConfig],
) -> tuple[SilverKiteConfig, dict[str, float]]:
    frames = load_cell(freq, exog, target)
    columns = feature_columns(frames["train"])
    best_config: SilverKiteConfig | None = None
    best_row: dict[str, float] | None = None

    for config in tqdm(grid, desc=f"tune {target}/{freq}/{exog}", leave=False):
        try:
            row = evaluate_validation(frames, columns, config, freq)
        except Exception as exc:  # silverkite occasionally fails for a config; skip it
            print(f"  -> skipping {config} due to {type(exc).__name__}: {exc}")
            continue
        row.update(asdict(config))
        if best_row is None or row["qlike"] < best_row["qlike"]:
            best_row = row
            best_config = config

    if best_config is None or best_row is None:
        raise RuntimeError(f"No valid SilverKite configuration succeeded for {target}/{freq}/{exog}")
    return best_config, best_row


def expanding_test_forecast(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: SilverKiteConfig,
    freq: str,
) -> pd.DataFrame:
    cadence = REFIT_CADENCE[freq]
    test = frames["test"].reset_index(drop=True)
    rows: list[dict] = []
    model: SimpleSilverkiteEstimator | None = None

    for step in tqdm(range(len(test)), desc=f"test ECV {freq}", leave=False):
        if model is None or step % cadence == 0:
            history = history_until(frames, step)
            model = fit_silverkite(history, columns, config, freq)

        test_row = test.iloc[[step]]
        pred_var = predict_one(model, test_row, columns)
        y_true = float(realized_variance(test_row["ret"])[0])
        ret_pct = float(returns_pct(test_row["ret"])[0])
        rows.append(build_forecast_row(test_row["date"].iloc[0], ret_pct, y_true, pred_var))

    return pd.DataFrame(rows)


def default_grid(args: argparse.Namespace) -> list[SilverKiteConfig]:
    fit_algorithms = parse_str_list(args.fit_algorithms)
    yearly_seas = [int(x) for x in parse_str_list(args.yearly_seasonality)]
    cps_methods = parse_str_list(args.changepoint_methods)
    grid: list[SilverKiteConfig] = []
    for algo, ys, cm in itertools.product(fit_algorithms, yearly_seas, cps_methods):
        grid.append(
            SilverKiteConfig(
                fit_algorithm=algo,
                yearly_seasonality=ys,
                weekly_seasonality=5 if args.weekly_seasonality else 0,
                growth_term="linear",
                changepoints_method=cm if cm.lower() != "none" else "",
                use_regressors=not args.disable_regressors,
                feature_sets_enabled=False,
            )
        )
    return grid


def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "forecasts").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    targets = parse_selection(args.targets, TARGETS)
    freqs = parse_selection(args.freqs, FREQS)
    exogs = parse_selection(args.exogs, EXOGS)
    grid = default_grid(args)

    selection_rows: list[dict] = []
    test_rows: list[dict] = []

    cells = list(cells_iterator(targets, freqs, exogs))
    for target, freq, exog in tqdm(cells, desc="SilverKite cells"):
        print(f"Tuning SilverKite on {target}/{freq}/{exog}...")
        best_config, val_row = tune_cell(freq, exog, target, grid)
        val_row.update({"target": target, "freq": freq, "exog": exog})
        selection_rows.append(val_row)

        print(f"Refit + expanding test on {target}/{freq}/{exog}...")
        frames = load_cell(freq, exog, target)
        columns = feature_columns(frames["train"])
        forecast_df = expanding_test_forecast(frames, columns, best_config, freq)
        forecast_df = add_residual_columns(forecast_df)
        forecast_path = OUT_DIR / "forecasts" / f"{target}_{freq}_{exog}_test_forecasts.csv"
        forecast_df.to_csv(forecast_path, index=False)

        if not args.no_plots:
            plot_cell_diagnostics(frames, forecast_df, OUT_DIR, target, freq, exog, MODEL_LABEL)

        test_metric = metrics(
            forecast_df["realized_var"].to_numpy(),
            forecast_df["pred_var"].to_numpy(),
            forecast_df["ret_pct"].to_numpy(),
        )
        test_metric.update(
            {
                "target": target,
                "freq": freq,
                "exog": exog,
                "forecast_file": str(forecast_path.relative_to(Path(__file__).resolve().parents[1])),
                **asdict(best_config),
            }
        )
        test_rows.append(test_metric)

    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "silverkite_validation_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT_DIR / "silverkite_test_results.csv", index=False)
    with (OUT_DIR / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"Done. Results saved under {OUT_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LinkedIn SilverKite volatility forecasting.")
    add_grid_arguments(parser)
    parser.add_argument(
        "--fit-algorithms", default="ridge,linear",
        help="Comma list of fit_algorithm values (e.g. ridge,linear,lasso).",
    )
    parser.add_argument(
        "--yearly-seasonality", default="0,8",
        help="Comma list of yearly seasonality Fourier orders.",
    )
    parser.add_argument(
        "--changepoint-methods", default="auto,none",
        help="Comma list of changepoints methods (auto, none).",
    )
    parser.add_argument(
        "--weekly-seasonality", action="store_true",
        help="Enable weekly seasonality at order 5 (daily data only).",
    )
    parser.add_argument(
        "--disable-regressors", action="store_true",
        help="Fit SilverKite without lagged feature regressors (univariate baseline).",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
