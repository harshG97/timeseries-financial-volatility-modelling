"""
Uber Orbit volatility forecasting on the 12 split cells in ``data/splits``.

The model predicts one-step-ahead realized variance ``(100 * ret)^2`` from the
lagged feature columns built into each split CSV.  Hyperparameters are tuned on
validation data; the selected blueprint is then refit on train + validation
and evaluated on test with expanding cross-validation (refit cadence:
20 daily / 4 weekly).

We use the DLT (Damped Local Trend) Bayesian state-space model. DLT can ingest
lagged regressors directly, supports both MAP and MCMC posterior inference,
and is the closest Orbit analogue to the Prophet / SilverKite univariate
forecasting style.

Outputs mirror ``lstm-model/outputs/`` so results are directly comparable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

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
    from orbit.models import DLT  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "orbit-ml is not installed. Install it via `pip install orbit-ml`."
    ) from e


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "orbit"
MODEL_LABEL = "Orbit-DLT"


@dataclass(frozen=True)
class OrbitConfig:
    estimator: str = "stan-map"
    seasonality: int | None = None  # None disables seasonality
    global_trend_option: str = "linear"
    damped_factor: float = 0.8
    level_sm_input: float | None = None
    use_regressors: bool = True


LOG_VAR_FLOOR = 1e-8  # log-variance space; keeps predictions non-negative


def to_orbit_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build Orbit input frame in log-variance space (non-negative when exponentiated)."""
    out = pd.DataFrame({"date": pd.to_datetime(df["date"]).values})
    rv = realized_variance(df["ret"])
    out["y"] = np.log(np.maximum(rv, LOG_VAR_FLOOR))
    for col in columns:
        out[col] = df[col].to_numpy(dtype=np.float64)
    return out


def build_dlt(config: OrbitConfig, columns: list[str], seed: int) -> DLT:
    kwargs: dict = dict(
        response_col="y",
        date_col="date",
        seasonality=config.seasonality,
        estimator=config.estimator,
        global_trend_option=config.global_trend_option,
        damped_factor=config.damped_factor,
        seed=seed,
    )
    if config.level_sm_input is not None:
        kwargs["level_sm_input"] = config.level_sm_input
    if config.use_regressors and columns:
        kwargs["regressor_col"] = list(columns)
    return DLT(**kwargs)


def fit_orbit(history_df: pd.DataFrame, columns: list[str], config: OrbitConfig, seed: int) -> DLT:
    model = build_dlt(config, columns, seed)
    model.fit(df=to_orbit_df(history_df, columns))
    return model


def predict_one(model: DLT, future_row: pd.DataFrame, columns: list[str]) -> float:
    future = pd.DataFrame({"date": pd.to_datetime(future_row["date"]).values})
    for col in columns:
        future[col] = future_row[col].to_numpy(dtype=np.float64)
    pred_df = model.predict(df=future)
    # Orbit returns "prediction" for MAP and percentiles for MCMC; prefer the
    # 50th percentile / point estimate.
    candidate_cols = [c for c in ["prediction", "prediction_50", "prediction_5"] if c in pred_df.columns]
    if not candidate_cols:
        candidate_cols = [c for c in pred_df.columns if c != "date"]
    yhat_log = float(pred_df[candidate_cols[0]].iloc[0])
    return float(max(np.exp(yhat_log), LOG_VAR_FLOOR))


def evaluate_validation(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: OrbitConfig,
    seed: int,
) -> dict[str, float]:
    """Single-fit evaluation of the validation block (matches Prophet convention)."""
    val_df = frames["val"].reset_index(drop=True)
    model = fit_orbit(frames["train"], columns, config, seed)

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
    grid: list[OrbitConfig],
    seed: int,
) -> tuple[OrbitConfig, dict[str, float]]:
    frames = load_cell(freq, exog, target)
    columns = feature_columns(frames["train"])
    best_config: OrbitConfig | None = None
    best_row: dict[str, float] | None = None

    for idx, config in enumerate(tqdm(grid, desc=f"tune {target}/{freq}/{exog}", leave=False)):
        try:
            row = evaluate_validation(frames, columns, config, seed + idx)
        except Exception as exc:
            print(f"  -> skipping {config} due to {type(exc).__name__}: {exc}")
            continue
        row.update(asdict(config))
        if best_row is None or row["qlike"] < best_row["qlike"]:
            best_row = row
            best_config = config

    if best_config is None or best_row is None:
        raise RuntimeError(f"No valid Orbit configuration succeeded for {target}/{freq}/{exog}")
    return best_config, best_row


def expanding_test_forecast(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: OrbitConfig,
    freq: str,
    seed: int,
) -> pd.DataFrame:
    cadence = REFIT_CADENCE[freq]
    test = frames["test"].reset_index(drop=True)
    rows: list[dict] = []
    model: DLT | None = None

    for step in tqdm(range(len(test)), desc=f"test ECV {freq}", leave=False):
        if model is None or step % cadence == 0:
            history = history_until(frames, step)
            model = fit_orbit(history, columns, config, seed + step)

        test_row = test.iloc[[step]]
        pred_var = predict_one(model, test_row, columns)
        y_true = float(realized_variance(test_row["ret"])[0])
        ret_pct = float(returns_pct(test_row["ret"])[0])
        rows.append(build_forecast_row(test_row["date"].iloc[0], ret_pct, y_true, pred_var))

    return pd.DataFrame(rows)


def default_grid(args: argparse.Namespace) -> list[OrbitConfig]:
    estimators = parse_str_list(args.estimators)
    damped = parse_float_list(args.damped_factors)
    trends = parse_str_list(args.trends)
    grid: list[OrbitConfig] = []
    for est, df_, tr in itertools.product(estimators, damped, trends):
        grid.append(
            OrbitConfig(
                estimator=est,
                seasonality=None,
                global_trend_option=tr,
                damped_factor=df_,
                level_sm_input=None,
                use_regressors=not args.disable_regressors,
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
    for target, freq, exog in tqdm(cells, desc="Orbit cells"):
        print(f"Tuning Orbit on {target}/{freq}/{exog}...")
        best_config, val_row = tune_cell(freq, exog, target, grid, args.seed)
        val_row.update({"target": target, "freq": freq, "exog": exog})
        selection_rows.append(val_row)

        print(f"Refit + expanding test on {target}/{freq}/{exog}...")
        frames = load_cell(freq, exog, target)
        columns = feature_columns(frames["train"])
        forecast_df = expanding_test_forecast(frames, columns, best_config, freq, args.seed + 10_000)
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

    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "orbit_validation_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT_DIR / "orbit_test_results.csv", index=False)
    with (OUT_DIR / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"Done. Results saved under {OUT_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uber Orbit (DLT) volatility forecasting.")
    add_grid_arguments(parser)
    parser.add_argument(
        "--estimators", default="stan-map",
        help="Comma list of Orbit estimators (stan-map, stan-mcmc).",
    )
    parser.add_argument(
        "--damped-factors", default="0.7,0.9",
        help="Comma list of damped_factor values for DLT.",
    )
    parser.add_argument(
        "--trends", default="linear,loglinear",
        help="Comma list of global_trend_option values (linear, loglinear, flat, logistic).",
    )
    parser.add_argument(
        "--disable-regressors", action="store_true",
        help="Fit Orbit DLT without lagged regressors (univariate baseline).",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
