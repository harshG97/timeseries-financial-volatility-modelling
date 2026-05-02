"""
Meta Prophet volatility forecasting on the 12 split cells in ``data/splits``.

The model predicts one-step-ahead realized variance ``(100 * ret)^2`` from the
lagged feature columns already built into each split CSV. Hyperparameters are
selected on validation data, then the selected blueprint is refit on
train + validation and evaluated on test with expanding cross-validation
(refit cadence: 20 daily / 4 weekly).

Outputs mirror ``lstm-model/outputs/`` so results are directly comparable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
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

# Prophet emits very chatty cmdstan logs by default; quiet them to keep
# tuning loops readable.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

try:
    from prophet import Prophet  # type: ignore
except ImportError as e:  # pragma: no cover - import-time error
    raise SystemExit(
        "Prophet is not installed. Install it via `pip install prophet`."
    ) from e


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "prophet"
MODEL_LABEL = "Prophet"


@dataclass(frozen=True)
class ProphetConfig:
    seasonality_mode: str = "additive"
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    use_regressors: bool = True
    yearly_seasonality: str = "auto"
    weekly_seasonality: str = "auto"


LOG_VAR_FLOOR = 1e-8  # avoid log(0); realized variance is non-negative


def to_prophet_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Translate a split frame into Prophet's ``ds, y, regressors`` format.

    We model ``log(realized_variance)`` as the response so that extrapolated
    predictions can never go negative when exponentiated back to variance
    space.  This is the standard recipe for using additive forecasting models
    on non-negative quantities.
    """
    out = pd.DataFrame({"ds": pd.to_datetime(df["date"]).values})
    rv = realized_variance(df["ret"])
    out["y"] = np.log(np.maximum(rv, LOG_VAR_FLOOR))
    for col in columns:
        out[col] = df[col].to_numpy(dtype=np.float64)
    return out


def build_prophet(config: ProphetConfig, columns: list[str], freq: str) -> Prophet:
    if freq == "daily":
        weekly = config.weekly_seasonality
        yearly = config.yearly_seasonality
        daily = False
    else:  # weekly
        weekly = False
        yearly = config.yearly_seasonality
        daily = False
    model = Prophet(
        seasonality_mode=config.seasonality_mode,
        changepoint_prior_scale=config.changepoint_prior_scale,
        seasonality_prior_scale=config.seasonality_prior_scale,
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        daily_seasonality=daily,
    )
    if config.use_regressors:
        for col in columns:
            model.add_regressor(col)
    return model


def fit_prophet(history_df: pd.DataFrame, columns: list[str], config: ProphetConfig, freq: str) -> Prophet:
    model = build_prophet(config, columns, freq)
    model.fit(to_prophet_df(history_df, columns))
    return model


def predict_one(model: Prophet, future_row: pd.DataFrame, columns: list[str]) -> float:
    """One-step-ahead variance prediction for the single date in ``future_row``.

    Prophet returns ``yhat`` in log-variance space; we exponentiate back to
    variance and floor at a tiny positive value.
    """
    future = pd.DataFrame({"ds": pd.to_datetime(future_row["date"]).values})
    for col in columns:
        future[col] = future_row[col].to_numpy(dtype=np.float64)
    pred = model.predict(future)
    yhat_log = float(pred["yhat"].iloc[0])
    return float(max(np.exp(yhat_log), LOG_VAR_FLOOR))


def evaluate_validation(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: ProphetConfig,
    freq: str,
) -> dict[str, float]:
    """Fit on train, score one-step ahead across the validation block."""
    train_df = frames["train"]
    val_df = frames["val"].reset_index(drop=True)
    history = train_df.copy()

    preds = np.empty(len(val_df), dtype=np.float64)
    for i in range(len(val_df)):
        model = fit_prophet(history, columns, config, freq) if i == 0 else model  # type: ignore[has-type]
        # Re-use the same fitted model across the validation block — Prophet's
        # additive structure means refitting on every val step would dwarf
        # tuning runtime.  We refit at most once per config here; the test
        # block performs the proper expanding ECV refit cadence.
        future_row = val_df.iloc[[i]]
        preds[i] = predict_one(model, future_row, columns)
        # Append the latest observation to the history so the next forecast
        # has it available.  Refit happens lazily via the i==0 branch above.
        history = pd.concat([history, val_df.iloc[[i]]], ignore_index=True)

    y_true = realized_variance(val_df["ret"])
    rets = returns_pct(val_df["ret"])
    return metrics(y_true, preds, rets)


def tune_cell(
    freq: str,
    exog: str,
    target: str,
    grid: list[ProphetConfig],
) -> tuple[ProphetConfig, dict[str, float]]:
    frames = load_cell(freq, exog, target)
    columns = feature_columns(frames["train"])
    best_config: ProphetConfig | None = None
    best_row: dict[str, float] | None = None

    for config in tqdm(grid, desc=f"tune {target}/{freq}/{exog}", leave=False):
        row = evaluate_validation(frames, columns, config, freq)
        row.update(asdict(config))
        if best_row is None or row["qlike"] < best_row["qlike"]:
            best_row = row
            best_config = config

    if best_config is None or best_row is None:
        raise RuntimeError(f"No valid Prophet configuration for {target}/{freq}/{exog}")
    return best_config, best_row


def expanding_test_forecast(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: ProphetConfig,
    freq: str,
) -> pd.DataFrame:
    cadence = REFIT_CADENCE[freq]
    test = frames["test"].reset_index(drop=True)
    rows = []
    model: Prophet | None = None

    for step in tqdm(range(len(test)), desc=f"test ECV {freq}", leave=False):
        if model is None or step % cadence == 0:
            history = history_until(frames, step)
            model = fit_prophet(history, columns, config, freq)

        test_row = test.iloc[[step]]
        pred_var = predict_one(model, test_row, columns)
        y_true = float(realized_variance(test_row["ret"])[0])
        ret_pct = float(returns_pct(test_row["ret"])[0])
        rows.append(build_forecast_row(test_row["date"].iloc[0], ret_pct, y_true, pred_var))

    return pd.DataFrame(rows)


def default_grid(args: argparse.Namespace) -> list[ProphetConfig]:
    seasonality_modes = parse_str_list(args.seasonality_modes)
    cps = parse_float_list(args.changepoint_priors)
    sps = parse_float_list(args.seasonality_priors)
    grid: list[ProphetConfig] = []
    for mode, cp, sp in itertools.product(seasonality_modes, cps, sps):
        grid.append(
            ProphetConfig(
                seasonality_mode=mode,
                changepoint_prior_scale=cp,
                seasonality_prior_scale=sp,
                use_regressors=not args.disable_regressors,
                yearly_seasonality="auto",
                weekly_seasonality="auto",
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
    for target, freq, exog in tqdm(cells, desc="Prophet cells"):
        print(f"Tuning Prophet on {target}/{freq}/{exog}...")
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

    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "prophet_validation_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT_DIR / "prophet_test_results.csv", index=False)
    with (OUT_DIR / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"Done. Results saved under {OUT_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prophet volatility forecasting.")
    add_grid_arguments(parser)
    parser.add_argument(
        "--seasonality-modes", default="additive,multiplicative",
        help="Comma list of seasonality_mode values to grid over.",
    )
    parser.add_argument(
        "--changepoint-priors", default="0.05,0.5",
        help="Comma list of changepoint_prior_scale values.",
    )
    parser.add_argument(
        "--seasonality-priors", default="10.0",
        help="Comma list of seasonality_prior_scale values.",
    )
    parser.add_argument(
        "--disable-regressors", action="store_true",
        help="Fit Prophet without lagged feature regressors (univariate baseline).",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
