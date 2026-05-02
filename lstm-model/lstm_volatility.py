"""
LSTM volatility forecasting for the 12 split cells in data/splits.

The model predicts one-period-ahead realized variance, defined as
100 * ret squared, from the lagged feature columns already built in the split
CSVs. Hyperparameters are selected on validation data, then the selected
blueprint is refit on train + validation and evaluated on test with expanding
cross-validation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm



ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]
REFIT_CADENCE = {"daily": 20, "weekly": 4}
VAR_LEVELS = (0.01, 0.05)


@dataclass(frozen=True)
class LSTMConfig:
    lookback: int = 22
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 80
    patience: int = 10


class VolatilityLSTM(nn.Module):
    def __init__(self, n_features: int, config: LSTMConfig):
        super().__init__()
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cell(freq: str, exog: str, target: str) -> dict[str, pd.DataFrame]:
    base = SPLIT_DIR / freq / exog / target
    frames = {}
    for stage in ("train", "val", "test"):
        path = base / f"{stage}.csv"
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        frames[stage] = df.reset_index(drop=True)
    return frames


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"date", "ret"}]


def realized_variance(ret: pd.Series) -> np.ndarray:
    returns_pct = ret.to_numpy(dtype=np.float32) * 100.0
    return np.square(returns_pct).astype(np.float32)


def make_sequences(
    df: pd.DataFrame,
    columns: list[str],
    lookback: int,
    x_scaler: StandardScaler,
    start_output_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_scaled = x_scaler.transform(df[columns].to_numpy(dtype=np.float32))
    y = realized_variance(df["ret"])
    dates = df["date"].to_numpy()
    returns_pct = df["ret"].to_numpy(dtype=np.float32) * 100.0

    x_seq, y_seq, date_seq, ret_seq = [], [], [], []
    for output_idx in range(max(lookback - 1, start_output_idx), len(df)):
        start = output_idx - lookback + 1
        x_seq.append(x_scaled[start : output_idx + 1])
        y_seq.append(y[output_idx])
        date_seq.append(dates[output_idx])
        ret_seq.append(returns_pct[output_idx])

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(date_seq),
        np.asarray(ret_seq, dtype=np.float32),
    )


def fit_scaler(df: pd.DataFrame, columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[columns].to_numpy(dtype=np.float32))
    return scaler


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray | None,
    val_y: np.ndarray | None,
    config: LSTMConfig,
    device: torch.device,
    seed: int,
    progress_label: str | None = None,
    show_progress: bool = False,
) -> tuple[VolatilityLSTM, float]:
    set_seed(seed)
    model = VolatilityLSTM(train_x.shape[-1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    best_state = None
    best_val = math.inf
    stale_epochs = 0

    epoch_iter = range(config.epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc=progress_label or "training", leave=False)

    for _ in epoch_iter:
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if val_x is None or val_y is None or len(val_x) == 0:
            continue

        val_loss = evaluate_loss(model, val_x, val_y, device)
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif val_x is None:
        best_val = evaluate_loss(model, train_x, train_y, device)
    return model, best_val


def evaluate_loss(model: VolatilityLSTM, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
    return float(np.mean(np.square(pred - y)))


def predict(model: VolatilityLSTM, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
    return np.maximum(pred, 1e-8)


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def metrics(y_true: np.ndarray, pred_var: np.ndarray, returns_pct: np.ndarray) -> dict[str, float]:
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


def add_residual_columns(forecast_df: pd.DataFrame) -> pd.DataFrame:
    out = forecast_df.copy()
    out["std_resid"] = out["ret_pct"] / np.maximum(out["pred_vol"], 1e-8)
    out["squared_std_resid"] = np.square(out["std_resid"])
    return out


def plot_cell_diagnostics(
    frames: dict[str, pd.DataFrame],
    forecast_df: pd.DataFrame,
    target: str,
    freq: str,
    exog: str,
) -> None:
    plot_dir = OUT_DIR / "plots" / target / freq / exog
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
        forecast_plot["date"],
        forecast_plot["observed_vol"],
        color="#1f77b4",
        linewidth=1.1,
        label="Test observed vol",
    )
    plt.plot(
        forecast_plot["date"],
        forecast_plot["pred_vol"],
        color="#d62728",
        linewidth=1.1,
        label="Test predicted vol",
    )
    plt.title(f"{target} {freq} {exog}: observed vs predicted volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "volatility_forecast_timeseries.png", dpi=150)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(forecast_plot["date"], forecast_plot["std_resid"], color="#4c78a8", linewidth=0.9)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(f"{target} {freq} {exog}: standardized residuals")
    plt.xlabel("Date")
    plt.ylabel("Return / predicted volatility")
    plt.tight_layout()
    plt.savefig(plot_dir / "standardized_residuals.png", dpi=150)
    plt.close()

    max_lags = min(40, max(1, len(forecast_plot) // 4))
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(forecast_plot["std_resid"].dropna(), lags=max_lags, ax=ax)
    ax.set_title(f"{target} {freq} {exog}: ACF of standardized residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "acf_standardized_residuals.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(forecast_plot["squared_std_resid"].dropna(), lags=max_lags, ax=ax)
    ax.set_title(f"{target} {freq} {exog}: ACF of squared standardized residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "acf_squared_standardized_residuals.png", dpi=150)
    plt.close(fig)


def build_train_val_arrays(
    frames: dict[str, pd.DataFrame],
    config: LSTMConfig,
) -> tuple[list[str], StandardScaler, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    train_df = frames["train"]
    val_df = frames["val"]
    columns = feature_columns(train_df)
    scaler = fit_scaler(train_df, columns)

    train_arrays = make_sequences(train_df, columns, config.lookback, scaler)
    combined = pd.concat([train_df, val_df], ignore_index=True)
    val_arrays = make_sequences(
        combined,
        columns,
        config.lookback,
        scaler,
        start_output_idx=len(train_df),
    )
    return columns, scaler, train_arrays, val_arrays


def tune_cell(
    freq: str,
    exog: str,
    target: str,
    grid: list[LSTMConfig],
    device: torch.device,
    seed: int,
    show_epoch_progress: bool,
) -> tuple[LSTMConfig, dict[str, float]]:
    frames = load_cell(freq, exog, target)
    best_config = None
    best_row = None

    config_iter = tqdm(grid, desc=f"tune {target}/{freq}/{exog}", leave=False)
    for idx, config in enumerate(config_iter):
        _, _, train_arrays, val_arrays = build_train_val_arrays(frames, config)
        train_x, train_y = train_arrays[0], train_arrays[1]
        val_x, val_y, _, val_ret = val_arrays
        model, _ = train_model(
            train_x,
            train_y,
            val_x,
            val_y,
            config,
            device,
            seed + idx,
            progress_label=f"epochs {target}/{freq}/{exog}",
            show_progress=show_epoch_progress,
        )
        pred = predict(model, val_x, device)
        row = metrics(val_y, pred, val_ret)
        row.update(asdict(config))
        if best_row is None or row["qlike"] < best_row["qlike"]:
            best_row = row
            best_config = config

    if best_config is None or best_row is None:
        raise RuntimeError(f"No valid LSTM configuration for {target}/{freq}/{exog}")
    return best_config, best_row


def refit_train_val(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: LSTMConfig,
    device: torch.device,
    seed: int,
) -> tuple[VolatilityLSTM, StandardScaler]:
    train_val = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    scaler = fit_scaler(train_val, columns)
    x, y, _, _ = make_sequences(train_val, columns, config.lookback, scaler)
    model, _ = train_model(x, y, None, None, config, device, seed)
    return model, scaler


def expanding_test_forecast(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: LSTMConfig,
    freq: str,
    device: torch.device,
    seed: int,
    show_epoch_progress: bool,
) -> pd.DataFrame:
    history = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    test = frames["test"].reset_index(drop=True)
    cadence = REFIT_CADENCE[freq]
    rows = []
    model = None
    scaler = None

    step_iter = tqdm(range(len(test)), desc=f"test ECV {freq}", leave=False)
    for step in step_iter:
        if model is None or step % cadence == 0:
            scaler = fit_scaler(history, columns)
            x_hist, y_hist, _, _ = make_sequences(history, columns, config.lookback, scaler)
            model, _ = train_model(
                x_hist,
                y_hist,
                None,
                None,
                config,
                device,
                seed + step,
                progress_label=f"refit step {step}",
                show_progress=show_epoch_progress,
            )

        forecast_context = pd.concat([history, test.iloc[[step]]], ignore_index=True)
        x_step, y_step, dates, ret_step = make_sequences(
            forecast_context,
            columns,
            config.lookback,
            scaler,
            start_output_idx=len(forecast_context) - 1,
        )
        pred_var = predict(model, x_step, device)[0]
        rows.append(
            {
                "date": pd.Timestamp(dates[0]).date().isoformat(),
                "ret_pct": float(ret_step[0]),
                "realized_var": float(y_step[0]),
                "pred_var": float(pred_var),
                "pred_vol": float(np.sqrt(pred_var)),
                "VaR_1": float(NormalDist().inv_cdf(0.01) * np.sqrt(pred_var)),
                "VaR_5": float(NormalDist().inv_cdf(0.05) * np.sqrt(pred_var)),
            }
        )
        history = pd.concat([history, test.iloc[[step]]], ignore_index=True)

    return pd.DataFrame(rows)


def default_grid(args: argparse.Namespace) -> list[LSTMConfig]:
    lookbacks = parse_int_list(args.lookbacks)
    hidden_sizes = parse_int_list(args.hidden_sizes)
    dropouts = parse_float_list(args.dropouts)
    learning_rates = parse_float_list(args.learning_rates)

    grid = []
    for lookback, hidden_size, dropout, lr in itertools.product(
        lookbacks, hidden_sizes, dropouts, learning_rates
    ):
        grid.append(
            LSTMConfig(
                lookback=lookback,
                hidden_size=hidden_size,
                num_layers=args.num_layers,
                dropout=dropout,
                learning_rate=lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
            )
        )
    return grid


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.cpu)
    print(f"Using device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "forecasts").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    targets = parse_selection(args.targets, TARGETS)
    freqs = parse_selection(args.freqs, FREQS)
    exogs = parse_selection(args.exogs, EXOGS)
    grid = default_grid(args)

    selection_rows = []
    test_rows = []

    cells = list(itertools.product(targets, freqs, exogs))
    for target, freq, exog in tqdm(cells, desc="LSTM cells"):
        print(f"Tuning {target}/{freq}/{exog} on validation...")
        best_config, val_row = tune_cell(
            freq,
            exog,
            target,
            grid,
            device,
            args.seed,
            args.show_epoch_progress,
        )
        val_row.update({"target": target, "freq": freq, "exog": exog})
        selection_rows.append(val_row)

        print(f"Refitting {target}/{freq}/{exog} and running expanding test forecast...")
        frames = load_cell(freq, exog, target)
        columns = feature_columns(frames["train"])
        forecast_df = expanding_test_forecast(
            frames,
            columns,
            best_config,
            freq,
            device,
            args.seed + 10_000,
            args.show_epoch_progress,
        )
        forecast_path = OUT_DIR / "forecasts" / f"{target}_{freq}_{exog}_test_forecasts.csv"
        forecast_df = add_residual_columns(forecast_df)
        forecast_df.to_csv(forecast_path, index=False)

        if not args.no_plots:
            plot_cell_diagnostics(frames, forecast_df, target, freq, exog)

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
                "forecast_file": str(forecast_path.relative_to(ROOT)),
                **asdict(best_config),
            }
        )
        test_rows.append(test_metric)

    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "lstm_validation_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT_DIR / "lstm_test_results.csv", index=False)
    with (OUT_DIR / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    print(f"Done. Results saved under {OUT_DIR.relative_to(ROOT)}")


def parse_selection(raw: str, allowed: list[str]) -> list[str]:
    if raw.lower() == "all":
        return allowed
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    bad = sorted(set(selected) - set(allowed))
    if bad:
        raise ValueError(f"Invalid values {bad}; allowed values are {allowed}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM volatility models.")
    parser.add_argument("--targets", default="all", help="Comma list or 'all': SPY,OIL,GOLD")
    parser.add_argument("--freqs", default="all", help="Comma list or 'all': daily,weekly")
    parser.add_argument("--exogs", default="all", help="Comma list or 'all': no_exog,with_exog")
    parser.add_argument("--lookbacks", default="10,22", help="Comma-separated sequence lengths.")
    parser.add_argument("--hidden-sizes", default="32,64", help="Comma-separated hidden sizes.")
    parser.add_argument("--dropouts", default="0.0,0.2", help="Comma-separated dropout rates.")
    parser.add_argument("--learning-rates", default="0.001", help="Comma-separated learning rates.")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic plot generation.")
    parser.add_argument(
        "--show-epoch-progress",
        action="store_true",
        help="Show nested epoch progress bars. Off by default to keep output compact.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
