"""
Transformer (encoder-only, sinusoidal positional encoding) volatility
forecasting on the 12 split cells in ``data/splits``.

The architecture mirrors the LSTM baseline as closely as possible so that
the comparison isolates the temporal-modeling backbone:

- Same input window: a lookback of past lagged-feature vectors.
- Same target: one-step-ahead realized variance ``(100 * ret)^2``.
- Same training loop: AdamW, MSE loss, gradient clipping, val-loss early
  stopping, soft-plus head to keep variance positive.
- Same outputs format and same expanding-CV refit cadence on the test block.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
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
    load_cell,
    metrics,
    parse_float_list,
    parse_int_list,
    parse_selection,
    plot_cell_diagnostics,
    realized_variance,
    returns_pct,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "transformer"
MODEL_LABEL = "Transformer"


@dataclass(frozen=True)
class TransformerConfig:
    lookback: int = 22
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 80
    patience: int = 10


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class VolatilityTransformer(nn.Module):
    def __init__(self, n_features: int, config: TransformerConfig) -> None:
        super().__init__()
        # Force d_model to be divisible by nhead.
        d_model = config.d_model
        if d_model % config.nhead != 0:
            d_model = ((d_model // config.nhead) + 1) * config.nhead
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(config.lookback, 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(d_model, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_scaler(df: pd.DataFrame, columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[columns].to_numpy(dtype=np.float32))
    return scaler


def make_sequences(
    df: pd.DataFrame,
    columns: list[str],
    lookback: int,
    x_scaler: StandardScaler,
    start_output_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_scaled = x_scaler.transform(df[columns].to_numpy(dtype=np.float32))
    y = realized_variance(df["ret"]).astype(np.float32)
    dates = df["date"].to_numpy()
    rets_pct = (df["ret"].to_numpy(dtype=np.float32) * 100.0)

    x_seq, y_seq, date_seq, ret_seq = [], [], [], []
    for output_idx in range(max(lookback - 1, start_output_idx), len(df)):
        start = output_idx - lookback + 1
        x_seq.append(x_scaled[start : output_idx + 1])
        y_seq.append(y[output_idx])
        date_seq.append(dates[output_idx])
        ret_seq.append(rets_pct[output_idx])

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(date_seq),
        np.asarray(ret_seq, dtype=np.float32),
    )


def evaluate_loss_t(model: VolatilityTransformer, x_t: torch.Tensor, y_t: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x_t)
    return float(torch.mean((pred - y_t) ** 2).item())


def predict(model: VolatilityTransformer, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device)
        pred = model(x_t).detach().cpu().numpy()
    return np.maximum(pred, 1e-8)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray | None,
    val_y: np.ndarray | None,
    config: TransformerConfig,
    device: torch.device,
    seed: int,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> tuple[VolatilityTransformer, float]:
    set_seed(seed)
    model = VolatilityTransformer(train_x.shape[-1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    # MPS / CPU need every mini-batch to ship from host -> device when using a
    # DataLoader, which utterly dominates runtime for this small model. Move
    # the *whole* training tensor to the target device once and slice in-place
    # via a permuted index buffer; this is also faster on CUDA for tensors
    # that fit in VRAM.
    train_x_t = torch.from_numpy(train_x).to(device)
    train_y_t = torch.from_numpy(train_y).to(device)
    n = train_x_t.size(0)
    batch_size = min(config.batch_size, max(1, n))

    val_x_t = torch.from_numpy(val_x).to(device) if val_x is not None and len(val_x) > 0 else None
    val_y_t = torch.from_numpy(val_y).to(device) if val_y is not None and len(val_y) > 0 else None

    best_state = None
    best_val = math.inf
    stale_epochs = 0

    epoch_iter = range(config.epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc=progress_label or "training", leave=False)

    for _ in epoch_iter:
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = train_x_t.index_select(0, idx)
            yb = train_y_t.index_select(0, idx)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if val_x_t is None or val_y_t is None:
            continue

        val_loss = evaluate_loss_t(model, val_x_t, val_y_t)
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
    elif val_x_t is None:
        best_val = evaluate_loss_t(model, train_x_t, train_y_t)
    return model, best_val


def get_device(force_cpu: bool, use_mps: bool) -> torch.device:
    """Pick a torch device.

    Priority: --cpu > CUDA (when available) > --mps (when available) > CPU.

    MPS is *not* auto-selected: empirically, for this small model + small
    batch workload, MPS on Apple Silicon is 2–4x slower than CPU because
    PyTorch CPU saturates all performance cores while MPS pays kernel-launch
    overhead per op. Pass ``--mps`` only if you've benchmarked it for your
    specific grid and confirmed it wins.
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if use_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_train_val_arrays(
    frames: dict[str, pd.DataFrame],
    config: TransformerConfig,
) -> tuple[list[str], StandardScaler, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    train_df = frames["train"]
    val_df = frames["val"]
    columns = feature_columns(train_df)
    scaler = fit_scaler(train_df, columns)
    train_arrays = make_sequences(train_df, columns, config.lookback, scaler)
    combined = pd.concat([train_df, val_df], ignore_index=True)
    val_arrays = make_sequences(
        combined, columns, config.lookback, scaler, start_output_idx=len(train_df),
    )
    return columns, scaler, train_arrays, val_arrays


def tune_cell(
    freq: str,
    exog: str,
    target: str,
    grid: list[TransformerConfig],
    device: torch.device,
    seed: int,
    show_epoch_progress: bool,
) -> tuple[TransformerConfig, dict[str, float]]:
    frames = load_cell(freq, exog, target)
    best_config: TransformerConfig | None = None
    best_row: dict[str, float] | None = None

    for idx, config in enumerate(tqdm(grid, desc=f"tune {target}/{freq}/{exog}", leave=False)):
        _, _, train_arrays, val_arrays = build_train_val_arrays(frames, config)
        train_x, train_y = train_arrays[0], train_arrays[1]
        val_x, val_y, _, val_ret = val_arrays
        model, _ = train_model(
            train_x, train_y, val_x, val_y, config, device, seed + idx,
            show_progress=show_epoch_progress,
            progress_label=f"epochs {target}/{freq}/{exog}",
        )
        pred = predict(model, val_x, device)
        row = metrics(val_y, pred, val_ret)
        row.update(asdict(config))
        if best_row is None or row["qlike"] < best_row["qlike"]:
            best_row = row
            best_config = config

    if best_config is None or best_row is None:
        raise RuntimeError(f"No valid Transformer configuration for {target}/{freq}/{exog}")
    return best_config, best_row


def expanding_test_forecast(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: TransformerConfig,
    freq: str,
    device: torch.device,
    seed: int,
    show_epoch_progress: bool,
) -> pd.DataFrame:
    history = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    test = frames["test"].reset_index(drop=True)
    cadence = REFIT_CADENCE[freq]
    rows: list[dict] = []
    model: VolatilityTransformer | None = None
    scaler: StandardScaler | None = None

    for step in tqdm(range(len(test)), desc=f"test ECV {freq}", leave=False):
        if model is None or step % cadence == 0:
            scaler = fit_scaler(history, columns)
            x_hist, y_hist, _, _ = make_sequences(history, columns, config.lookback, scaler)
            model, _ = train_model(
                x_hist, y_hist, None, None, config, device, seed + step,
                show_progress=show_epoch_progress,
                progress_label=f"refit step {step}",
            )

        forecast_context = pd.concat([history, test.iloc[[step]]], ignore_index=True)
        x_step, y_step, dates, ret_step = make_sequences(
            forecast_context, columns, config.lookback, scaler,
            start_output_idx=len(forecast_context) - 1,
        )
        pred_var = predict(model, x_step, device)[0]
        rows.append(build_forecast_row(dates[0], float(ret_step[0]), float(y_step[0]), float(pred_var)))
        history = pd.concat([history, test.iloc[[step]]], ignore_index=True)

    return pd.DataFrame(rows)


def default_grid(args: argparse.Namespace) -> list[TransformerConfig]:
    lookbacks = parse_int_list(args.lookbacks)
    d_models = parse_int_list(args.d_models)
    nheads = parse_int_list(args.nheads)
    layers = parse_int_list(args.num_layers)
    dropouts = parse_float_list(args.dropouts)
    learning_rates = parse_float_list(args.learning_rates)

    grid: list[TransformerConfig] = []
    for lookback, d_model, nhead, n_layers, dropout, lr in itertools.product(
        lookbacks, d_models, nheads, layers, dropouts, learning_rates
    ):
        if d_model % nhead != 0:
            continue  # skip incompatible head/dim combinations
        grid.append(
            TransformerConfig(
                lookback=lookback,
                d_model=d_model,
                nhead=nhead,
                num_layers=n_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=dropout,
                learning_rate=lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
            )
        )
    if not grid:
        raise ValueError("Empty Transformer grid; check d_model/nhead compatibility.")
    return grid


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.cpu, args.mps)
    print(f"Using device: {device}")
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
    for target, freq, exog in tqdm(cells, desc="Transformer cells"):
        print(f"Tuning Transformer on {target}/{freq}/{exog}...")
        best_config, val_row = tune_cell(
            freq, exog, target, grid, device, args.seed, args.show_epoch_progress,
        )
        val_row.update({"target": target, "freq": freq, "exog": exog})
        selection_rows.append(val_row)

        print(f"Refit + expanding test on {target}/{freq}/{exog}...")
        frames = load_cell(freq, exog, target)
        columns = feature_columns(frames["train"])
        forecast_df = expanding_test_forecast(
            frames, columns, best_config, freq, device,
            args.seed + 10_000, args.show_epoch_progress,
        )
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
                "target": target, "freq": freq, "exog": exog,
                "forecast_file": str(forecast_path.relative_to(Path(__file__).resolve().parents[1])),
                **asdict(best_config),
            }
        )
        test_rows.append(test_metric)

    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "transformer_validation_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT_DIR / "transformer_test_results.csv", index=False)
    with (OUT_DIR / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"Done. Results saved under {OUT_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transformer volatility forecasting.")
    add_grid_arguments(parser)
    # Defaults are deliberately single-config and compact so the full 12-cell
    # grid finishes in ~30-60 min on a modern laptop CPU. Pass explicit lists
    # to widen the grid (e.g. `--d-models 32,64`).
    parser.add_argument("--lookbacks", default="22", help="Comma list of lookback window sizes.")
    parser.add_argument("--d-models", default="32", help="Comma list of model embedding dims.")
    parser.add_argument("--nheads", default="4", help="Comma list of attention head counts.")
    parser.add_argument("--num-layers", default="1", help="Comma list of encoder layer counts.")
    parser.add_argument("--dropouts", default="0.1", help="Comma list of dropout values.")
    parser.add_argument("--learning-rates", default="0.001", help="Comma list of learning rates.")
    parser.add_argument("--dim-feedforward", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument(
        "--mps", action="store_true",
        help="Use Apple Silicon MPS (only when CUDA is not available). Empirically slower than CPU for this workload — opt-in only.",
    )
    parser.add_argument(
        "--show-epoch-progress", action="store_true",
        help="Show nested epoch progress bars during training.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
