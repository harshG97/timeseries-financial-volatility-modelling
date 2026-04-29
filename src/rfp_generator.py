"""
RFP (Random Forecast Periods) data generator.

Streams per-window train / forecast slices for any cell in the 12-cell volatility
modeling grid under ``data/splits/``. Each window provides:

* metadata: ``window_id``, ``regime``, ``fit_end``, ``forecast_start``, ``forecast_end``
* training data: all rows with ``date <= fit_end`` (drawn from train ∪ val of the cell)
* forecast data: rows with ``forecast_start <= date <= forecast_end``

Usage
-----

>>> from src.rfp_generator import RFPGenerator
>>> gen = RFPGenerator()                       # defaults to <repo>/data/splits
>>> for w in gen.iter_windows(freq="daily", target="SPY", use_exog=True):
...     model.fit(w.X_train, w.y_train)
...     preds = model.predict(w.X_forecast)
...     score = qlike(preds, w.y_forecast)

Filter by regime or specific window ids:

>>> for w in gen.iter_windows("daily", "SPY", True, regimes=["GFC", "COVID"]):
...     ...
>>> w = gen.get_window("d_GFC_1", target="OIL", use_exog=False)

The generator does not fit any models; it only assembles the data slices each
window needs. Models, scoring, and aggregation are the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

DEFAULT_SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"

VALID_FREQUENCIES = ("daily", "weekly")
VALID_TARGETS = ("SPY", "OIL", "GOLD")
TARGET_COL = "ret"
DATE_COL = "date"


@dataclass
class RFPWindow:
    """A single RFP window with its data slices ready for fitting and forecasting."""

    window_id: str
    regime: str
    freq: str
    target: str
    use_exog: bool
    fit_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    train: pd.DataFrame = field(repr=False)
    forecast: pd.DataFrame = field(repr=False)

    @property
    def y_train(self) -> pd.Series:
        return self.train[TARGET_COL]

    @property
    def X_train(self) -> pd.DataFrame:
        return self.train.drop(columns=[DATE_COL, TARGET_COL])

    @property
    def y_forecast(self) -> pd.Series:
        return self.forecast[TARGET_COL]

    @property
    def X_forecast(self) -> pd.DataFrame:
        return self.forecast.drop(columns=[DATE_COL, TARGET_COL])

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_forecast(self) -> int:
        return len(self.forecast)


class RFPGenerator:
    """Streams RFPWindow objects from materialised splits.

    Parameters
    ----------
    splits_dir : str or Path, optional
        Root of the splits tree. Defaults to ``<repo>/data/splits``.

    Notes
    -----
    Pre-test data (train + val concatenated, sorted by date) is the source for
    every window's training slice. This matches the RFP rule that windows lie
    strictly before the frequency's ``test_start``; depending on the regime,
    a window's ``fit_end`` may fall inside the validation period, in which
    case the model is fit on (train ∪ early-val).

    Loaded data is cached per (freq, sub, target) so iterating across windows
    of the same cell only reads each CSV once.
    """

    def __init__(self, splits_dir: str | Path = DEFAULT_SPLITS_DIR):
        self.splits_dir = Path(splits_dir)
        if not self.splits_dir.exists():
            raise FileNotFoundError(f"splits_dir does not exist: {self.splits_dir}")
        self._cell_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._window_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ public

    def iter_windows(
        self,
        freq: str,
        target: str,
        use_exog: bool,
        regimes: Optional[Iterable[str]] = None,
        window_ids: Optional[Iterable[str]] = None,
    ) -> Iterator[RFPWindow]:
        """Yield RFPWindow objects for the requested cell.

        Parameters
        ----------
        freq : {"daily", "weekly"}
        target : {"SPY", "OIL", "GOLD"}
        use_exog : bool
            ``True`` for ``with_exog`` cells, ``False`` for ``no_exog``.
        regimes : iterable of str, optional
            If provided, restrict to windows whose ``regime`` is in this set.
        window_ids : iterable of str, optional
            If provided, restrict to these specific window ids. Overrides
            ``regimes`` when both are given.
        """
        self._validate(freq, target)
        sub = "with_exog" if use_exog else "no_exog"
        data = self._load_cell(freq, sub, target)
        windows = self._load_windows(freq)
        if window_ids is not None:
            windows = windows[windows["window_id"].isin(set(window_ids))]
        elif regimes is not None:
            windows = windows[windows["regime"].isin(set(regimes))]
        for row in windows.itertuples(index=False):
            train = data[data[DATE_COL] <= row.fit_end].reset_index(drop=True)
            forecast = data[
                (data[DATE_COL] >= row.forecast_start)
                & (data[DATE_COL] <= row.forecast_end)
            ].reset_index(drop=True)
            yield RFPWindow(
                window_id=row.window_id,
                regime=row.regime,
                freq=freq,
                target=target,
                use_exog=use_exog,
                fit_end=row.fit_end,
                forecast_start=row.forecast_start,
                forecast_end=row.forecast_end,
                train=train,
                forecast=forecast,
            )

    def get_window(
        self, window_id: str, target: str, use_exog: bool
    ) -> RFPWindow:
        """Return a single RFPWindow by id. Frequency is inferred from the prefix."""
        if window_id.startswith("d_"):
            freq = "daily"
        elif window_id.startswith("w_"):
            freq = "weekly"
        else:
            raise ValueError(
                f"window_id {window_id!r} must start with 'd_' or 'w_'"
            )
        for w in self.iter_windows(
            freq=freq, target=target, use_exog=use_exog, window_ids=[window_id]
        ):
            return w
        raise KeyError(f"no window with id {window_id!r} for {freq}")

    def list_windows(self, freq: str) -> pd.DataFrame:
        """Return a copy of the window-definition table for the given frequency."""
        if freq not in VALID_FREQUENCIES:
            raise ValueError(f"freq must be one of {VALID_FREQUENCIES}, got {freq!r}")
        return self._load_windows(freq).copy()

    def cells(self) -> list[tuple[str, str, str]]:
        """Enumerate all 12 cells as (freq, sub, target) tuples."""
        return [
            (freq, sub, target)
            for freq in VALID_FREQUENCIES
            for sub in ("no_exog", "with_exog")
            for target in VALID_TARGETS
        ]

    # ------------------------------------------------------------------ private

    @staticmethod
    def _validate(freq: str, target: str) -> None:
        if freq not in VALID_FREQUENCIES:
            raise ValueError(f"freq must be one of {VALID_FREQUENCIES}, got {freq!r}")
        if target not in VALID_TARGETS:
            raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")

    def _load_cell(self, freq: str, sub: str, target: str) -> pd.DataFrame:
        key = (freq, sub, target)
        if key in self._cell_cache:
            return self._cell_cache[key]
        base = self.splits_dir / freq / sub / target
        train = pd.read_csv(base / "train.csv", parse_dates=[DATE_COL])
        val = pd.read_csv(base / "val.csv", parse_dates=[DATE_COL])
        df = pd.concat([train, val], ignore_index=True).sort_values(DATE_COL)
        df = df.reset_index(drop=True)
        self._cell_cache[key] = df
        return df

    def _load_windows(self, freq: str) -> pd.DataFrame:
        if freq in self._window_cache:
            return self._window_cache[freq]
        path = self.splits_dir / "rfp" / f"{freq}_windows.csv"
        df = pd.read_csv(
            path,
            parse_dates=["fit_end", "forecast_start", "forecast_end"],
        )
        self._window_cache[freq] = df
        return df


# ---------------------------------------------------------------------- CLI demo

def _demo() -> None:
    """Print a short summary of every cell × window combination."""
    gen = RFPGenerator()
    for freq in VALID_FREQUENCIES:
        windows = gen.list_windows(freq)
        print(f"\n[{freq}] {len(windows)} windows")
        if windows.empty:
            continue
        # show one example per cell to keep output compact
        for sub in ("no_exog", "with_exog"):
            for target in VALID_TARGETS:
                first = next(
                    gen.iter_windows(freq=freq, target=target, use_exog=(sub == "with_exog")),
                    None,
                )
                if first is None:
                    continue
                print(
                    f"  {sub:9s}/{target:4s}  "
                    f"first_window={first.window_id:18s}  "
                    f"X_train={first.X_train.shape}  "
                    f"X_forecast={first.X_forecast.shape}"
                )


if __name__ == "__main__":
    _demo()
