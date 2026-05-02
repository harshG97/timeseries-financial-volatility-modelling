"""
GJR-GARCH / GARCH-X grid search over the 12-cell volatility grid.

For ``no_exog`` cells: fits Constant-mean GJR-GARCH(p,o,q) on the target's
own returns (univariate).

For ``with_exog`` cells: fits ARX-mean GJR-GARCH(p,o,q) with cross-asset
and market features as mean-equation regressors. The exogenous info enters
through the mean, producing different innovations (ε_t) that feed into the
GARCH variance equation.

Best (p,o,q) per cell is selected by AIC on training data.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]

# Endogenous columns (present in both no_exog and with_exog)
ENDO_COLS = {"ret", "date", "ret_lag1", "ret_sq_lag1", "neg_ret_sq_lag1",
             "RV_5_lag1", "RV_10_lag1", "RV_22_lag1"}


def exog_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are exogenous (cross-asset/market features)."""
    return [c for c in df.columns if c not in ENDO_COLS]


def parse_selection(raw: str, allowed: list[str]) -> list[str]:
    if raw.lower() == "all":
        return allowed
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    bad = sorted(set(selected) - set(allowed))
    if bad:
        raise ValueError(f"Invalid values {bad}; allowed values are {allowed}")
    return selected


def garch_grid_search(
    target: str, freq: str, exog: str,
    p_range: range, o_range: range, q_range: range,
) -> dict:
    """Grid search for one cell, returning best params and AIC."""
    path = SPLIT_DIR / freq / exog / target / "train.csv"
    data = pd.read_csv(path, index_col="date", parse_dates=True)

    returns = data["ret"] * 100  # scale for convergence
    use_exog = exog == "with_exog"
    exog_df = data[exog_columns(data)] if use_exog else None

    best_aic = np.inf
    best_params: dict = {}

    combos = list(itertools.product(p_range, o_range, q_range))
    for p, o, q in combos:
        try:
            if use_exog and exog_df is not None:
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
            if res.aic < best_aic:
                best_aic = res.aic
                best_params = {"p": p, "o": o, "q": q}
        except Exception:
            continue

    return {
        "target": target,
        "freq": freq,
        "exog": exog,
        "best_p": best_params.get("p"),
        "best_o": best_params.get("o"),
        "best_q": best_params.get("q"),
        "aic": best_aic,
    }


def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = parse_selection(args.targets, TARGETS)
    freqs = parse_selection(args.freqs, FREQS)
    exogs = parse_selection(args.exogs, EXOGS)

    p_range = range(1, args.max_p + 1)
    o_range = range(0, args.max_o + 1)
    q_range = range(1, args.max_q + 1)

    cells = list(itertools.product(targets, freqs, exogs))
    results = []

    for target, freq, exog in tqdm(cells, desc="Grid search"):
        print(f"  {target}/{freq}/{exog} ...")
        result = garch_grid_search(target, freq, exog, p_range, o_range, q_range)
        results.append(result)
        print(f"    best: p={result['best_p']}, o={result['best_o']}, "
              f"q={result['best_q']}, AIC={result['aic']:.2f}")

    results_df = pd.DataFrame(results)
    out_path = OUT_DIR / "garch_grid_search_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path.relative_to(ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GJR-GARCH / GARCH-X grid search.")
    p.add_argument("--targets", default="all", help="SPY,OIL,GOLD or 'all'")
    p.add_argument("--freqs", default="all", help="daily,weekly or 'all'")
    p.add_argument("--exogs", default="all", help="no_exog,with_exog or 'all'")
    p.add_argument("--max-p", type=int, default=3, help="Max GARCH p (default: 3)")
    p.add_argument("--max-o", type=int, default=1, help="Max GJR o (default: 1)")
    p.add_argument("--max-q", type=int, default=3, help="Max GARCH q (default: 3)")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
