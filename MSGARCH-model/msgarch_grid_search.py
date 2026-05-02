"""
Markov-Switching GARCH (MS-GARCH) grid search over the 12-cell grid.

For ``no_exog`` cells: fits MS-GARCH on the target's own returns.
For ``with_exog`` cells: performs a 2-step procedure:
  1. Fit ARX-mean model using arch_model (Constant volatility) to extract residuals.
  2. Fit MS-GARCH on the residuals.

Best spec per cell is selected by AIC on training data.
"""

import argparse
import itertools
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from tqdm.auto import tqdm

# rpy2 for R integration
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

warnings.filterwarnings("ignore")

msgarch = importr("MSGARCH")

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]

ENDO_COLS = {"ret", "date", "ret_lag1", "ret_sq_lag1", "neg_ret_sq_lag1",
             "RV_5_lag1", "RV_10_lag1", "RV_22_lag1"}


def exog_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ENDO_COLS]


def parse_selection(raw: str, allowed: list[str]) -> list[str]:
    if raw.lower() == "all":
        return allowed
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    bad = sorted(set(selected) - set(allowed))
    if bad:
        raise ValueError(f"Invalid values {bad}; allowed values are {allowed}")
    return selected


def get_arx_residuals(returns: pd.Series, exog_df: pd.DataFrame) -> np.ndarray:
    """Fit ARX mean with constant volatility and return residuals."""
    if exog_df is None or len(exog_df.columns) == 0:
        return returns.values
    model = arch_model(returns, x=exog_df, mean="ARX", lags=0, vol="Constant")
    res = model.fit(disp="off")
    return res.resid.values


def msgarch_grid_search(target: str, freq: str, exog: str) -> dict:
    path = SPLIT_DIR / freq / exog / target / "train.csv"
    data = pd.read_csv(path, index_col="date", parse_dates=True)

    # Scale returns
    returns = data["ret"] * 100
    use_exog = (exog == "with_exog")
    exog_df = data[exog_columns(data)] if use_exog else None

    # Step 1: ARX filter if with_exog
    if use_exog:
        y = get_arx_residuals(returns, exog_df)
    else:
        y = returns.values

    # Convert to R vector
    y_r = robjects.FloatVector(y)

    grid_regimes = [1, 2]
    grid_models = ["sGARCH", "gjrGARCH"]
    grid_dists = ["norm", "std"]

    best_aic = np.inf
    best_params = {}

    combos = list(itertools.product(grid_regimes, grid_models, grid_dists))
    for k, model_type, dist in combos:
        try:
            # Create MSGARCH spec in R
            spec = msgarch.CreateSpec(
                variance_spec=robjects.ListVector({"model": robjects.StrVector([model_type] * k)}),
                distribution_spec=robjects.ListVector({"distribution": robjects.StrVector([dist] * k)}),
                switch_spec=robjects.ListVector({"do.mix": False})
            )
            
            # Fit model
            fit = msgarch.FitML(spec=spec, data=y_r)
            
            # Extract loglik and calculate AIC
            loglik = fit.rx2("loglik")[0]
            n_par = len(fit.rx2("par"))
            aic = 2 * n_par - 2 * loglik
            
            if aic < best_aic:
                best_aic = aic
                best_params = {"k": k, "model": model_type, "dist": dist}
                
        except Exception as e:
            # R might fail to converge for some complex specs
            continue

    return {
        "target": target,
        "freq": freq,
        "exog": exog,
        "k": best_params.get("k", 1),
        "model": best_params.get("model", "sGARCH"),
        "dist": best_params.get("dist", "norm"),
        "aic": best_aic,
    }


def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = parse_selection(args.targets, TARGETS)
    freqs = parse_selection(args.freqs, FREQS)
    exogs = parse_selection(args.exogs, EXOGS)

    cells = list(itertools.product(targets, freqs, exogs))
    results = []

    for target, freq, exog in tqdm(cells, desc="MS-GARCH Grid Search"):
        print(f"  {target}/{freq}/{exog} ...")
        result = msgarch_grid_search(target, freq, exog)
        results.append(result)
        print(f"    best: k={result['k']}, model={result['model']}, "
              f"dist={result['dist']}, AIC={result['aic']:.2f}")

    results_df = pd.DataFrame(results)
    out_path = OUT_DIR / "msgarch_grid_search_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path.relative_to(ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MS-GARCH grid search.")
    p.add_argument("--targets", default="all", help="SPY,OIL,GOLD or 'all'")
    p.add_argument("--freqs", default="all", help="daily,weekly or 'all'")
    p.add_argument("--exogs", default="all", help="no_exog,with_exog or 'all'")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
