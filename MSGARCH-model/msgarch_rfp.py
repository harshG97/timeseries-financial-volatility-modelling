"""
MS-GARCH Random Forecast Periods (RFP) evaluation.

Evaluates the MSGARCH blueprints across historical shock regimes.
For each window:
1. ARX mean equation (with_exog only) and MSGARCH variance model are fitted
   ONCE on data up to fit_end. No refit during the forecast period.
2. The window is forecast sequentially: at step t, MSGARCH conditions on
   observed residuals up to t-1 (computed using the fixed ARX params for
   with_exog cells, or raw returns for no_exog) and produces a 1-step-ahead
   variance forecast for t.
"""

import argparse
import itertools
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from tqdm.auto import tqdm
from scipy.stats import norm
import seaborn as sns

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

warnings.filterwarnings("ignore")
msgarch = importr("MSGARCH")
stats = importr("stats")

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.rfp_generator import RFPGenerator

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "rfp"
GRID_RESULTS_PATH = Path(__file__).resolve().parent / "outputs" / "msgarch_grid_search_results.csv"

TARGETS = ["SPY", "OIL", "GOLD"]
FREQS = ["daily", "weekly"]
EXOGS = ["no_exog", "with_exog"]
REGIMES = ["GFC", "COVID", "OIL_CRASH", "ENERGY_22", "CALM_17_19"]


def parse_selection(raw: str, allowed: list[str]) -> list[str]:
    if raw.lower() == "all":
        return allowed
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    bad = sorted(set(selected) - set(allowed))
    if bad:
        raise ValueError(f"Invalid values {bad}; allowed values are {allowed}")
    return selected


def metrics(y_true: np.ndarray, pred_var: np.ndarray, returns_pct: np.ndarray) -> dict:
    pred_var = np.maximum(pred_var, 1e-8)
    errors = pred_var - y_true
    out = {
        "mse": float(np.mean(np.square(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "qlike": float(np.mean(np.log(pred_var) + y_true / pred_var)),
    }
    sigma = np.sqrt(pred_var)
    for level in [0.01, 0.05]:
        var_forecast = norm.ppf(level) * sigma
        hits = returns_pct < var_forecast
        out[f"var_{int(level * 100)}_hit_rate"] = float(np.mean(hits))
        out[f"var_{int(level * 100)}_exceptions"] = int(np.sum(hits))
    return out


def evaluate_window(
    train_df: pd.DataFrame, 
    forecast_df: pd.DataFrame, 
    spec_params: dict,
    window_id: str
) -> pd.DataFrame:
    exog = spec_params["exog"]
    use_exog = (exog == "with_exog")
    
    ENDO_COLS = {"ret", "date", "ret_lag1", "ret_sq_lag1", "neg_ret_sq_lag1", "RV_5_lag1", "RV_10_lag1", "RV_22_lag1"}
    exog_cols = [c for c in train_df.columns if c not in ENDO_COLS] if use_exog else []

    all_data = pd.concat([train_df, forecast_df], ignore_index=True)
    returns = all_data["ret"] * 100  # pct returns; positional indexing only
    exog_data = all_data[exog_cols] if use_exog else None

    k = spec_params["k"]
    model_type = spec_params["model"]
    dist = spec_params["dist"]

    spec = msgarch.CreateSpec(
        variance_spec=robjects.ListVector({"model": robjects.StrVector([model_type] * k)}),
        distribution_spec=robjects.ListVector({"distribution": robjects.StrVector([dist] * k)}),
        switch_spec=robjects.ListVector({"do.mix": False})
    )

    eval_start_idx = len(train_df)
    n_eval = len(forecast_df)

    # ---- ARX mean (fit ONCE on training data) ------------------------------
    # For with_exog: y_t = Const + beta @ x_t + eps_t (lags=0).
    # We then compute residuals for the whole concatenated series using the
    # fixed fitted params — no refit per step.
    train_y = returns.iloc[:eval_start_idx]
    if use_exog:
        train_x = exog_data.iloc[:eval_start_idx]
        arx_model = arch_model(
            train_y, x=train_x, mean="ARX", lags=0, vol="Constant"
        ).fit(disp="off")
        params = arx_model.params
        missing = [c for c in exog_data.columns if c not in params.index]
        if missing:
            raise RuntimeError(
                f"ARX fitted params missing exog columns {missing}; "
                f"got params={list(params.index)}"
            )
        beta = params.loc[list(exog_data.columns)].values
        const = float(params.get("Const", 0.0))
        full_resids = returns.values - const - exog_data.values @ beta
    else:
        full_resids = returns.values

    # ---- MSGARCH (fit ONCE on training residuals) --------------------------
    try:
        current_fit = msgarch.FitML(
            spec=spec, data=robjects.FloatVector(full_resids[:eval_start_idx])
        )
    except Exception as e:
        raise RuntimeError("MSGARCH fit failed for window!") from e

    # ---- Sequential 1-step-ahead forecasts (no refit) ----------------------
    preds = []
    for i in range(n_eval):
        current_t = eval_start_idx + i
        r_resids = robjects.FloatVector(full_resids[:current_t])
        fc = stats.predict(object=current_fit, newdata=r_resids, nahead=1)
        vol_pred = float(fc.rx2("vol")[0])
        var_pred = vol_pred ** 2

        actual_ret_pct = float(returns.iloc[current_t])
        preds.append({
            "date": all_data["date"].iloc[current_t],
            "ret_pct": actual_ret_pct,
            "realized_var": actual_ret_pct ** 2,
            "pred_var": var_pred,
            "pred_vol": vol_pred,
            "VaR_1": float(norm.ppf(0.01) * vol_pred),
            "VaR_5": float(norm.ppf(0.05) * vol_pred),
        })

    df_preds = pd.DataFrame(preds)
    df_preds["std_resid"] = df_preds["ret_pct"] / np.maximum(df_preds["pred_vol"], 1e-8)
    df_preds["squared_std_resid"] = np.square(df_preds["std_resid"])
    return df_preds


def generate_plots(rfp_results: pd.DataFrame):
    """Generate 5 RFP diagnostic plot types."""
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Regime Bars
    bar_dir = plots_dir / "regime_bars"
    bar_dir.mkdir(exist_ok=True)
    for (t, f, e), grp in rfp_results.groupby(["target", "freq", "exog"]):
        mean_qlike = grp.groupby("regime")["qlike"].mean().sort_values()
        plt.figure(figsize=(10, 6))
        mean_qlike.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title(f"Mean QLIKE by Regime: {t} {f} ({e})")
        plt.ylabel("QLIKE")
        plt.tight_layout()
        plt.savefig(bar_dir / f"{t}_{f}_{e}.png")
        plt.close()

    # 3. Ablation
    ablation_dir = plots_dir / "ablation"
    ablation_dir.mkdir(exist_ok=True)
    for (t, f), grp in rfp_results.groupby(["target", "freq"]):
        mean_df = grp.groupby(["regime", "exog"])["qlike"].mean().unstack()
        if not mean_df.empty and mean_df.shape[1] == 2:
            mean_df.plot(kind="bar", figsize=(10, 6), edgecolor="black")
            plt.title(f"MS-GARCH Ablation (no_exog vs with_exog): {t} {f}")
            plt.ylabel("Mean QLIKE")
            plt.tight_layout()
            plt.savefig(ablation_dir / f"{t}_{f}.png")
            plt.close()

    # 4. Heatmap
    heatmap_data = rfp_results.groupby(["target", "freq", "exog", "regime"])["qlike"].mean().reset_index()
    heatmap_data["cell"] = heatmap_data["target"] + "_" + heatmap_data["freq"] + "_" + heatmap_data["exog"]
    pivot = heatmap_data.pivot(index="cell", columns="regime", values="qlike")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title("Mean QLIKE Heatmap (Cells x Regimes)")
    plt.tight_layout()
    plt.savefig(plots_dir / "heatmap_qlike.png")
    plt.close()

    # 5. Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=rfp_results, x="regime", y="qlike", hue="exog")
    plt.title("QLIKE Distribution by Regime and Feature Set")
    plt.tight_layout()
    plt.savefig(plots_dir / "boxplot_regime_qlike.png")
    plt.close()


def run(args: argparse.Namespace):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not GRID_RESULTS_PATH.exists():
        print("Error: run msgarch_grid_search.py first.")
        return

    grid_df = pd.read_csv(GRID_RESULTS_PATH)
    targets = parse_selection(args.targets, TARGETS)
    freqs = parse_selection(args.freqs, FREQS)
    exogs = parse_selection(args.exogs, EXOGS)
    
    rfp_gen = RFPGenerator(splits_dir=SPLIT_DIR)
    
    all_results = []
    fc_dir = OUT_DIR / "forecasts"
    fc_dir.mkdir(exist_ok=True)
    
    per_window_plot_dir = OUT_DIR / "plots" / "per_window"
    
    cells = list(itertools.product(targets, freqs, exogs))
    print(f"Running MS-GARCH RFP on {len(cells)} cells...")

    for t, f, e in tqdm(cells, desc="RFP cells"):
        match = grid_df[(grid_df["target"] == t) & (grid_df["freq"] == f) & (grid_df["exog"] == e)]
        if match.empty:
            continue
            
        spec = {
            "target": t, "freq": f, "exog": e,
            "k": int(match.iloc[0]["k"]),
            "model": match.iloc[0]["model"],
            "dist": match.iloc[0]["dist"]
        }
        
        try:
            windows = list(rfp_gen.iter_windows(freq=f, target=t, use_exog=(e == "with_exog")))
        except FileNotFoundError:
            continue
            
        for w in tqdm(windows, desc=f"windows {t}/{f}/{e}", leave=False):
            if args.regimes != "all" and w.regime not in parse_selection(args.regimes, REGIMES):
                continue
                
            try:
                preds = evaluate_window(w.train, w.forecast, spec, w.window_id)
                met = metrics(preds["realized_var"], preds["pred_var"], preds["ret_pct"])

                # Save forecast
                preds.to_csv(fc_dir / f"{t}_{f}_{e}_{w.window_id}.csv", index=False)

                res = {"target": t, "freq": f, "exog": e, "regime": w.regime, "window_id": w.window_id}
                res.update({k: spec[k] for k in ["k", "model", "dist"]})
                res.update(met)
                res["n_forecast"] = len(preds)
                all_results.append(res)

                if not args.no_plots:
                    wd_plot = per_window_plot_dir / t / f / e
                    wd_plot.mkdir(parents=True, exist_ok=True)
                    dates = pd.to_datetime(preds["date"])
                    plt.figure(figsize=(10, 4))
                    plt.plot(dates, np.sqrt(preds["realized_var"]),
                             color="#1f77b4", linewidth=1.2, label="Observed vol")
                    plt.plot(dates, preds["pred_vol"],
                             color="#d62728", linewidth=1.2, label="Predicted vol")
                    plt.title(f"[{w.regime}] {t} {f} {e} - {w.window_id}")
                    plt.xlabel("Date"); plt.ylabel("Volatility (%)")
                    plt.legend(fontsize=9)
                    plt.tight_layout()
                    plt.savefig(wd_plot / f"{w.window_id}.png", dpi=150)
                    plt.close()
                    
            except Exception as exc:
                print(f"Error on {t}/{f}/{e} window {w.window_id}: {exc}")

    if not all_results:
        print("No windows evaluated.")
        return

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUT_DIR / "msgarch_rfp_results.csv", index=False)

    # Canonical summary: mean/median per metric per (target, freq, exog, regime).
    # Matches the schema produced by garch_rfp.py / transformer_rfp.py / lstm_rfp.py.
    group_cols = ["target", "freq", "exog", "regime"]
    metric_cols = ["mse", "rmse", "mae", "qlike",
                   "var_1_hit_rate", "var_5_hit_rate"]
    available = [c for c in metric_cols if c in res_df.columns]
    summary = res_df.groupby(group_cols)[available].agg(["mean", "median"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / "msgarch_rfp_summary.csv", index=False)
    
    print("\nResults saved to MSGARCH-model/outputs/rfp/")
    
    if not args.no_plots:
        print("Generating aggregate plots...")
        generate_plots(res_df)
    
    print("Done.")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="all")
    p.add_argument("--freqs", default="all")
    p.add_argument("--exogs", default="all")
    p.add_argument("--regimes", default="all")
    p.add_argument("--no-plots", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
