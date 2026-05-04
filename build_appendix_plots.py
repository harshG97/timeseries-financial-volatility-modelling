"""Generate the four supplementary figures for Appendix C of the Final Report.

Saves PNGs into eda_outputs/appendix/ so they are reproducible and live
next to the existing EDA figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "eda_outputs" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("GJR-GARCH",       ROOT / "ARMA-GARCH-model" / "outputs"),
    ("MS-GARCH",        ROOT / "MSGARCH-model" / "outputs"),
    ("LSTM-Attention",  ROOT / "LSTM-Attention-model" / "outputs"),
    ("Transformer",     ROOT / "additional-models" / "outputs" / "transformer"),
    ("XGBoost",         ROOT / "xgboost-model" / "outputs"),
]
EXOG = "no_exog"
WINDOW_ID = "d_COVID_3"  # representative COVID window
ACF_LAGS = 30


def _load_test_forecast(base: Path) -> pd.DataFrame:
    """Load a model's test-block forecast and harmonise the schema (MS-GARCH
    uses different column names)."""
    fc = pd.read_csv(base / "forecasts" / f"SPY_daily_{EXOG}_test_forecasts.csv")
    rename = {
        "ret": "ret_pct",
        "realized_variance": "realized_var",
        "pred_variance": "pred_var",
        "pred_volatility": "pred_vol",
    }
    fc = fc.rename(columns={k: v for k, v in rename.items() if k in fc.columns})
    if "ret_pct" in fc.columns and fc["ret_pct"].abs().max() < 1.0:
        # MS-GARCH stores raw log returns, not percent.
        fc["ret_pct"] = fc["ret_pct"] * 100.0
        if "realized_var" in fc.columns:
            fc["realized_var"] = fc["ret_pct"] ** 2
        if "pred_var" in fc.columns:
            # MS-GARCH already stored these on the percent scale; leave as is.
            pass
    if "squared_std_resid" not in fc.columns:
        fc["squared_std_resid"] = fc["std_resid"] ** 2
    return fc


# ---------------------------------------------------------------------------
# A1. Cross-model residual diagnostics (5 rows × 2 cols)
# ---------------------------------------------------------------------------

def fig_A1_residual_montage() -> Path:
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(11, 13), sharex=True)
    for i, (name, base) in enumerate(MODELS):
        fc = _load_test_forecast(base)
        plot_acf(fc["std_resid"].dropna(), lags=ACF_LAGS, ax=axes[i, 0])
        axes[i, 0].set_title(f"{name}: ACF of standardised residuals",
                             fontsize=10)
        plot_acf(fc["squared_std_resid"].dropna(), lags=ACF_LAGS, ax=axes[i, 1])
        axes[i, 1].set_title(f"{name}: ACF of squared standardised residuals",
                             fontsize=10)
        for ax in axes[i, :]:
            ax.set_ylim(-0.3, 0.3)
            ax.tick_params(labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Lag")
    fig.suptitle("Appendix Figure 1 — Cross-model residual diagnostics "
                 "(SPY daily test, no_exog)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = OUT_DIR / "A1_residual_montage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")
    return out


# ---------------------------------------------------------------------------
# A2. Per-regime QLIKE boxplot
# ---------------------------------------------------------------------------

def fig_A2_regime_boxplot() -> Path:
    rfp_files = {
        "GJR-GARCH":      ROOT / "ARMA-GARCH-model/outputs/rfp/garch_rfp_results.csv",
        "MS-GARCH":       ROOT / "MSGARCH-model/outputs/rfp/msgarch_rfp_results.csv",
        "LSTM-Attention": ROOT / "LSTM-Attention-model/outputs/rfp/lstm_attention_rfp_results.csv",
        "Transformer":    ROOT / "additional-models/outputs/transformer/rfp/transformer_rfp_results.csv",
        "XGBoost":        ROOT / "xgboost-model/outputs/rfp/xgboost_rfp_results.csv",
    }
    rows = []
    for name, p in rfp_files.items():
        df = pd.read_csv(p)
        df = df[(df["target"] == "SPY") & (df["freq"] == "daily") & (df["exog"] == EXOG)]
        for _, r in df.iterrows():
            rows.append({"model": name, "regime": r["regime"], "qlike": r["qlike"]})
    long_df = pd.DataFrame(rows)
    regime_order = ["CALM_17_19", "OIL_CRASH", "ENERGY_22", "GFC", "COVID"]
    model_order = [m for m, _ in MODELS]
    palette = ["#4c78a8", "#54a24b", "#f58518", "#b279a2", "#e45756"]

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.15
    positions = np.arange(len(regime_order))
    for i, m in enumerate(model_order):
        data_per_regime = [
            long_df[(long_df["model"] == m) & (long_df["regime"] == r)]["qlike"].values
            for r in regime_order
        ]
        bp = ax.boxplot(data_per_regime,
                        positions=positions + (i - 2) * width,
                        widths=width * 0.9,
                        patch_artist=True,
                        boxprops=dict(facecolor=palette[i], alpha=0.65,
                                       edgecolor="black", linewidth=0.7),
                        medianprops=dict(color="black", linewidth=1.0),
                        whiskerprops=dict(color="0.3", linewidth=0.7),
                        capprops=dict(color="0.3", linewidth=0.7),
                        flierprops=dict(marker="o", markersize=3,
                                         markerfacecolor=palette[i],
                                         markeredgecolor="0.3"))
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Calm 17-19", "Oil Crash", "Energy 22", "GFC", "COVID"])
    ax.set_ylabel("QLIKE per RFP window (log scale, lower is better)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i], alpha=0.65)
               for i in range(len(model_order))]
    ax.legend(handles, model_order, ncol=5, loc="upper left",
              fontsize=9, frameon=False)
    ax.set_title("Appendix Figure 2 — Distribution of per-window QLIKE by "
                 "regime and model (SPY daily, no_exog)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = OUT_DIR / "A2_regime_boxplot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")
    return out


# ---------------------------------------------------------------------------
# A3. Cross-model overlay on one COVID RFP window
# ---------------------------------------------------------------------------

def fig_A3_covid_overlay() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = ["#4c78a8", "#54a24b", "#f58518", "#b279a2", "#e45756"]
    realised_drawn = False
    for (name, base), color in zip(MODELS, palette):
        # Try main RFP folder, then transformer's nested path.
        candidates = [
            base / "rfp" / "forecasts" / f"SPY_daily_{EXOG}_{WINDOW_ID}.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            print(f"  skip {name}: no per-window file")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        if not realised_drawn:
            ax.plot(df["date"], np.sqrt(df["realized_var"]),
                    color="0.30", linewidth=1.6, label="Observed |return|")
            realised_drawn = True
        ax.plot(df["date"], df["pred_vol"], color=color,
                linewidth=1.4, label=name, alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("Appendix Figure 3 — Cross-model volatility forecasts on "
                 f"one COVID RFP window ({WINDOW_ID})",
                 fontsize=11, fontweight="bold")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = OUT_DIR / "A3_covid_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")
    return out


# ---------------------------------------------------------------------------
# A4. Returns distribution with Normal and Student-t overlays
# ---------------------------------------------------------------------------

def fig_A4_returns_distribution() -> Path:
    raw = pd.read_csv(ROOT / "data" / "daily" / "SPY_daily.csv",
                      skiprows=[1, 2], header=0).rename(columns={"Price": "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date")
    px = raw.set_index("Date")["Close"].astype(float)
    log_ret = (np.log(px).diff().dropna() * 100.0).values  # SPY log returns × 100

    # Fit Normal and Student-t on log returns.
    mu_n, sd_n = stats.norm.fit(log_ret)
    df_t, mu_t, sd_t = stats.t.fit(log_ret)
    print(f"  normal: mu={mu_n:.3f}, sd={sd_n:.3f}")
    print(f"  t:      df={df_t:.2f}, mu={mu_t:.3f}, sd={sd_t:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # Left: histogram + overlays (linear).
    ax = axes[0]
    bins = np.linspace(-8, 8, 121)
    ax.hist(log_ret, bins=bins, density=True, alpha=0.55, color="#4c78a8",
            edgecolor="white", linewidth=0.3, label="Empirical density")
    xs = np.linspace(-8, 8, 600)
    ax.plot(xs, stats.norm.pdf(xs, mu_n, sd_n), color="#e45756",
            linewidth=1.4, linestyle="--", label="Normal fit")
    ax.plot(xs, stats.t.pdf(xs, df_t, mu_t, sd_t), color="#000000",
            linewidth=1.4, label=f"Student-t fit (df={df_t:.1f})")
    ax.set_xlim(-8, 8)
    ax.set_xlabel("SPY daily log return (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_title("Empirical vs parametric densities", fontsize=10)
    # Right: log-y tail focus.
    ax = axes[1]
    ax.hist(log_ret, bins=bins, density=True, alpha=0.55, color="#4c78a8",
            edgecolor="white", linewidth=0.3)
    ax.plot(xs, stats.norm.pdf(xs, mu_n, sd_n), color="#e45756",
            linewidth=1.4, linestyle="--", label="Normal fit")
    ax.plot(xs, stats.t.pdf(xs, df_t, mu_t, sd_t), color="#000000",
            linewidth=1.4, label=f"Student-t fit (df={df_t:.1f})")
    ax.set_yscale("log")
    ax.set_xlim(-10, 10)
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("SPY daily log return (%)")
    ax.set_ylabel("Density (log)")
    ax.legend(fontsize=9)
    ax.set_title("Tail behaviour (log y-axis)", fontsize=10)

    fig.suptitle("Appendix Figure 4 — SPY daily log-return distribution with "
                 "Normal and Student-t fits",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "A4_returns_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    fig_A1_residual_montage()
    fig_A2_regime_boxplot()
    fig_A3_covid_overlay()
    fig_A4_returns_distribution()
