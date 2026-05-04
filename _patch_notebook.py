"""Patch volatility_modelling.ipynb:

1. Fix MS-GARCH tune-vs-load mismatch: tune now writes msgarch_validation_results.csv
   (the same file the default load reads), matching the user's brief that says
   "by default use the best parameters saved for each model in the
   [model name]_validation_results.csv files."
2. Same fix for GARCH (was loading garch_grid_search_results.csv).
3. Insert section 11 "Diagnostic Plots" with reusable plotting helpers.
4. Insert section 12 "RFP Comparison Plots" (ablation + per-window).
5. Add brief markdown explanations and helpful code comments.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "volatility_modelling.ipynb"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def replace_in_cell(cell: dict, old: str, new: str) -> bool:
    src = "".join(cell["source"])
    if old not in src:
        return False
    cell["source"] = src.replace(old, new).splitlines(keepends=True)
    return True


# ---------------------------------------------------------------------------
# new content
# ---------------------------------------------------------------------------

PLOTS_HEADER_MD = """## 11. Diagnostic Plots

For every model × exog variant we produce four standard diagnostic plots
from the RFP forecast set. Forecasts from all RFP windows are concatenated
into a single chronological series before plotting. Each plot is saved
under `<model>/plots/SPY/daily/<exog>/`:

* `volatility_forecast_timeseries.png` — predicted σ̂ₜ overlaid on the
  observed |rₜ|, gives the visual "does the forecast move with the truth"
  check.
* `standardized_residuals.png` — zₜ = rₜ / σ̂ₜ, should look like white
  noise with unit variance if the variance specification is adequate.
* `acf_standardized_residuals.png` — should have no significant lags;
  significant autocorrelation here means the *mean* equation is mis-
  specified.
* `acf_squared_standardized_residuals.png` — should have no significant
  lags; significant autocorrelation here means the *variance* equation
  has not absorbed all the heteroskedasticity.

The same helper is used across all five models for a 1-to-1 visual
comparison.
"""

PLOTS_HELPER_CODE = """from statsmodels.graphics.tsaplots import plot_acf


def _concat_forecasts(forecasts: dict[str, pd.DataFrame], exog: str) -> pd.DataFrame:
    \"\"\"Concatenate the RFP forecast frames for one exog variant, sorted by date.\"\"\"
    suffix = f"_{exog}_"
    parts = [df for key, df in forecasts.items() if suffix in key]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def make_diagnostic_plots(model_name: str, forecasts: dict[str, pd.DataFrame]) -> None:
    \"\"\"Save the standard 4-plot diagnostic suite per exog variant.\"\"\"
    if not forecasts:
        print(f"[{model_name}] no forecasts to plot")
        return
    base_dir = MODEL_DIRS[model_name] / "plots" / TARGET / FREQ
    for exog in EXOG_VARIANTS:
        fc = _concat_forecasts(forecasts, exog)
        if fc.empty:
            continue
        plot_dir = base_dir / exog
        plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Volatility forecast time series.
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(fc["date"], np.sqrt(fc["realized_var"]), color="#1f77b4",
                linewidth=1.0, label="Observed |return| (RFP windows)")
        ax.plot(fc["date"], fc["pred_vol"], color="#d62728",
                linewidth=1.0, label="Predicted volatility")
        ax.set_title(f"{model_name} — {TARGET} {FREQ} {exog}: observed vs predicted volatility")
        ax.set_xlabel("Date"); ax.set_ylabel("Volatility (%)")
        ax.legend(); fig.tight_layout()
        fig.savefig(plot_dir / "volatility_forecast_timeseries.png", dpi=150)
        plt.close(fig)

        # 2. Standardized residuals zₜ = rₜ / σ̂ₜ.
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(fc["date"], fc["std_resid"], color="#4c78a8", linewidth=0.8)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{model_name} — {TARGET} {FREQ} {exog}: standardized residuals")
        ax.set_xlabel("Date"); ax.set_ylabel("z_t = r_t / σ̂_t")
        fig.tight_layout()
        fig.savefig(plot_dir / "standardized_residuals.png", dpi=150)
        plt.close(fig)

        # 3 & 4. ACFs of zₜ and zₜ².
        max_lags = min(40, max(1, len(fc) // 4))
        for col, fname, title in [
            ("std_resid", "acf_standardized_residuals.png",
             "ACF of standardized residuals"),
            ("squared_std_resid", "acf_squared_standardized_residuals.png",
             "ACF of squared standardized residuals"),
        ]:
            fig, ax = plt.subplots(figsize=(12, 4))
            plot_acf(fc[col].dropna(), lags=max_lags, ax=ax)
            ax.set_title(f"{model_name} — {TARGET} {FREQ} {exog}: {title}")
            fig.tight_layout()
            fig.savefig(plot_dir / fname, dpi=150)
            plt.close(fig)

        if SHOW_PLOTS:
            print(f"[{model_name}/{exog}] saved 4 diagnostic plots -> "
                  f"{plot_dir.relative_to(ROOT)}")
"""

PLOTS_RUN_CODE = """# Generate diagnostic plots for every model that produced forecasts.
_diag_inputs = [
    ("garch", garch_forecasts),
    ("msgarch", locals().get("msgarch_forecasts", {})),
    ("lstm_attention", lstm_forecasts),
    ("transformer", tx_forecasts),
    ("xgboost", xgb_forecasts),
]
for _name, _fcs in _diag_inputs:
    make_diagnostic_plots(_name, _fcs)
"""

RFP_PLOTS_HEADER_MD = """## 12. RFP Comparison Plots

Two cross-window plot families are produced per model and saved under
`<model>/rfp/plots/`:

* `ablation/SPY_daily_exog_ablation.png` — bar chart of mean QLIKE per
  RFP regime, comparing `no_exog` against `with_exog`. This is the
  cleanest visual answer to the question *"does adding the VIX/cross-
  asset features actually help during this regime?"*
* `per_window/SPY/daily/<exog>/<window_id>.png` — for each individual
  RFP window, observed vs predicted volatility over the 60-day forecast
  horizon. Useful for spotting which specific windows are hard.
"""

RFP_PLOTS_HELPER_CODE = """def make_ablation_plot(model_name: str, results_df: pd.DataFrame) -> None:
    \"\"\"Bar chart: mean QLIKE per regime, no_exog vs with_exog.\"\"\"
    if results_df.empty or results_df["exog"].nunique() < 2:
        return
    plot_dir = MODEL_DIRS[model_name] / "rfp" / "plots" / "ablation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pivot = (results_df.pivot_table(index="regime", columns="exog",
                                     values="qlike", aggfunc="mean")
             .sort_index())
    x = np.arange(len(pivot.index)); width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, pivot.get("no_exog"), width,
           label="no_exog", color="#4c78a8")
    ax.bar(x + width / 2, pivot.get("with_exog"), width,
           label="with_exog", color="#f58518")
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_title(f"{model_name} {TARGET} {FREQ} — Exogenous Features Ablation by Regime",
                 fontweight="bold")
    ax.set_ylabel("Mean QLIKE (lower is better)")
    ax.legend(); fig.tight_layout()
    fname = f"{TARGET}_{FREQ}_exog_ablation.png"
    fig.savefig(plot_dir / fname, dpi=150)
    plt.close(fig)
    if SHOW_PLOTS:
        print(f"[{model_name}] ablation plot -> "
              f"{(plot_dir / fname).relative_to(ROOT)}")


def make_per_window_plots(model_name: str, forecasts: dict[str, pd.DataFrame],
                          results_df: pd.DataFrame) -> None:
    \"\"\"One observed-vs-predicted plot per RFP window, grouped by exog.\"\"\"
    if not forecasts:
        return
    base = MODEL_DIRS[model_name] / "rfp" / "plots" / "per_window" / TARGET / FREQ
    # Build a (window_id, exog) -> regime lookup so titles can name the regime.
    if results_df.empty:
        regime_lookup = {}
    else:
        regime_lookup = {
            (r["window_id"], r["exog"]): r["regime"]
            for r in results_df.to_dict("records")
        }
    n = 0
    for key, fc in forecasts.items():
        # Forecast keys are formatted as f"{TARGET}_{FREQ}_{exog}_{window_id}".
        prefix = f"{TARGET}_{FREQ}_"
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        # rest is "<exog>_<window_id>"; exog is one of EXOG_VARIANTS.
        exog = next((e for e in EXOG_VARIANTS if rest.startswith(e + "_")), None)
        if exog is None:
            continue
        window_id = rest[len(exog) + 1:]
        regime = regime_lookup.get((window_id, exog), "")
        plot_dir = base / exog
        plot_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.to_datetime(fc["date"])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, np.sqrt(fc["realized_var"]), color="#1f77b4",
                linewidth=1.2, label="Observed vol")
        ax.plot(dates, fc["pred_vol"], color="#d62728",
                linewidth=1.2, label="Predicted vol")
        title = f"{model_name} — {TARGET} {FREQ} {exog} — {window_id}"
        if regime:
            title += f" ({regime})"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("Volatility (%)")
        ax.legend(fontsize=9); fig.tight_layout()
        fig.savefig(plot_dir / f"{window_id}.png", dpi=150)
        plt.close(fig)
        n += 1
    if SHOW_PLOTS:
        print(f"[{model_name}] saved {n} per-window plots -> "
              f"{base.relative_to(ROOT)}")
"""

RFP_PLOTS_RUN_CODE = """# Generate ablation + per-window plots for every model.
_rfp_inputs = [
    ("garch", garch_results_df, garch_forecasts),
    ("msgarch", msgarch_results_df, locals().get("msgarch_forecasts", {})),
    ("lstm_attention", lstm_results_df, lstm_forecasts),
    ("transformer", tx_results_df, tx_forecasts),
    ("xgboost", xgb_results_df, xgb_forecasts),
]
for _name, _res, _fcs in _rfp_inputs:
    make_ablation_plot(_name, _res)
    make_per_window_plots(_name, _fcs, _res)
"""


# ---------------------------------------------------------------------------
# main patch
# ---------------------------------------------------------------------------

def patch():
    nb = json.loads(NB.read_text())
    cells = nb["cells"]

    # ---- Fix 1: GARCH default load should pull validation_results CSV ------
    # The val CSV has the columns p, o, q (and target/freq/exog) needed
    # downstream. We rename "best_p" / "best_o" / "best_q" usage in cell 17 too.
    garch_cell = cells[16]
    if not replace_in_cell(garch_cell,
        'garch_grid = load_best_params("garch", GARCH_GRID_CSV)',
        'garch_grid = load_best_params("garch", GARCH_VAL_CSV)'):
        raise RuntimeError("garch load line not found — notebook structure changed?")

    # And the tune branch should also save to the val csv so the fallback is
    # consistent across runs.
    replace_in_cell(garch_cell,
        'df.to_csv(MODEL_DIRS["garch"] / GARCH_GRID_CSV, index=False)\n'
        '    return df\n',
        'df.to_csv(MODEL_DIRS["garch"] / GARCH_GRID_CSV, index=False)\n'
        '    # Also persist as the validation-results CSV that the default-load\n'
        '    # branch reads, with the column names downstream code expects.\n'
        '    val_df = df.rename(columns={"best_p": "p", "best_o": "o",\n'
        '                                 "best_q": "q"})\n'
        '    val_df.to_csv(MODEL_DIRS["garch"] / GARCH_VAL_CSV, index=False)\n'
        '    return df\n')

    # cell 17 reads "best_p" / "best_o" / "best_q" — but the val CSV has p/o/q.
    # Make the read tolerant to either schema.
    cell17 = cells[17]
    replace_in_cell(cell17,
        'p, o, q = int(row["best_p"]), int(row["best_o"]), int(row["best_q"])',
        '# Tolerate both column schemas: grid_search CSV uses best_p/o/q,\n'
        '    # validation_results CSV uses p/o/q.\n'
        '    def _pick(r, *names):\n'
        '        for n in names:\n'
        '            if n in r and pd.notna(r[n]):\n'
        '                return int(r[n])\n'
        '        raise KeyError(names)\n'
        '    p = _pick(row, "p", "best_p")\n'
        '    o = _pick(row, "o", "best_o")\n'
        '    q = _pick(row, "q", "best_q")')

    # ---- Fix 2: MS-GARCH tune branch should also write to the val CSV ------
    msgarch_cell = cells[21]
    replace_in_cell(msgarch_cell,
        'df.to_csv(MODEL_DIRS["msgarch"] / MSGARCH_GRID_CSV, index=False)\n'
        '    return df\n',
        'df.to_csv(MODEL_DIRS["msgarch"] / MSGARCH_GRID_CSV, index=False)\n'
        '    # Mirror to the validation_results CSV (the default-load path)\n'
        '    # so re-runs without tuning find the freshly tuned params.\n'
        '    df.to_csv(MODEL_DIRS["msgarch"] / MSGARCH_VAL_CSV, index=False)\n'
        '    return df\n')

    # ---- Insert plotting sections at the end of the notebook ---------------
    # Find the index of the trailing empty cell (cell 39 is empty in the
    # current notebook). Insert before it if present, otherwise append.
    trailing = len(cells)
    if cells and cells[-1]["cell_type"] == "code" and not "".join(cells[-1]["source"]).strip():
        trailing = len(cells) - 1

    new_cells = [
        md(PLOTS_HEADER_MD),
        code(PLOTS_HELPER_CODE),
        code(PLOTS_RUN_CODE),
        md(RFP_PLOTS_HEADER_MD),
        code(RFP_PLOTS_HELPER_CODE),
        code(RFP_PLOTS_RUN_CODE),
    ]
    cells[trailing:trailing] = new_cells

    # ---- Tighten cell 1 markdown (close the ```yaml fence) -----------------
    env_cell = cells[1]
    src = "".join(env_cell["source"])
    if "```yaml" in src and src.count("```") < 2:
        env_cell["source"] = (src.rstrip() + "\n```\n").splitlines(keepends=True)

    # ---- Add inline comments to a few key cells ----------------------------
    # Cell 4 (paths): clarify that MODEL_DIRS keys are the names used by
    # load_best_params and save_rfp_artifacts elsewhere.
    cells[4]["source"] = (
        "# Output directories. Keys here must match the keys used by\n"
        "# load_best_params() and save_rfp_artifacts() further down.\n"
        + "".join(cells[4]["source"])
    ).splitlines(keepends=True)

    # Cell 6 (imports): add a one-line comment on ENDO_COLS purpose.
    cells[6]["source"] = "".join(cells[6]["source"]).replace(
        "VAR_LEVELS = (0.01, 0.05)\n",
        "VAR_LEVELS = (0.01, 0.05)\n"
        "# ENDO_COLS lists every column that is *not* an exogenous regressor —\n"
        "# used by exog_columns(df) to identify true exogenous features.\n"
    ).splitlines(keepends=True)

    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1))
    print(f"Patched: {NB}  cells={len(cells)}")


if __name__ == "__main__":
    patch()
