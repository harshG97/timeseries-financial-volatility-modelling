"""Patch v2 for volatility_modelling.ipynb:

A. Add MS-GARCH and Transformer disclaimers (markdown cells with reference
   results from the team's GPU/working-R run).
B. Move Section 11's RFP diagnostic plots into <model>/rfp/plots/diagnostics/
   so they do not collide with Section 13's test-block plots, which use the
   legacy <model>/plots/SPY/daily/<exog>/ paths.
C. Insert Section 13 "Test ECV": a single chunk-and-fit helper plus per-model
   run cells that produce <model>/<model>_test_results.csv,
   <model>/forecasts/SPY_daily_<exog>_test_forecasts.csv, and the four
   standard test-block diagnostic plots.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "volatility_modelling.ipynb"


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


def find_md_cell(cells: list, needle: str) -> int:
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and needle in "".join(c["source"]):
            return i
    raise LookupError(needle)


# ---------------------------------------------------------------------------
# new content
# ---------------------------------------------------------------------------

MSGARCH_DISCLAIMER_MD = """> **Reproducibility note — MS-GARCH (R / rpy2 setup):** The MS-GARCH
> section depends on the R package `MSGARCH` accessed through `rpy2`,
> which can be brittle to install (R version, `R_HOME`, library paths,
> compiler toolchain). Re-runs on a different R installation may differ
> from the team's reported numbers in the second decimal. For
> traceability, the RFP results from the team's reference run are:
>
> *Mean across all RFP windows (per exog):*
>
> | exog       | qlike  | rmse   |
> |------------|--------|--------|
> | no_exog    | 1.1857 | 4.0804 |
> | with_exog  | 1.1814 | 4.0810 |
>
> *Mean QLIKE / RMSE by regime:*
>
> | regime      | qlike no_exog | qlike with_exog | rmse no_exog | rmse with_exog |
> |-------------|---------------|-----------------|--------------|----------------|
> | CALM_17_19  | 0.4194        | 0.4066          | 2.6423       | 2.6428         |
> | COVID       | 1.8665        | 1.8318          | 8.6044       | 8.4499         |
> | ENERGY_22   | 1.3647        | 1.3722          | 2.4622       | 2.4643         |
> | GFC         | 2.1511        | 2.1625          | 7.2579       | 7.3532         |
> | OIL_CRASH   | 0.4351        | 0.4323          | 0.9213       | 0.9191         |
"""

TRANSFORMER_DISCLAIMER_MD = """> **Reproducibility note — Transformer (CPU vs GPU):** The Transformer
> training loop is sensitive to the floating-point reduction order on
> the device, so CPU and GPU runs with the *same seed* will not produce
> bit-identical numbers. The reported results in the Final Report were
> generated on GPU. For traceability, the GPU reference run gave:
>
> *Mean across all RFP windows (per exog):*
>
> | exog       | qlike  | rmse   |
> |------------|--------|--------|
> | no_exog    | 1.3227 | 4.2525 |
> | with_exog  | 1.2924 | 4.3171 |
>
> *Mean QLIKE / RMSE by regime:*
>
> | regime      | qlike no_exog | qlike with_exog | rmse no_exog | rmse with_exog |
> |-------------|---------------|-----------------|--------------|----------------|
> | CALM_17_19  | 0.4537        | 0.4213          | 2.7461       | 2.7204         |
> | COVID       | 2.0334        | 1.7775          | 8.9784       | 9.1121         |
> | ENERGY_22   | 1.3635        | 1.4620          | 2.4460       | 2.4389         |
> | GFC         | 2.6071        | 2.5301          | 7.7117       | 7.9107         |
> | OIL_CRASH   | 0.4483        | 0.4993          | 0.9094       | 0.9456         |
"""

TEST_ECV_HEADER_MD = """## 13. Test ECV (chronological one-step-ahead forecasting)

The RFP block in Sections 5–9 stress-tests each model on five 60-day
historical regimes. This section reproduces the **headline test-block
numbers used in the Final Report** by running an Expanding Cross-
Validation (ECV) over the chronological test split (2024-01 → 2026-04).

For each model and each exog variant we:

1. Initialise `history = train + val`.
2. Walk forward through the test set in chunks of `cadence` days. At
   the start of each chunk we *refit* the model on `history`, then
   produce one-step-ahead forecasts σ̂ₜ² for every step inside the
   chunk holding parameters fixed. After the chunk, we append its
   actuals to `history` and roll forward.
3. Concatenate forecasts across all chunks; score with the same
   `metrics()` helper used by the RFP block.

Refit cadence (matches the legacy per-model scripts):

| Model            | Cadence (days) | Reason |
|------------------|----------------|--------|
| GARCH            | 1              | QMLE fit is fast; refit every step. |
| MS-GARCH         | 20             | R/rpy2 fit is the slowest call in the loop. |
| LSTM-Attention   | 20             | Re-tunes the network from scratch each refit. |
| Transformer      | 20             | Same as LSTM. |
| XGBoost          | 20             | Boosted-tree refit is cheap but not free. |

Per-model outputs:

* `<model>/<model>_test_results.csv` — single-row metrics
  (RMSE / MAE / QLIKE / VaR exceptions) for each exog variant.
* `<model>/forecasts/SPY_daily_<exog>_test_forecasts.csv` — full
  date-indexed predictions.
* `<model>/plots/SPY/daily/<exog>/{volatility_forecast_timeseries,
  standardized_residuals, acf_standardized_residuals,
  acf_squared_standardized_residuals}.png` — the four diagnostic
  plots from the chronological test forecast.

Set `RUN_TEST_ECV = True` in the next cell to opt in. The full suite
takes a few minutes per model; the heaviest is the LSTM/Transformer
because they refit a fresh network every 20 days.
"""

TEST_ECV_HELPER_CODE = """RUN_TEST_ECV = False  # opt-in: this is the slowest section.


def _exception_counts(fc: pd.DataFrame) -> dict:
    \"\"\"Convert hit rates to raw counts for a finished test forecast.\"\"\"
    sigma = np.sqrt(fc["pred_var"].to_numpy())
    out = {}
    for level in VAR_LEVELS:
        z = NormalDist().inv_cdf(level)
        n_exc = int(np.sum(fc["ret_pct"].to_numpy() < z * sigma))
        out[f"var_{int(level * 100)}_exceptions"] = n_exc
    return out


def run_test_ecv(model_name: str, evaluate_fn, kwargs_per_exog: dict,
                 cadence: int) -> pd.DataFrame:
    \"\"\"Generic chronological ECV: chunk the test split into `cadence`-sized
    slices and call the per-window evaluator for each slice. The evaluator
    refits at the start of each slice (its own train_df is `history` up to
    that point) and produces fixed-parameter one-step-ahead forecasts within
    the slice.

    Writes <model>/<model>_test_results.csv (one row per exog) and
    <model>/forecasts/SPY_daily_<exog>_test_forecasts.csv.
    \"\"\"
    rows = []
    for exog in EXOG_VARIANTS:
        frames = load_cell(TARGET, FREQ, exog)
        history = pd.concat([frames["train"], frames["val"]],
                            ignore_index=True).sort_values("date").reset_index(drop=True)
        test = frames["test"].sort_values("date").reset_index(drop=True)
        kwargs = kwargs_per_exog[exog]
        chunks = []
        n_chunks = math.ceil(len(test) / cadence)
        for chunk_idx in tqdm(range(n_chunks),
                              desc=f"{model_name} test ECV {exog}", leave=False):
            cs = chunk_idx * cadence
            ce = min(cs + cadence, len(test))
            chunk = test.iloc[cs:ce].reset_index(drop=True)
            if chunk.empty:
                continue
            pseudo = RFPWindow(
                window_id=f"test_{cs:04d}", regime="TEST",
                target=TARGET, use_exog=(exog == "with_exog"),
                fit_end=history["date"].iloc[-1],
                forecast_start=chunk["date"].iloc[0],
                forecast_end=chunk["date"].iloc[-1],
                train=history.copy(), forecast=chunk,
            )
            try:
                _, fc = evaluate_fn(pseudo, **kwargs)
                chunks.append(fc)
            except Exception as exc:
                print(f"  {model_name}/{exog} chunk {cs}: {exc}")
            history = pd.concat([history, chunk], ignore_index=True)
        if not chunks:
            continue
        full_fc = pd.concat(chunks, ignore_index=True)
        full_fc["date"] = pd.to_datetime(full_fc["date"])
        full_fc = full_fc.sort_values("date").reset_index(drop=True)
        # Persist forecasts to disk.
        fc_dir = MODEL_DIRS[model_name] / "forecasts"
        fc_dir.mkdir(parents=True, exist_ok=True)
        fc_path = fc_dir / f"{TARGET}_{FREQ}_{exog}_test_forecasts.csv"
        full_fc.to_csv(fc_path, index=False)
        # Compute and store metrics.
        m = metrics(full_fc["realized_var"].to_numpy(),
                    full_fc["pred_var"].to_numpy(),
                    full_fc["ret_pct"].to_numpy())
        m.update(_exception_counts(full_fc))
        m.update({"target": TARGET, "freq": FREQ, "exog": exog,
                  "n_test": len(full_fc), "refit_cadence": cadence})
        rows.append(m)
    df = pd.DataFrame(rows)
    if not df.empty:
        out_path = MODEL_DIRS[model_name] / f"{model_name}_test_results.csv"
        df.to_csv(out_path, index=False)
        print(f"[{model_name}] saved -> {out_path.relative_to(ROOT)}")
    return df


def make_test_diagnostic_plots(model_name: str) -> None:
    \"\"\"Build the four standard diagnostic plots from the test_forecasts CSV
    written by run_test_ecv. Saves to <model>/plots/SPY/daily/<exog>/.\"\"\"
    base_dir = MODEL_DIRS[model_name] / "plots" / TARGET / FREQ
    for exog in EXOG_VARIANTS:
        fc_path = (MODEL_DIRS[model_name] / "forecasts"
                   / f"{TARGET}_{FREQ}_{exog}_test_forecasts.csv")
        if not fc_path.exists():
            continue
        fc = pd.read_csv(fc_path, parse_dates=["date"])
        plot_dir = base_dir / exog
        plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Forecast time series with historical |return| backdrop.
        base = SPLITS_DIR / FREQ / exog / TARGET
        train = pd.read_csv(base / "train.csv", parse_dates=["date"])
        val = pd.read_csv(base / "val.csv", parse_dates=["date"])
        history = pd.concat([train, val], ignore_index=True)
        history = history.assign(realized_vol=lambda x: np.abs(x["ret"]) * 100.0)

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(history["date"], history["realized_vol"], color="0.70",
                linewidth=0.8, label="Historical |return|")
        ax.plot(fc["date"], np.sqrt(fc["realized_var"]), color="#1f77b4",
                linewidth=1.1, label="Test observed vol")
        ax.plot(fc["date"], fc["pred_vol"], color="#d62728",
                linewidth=1.1, label="Test predicted vol")
        ax.set_title(f"{model_name} — {TARGET} {FREQ} {exog}: test ECV "
                     f"observed vs predicted volatility")
        ax.set_xlabel("Date"); ax.set_ylabel("Volatility (%)")
        ax.legend(); fig.tight_layout()
        fig.savefig(plot_dir / "volatility_forecast_timeseries.png", dpi=150)
        plt.close(fig)

        # 2. Standardized residuals.
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(fc["date"], fc["std_resid"], color="#4c78a8", linewidth=0.9)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{model_name} — {TARGET} {FREQ} {exog}: standardized residuals (test)")
        ax.set_xlabel("Date"); ax.set_ylabel("z_t = r_t / σ̂_t")
        fig.tight_layout()
        fig.savefig(plot_dir / "standardized_residuals.png", dpi=150)
        plt.close(fig)

        # 3 & 4. ACFs.
        max_lags = min(40, max(1, len(fc) // 4))
        for col, fname, title in [
            ("std_resid", "acf_standardized_residuals.png",
             "ACF of standardized residuals"),
            ("squared_std_resid", "acf_squared_standardized_residuals.png",
             "ACF of squared standardized residuals"),
        ]:
            fig, ax = plt.subplots(figsize=(12, 4))
            plot_acf(fc[col].dropna(), lags=max_lags, ax=ax)
            ax.set_title(f"{model_name} — {TARGET} {FREQ} {exog}: {title} (test)")
            fig.tight_layout()
            fig.savefig(plot_dir / fname, dpi=150)
            plt.close(fig)
        if SHOW_PLOTS:
            print(f"[{model_name}/{exog}] saved 4 test-block plots -> "
                  f"{plot_dir.relative_to(ROOT)}")
"""

TEST_ECV_GARCH_CODE = """if RUN_TEST_ECV:
    # Pull validated (p, o, q) per exog and run the GARCH ECV at cadence=1.
    garch_test_kwargs = {}
    for exog in EXOG_VARIANTS:
        row = garch_grid[(garch_grid["target"] == TARGET) &
                         (garch_grid["freq"] == FREQ) &
                         (garch_grid["exog"] == exog)].iloc[0]
        def _pick(r, *names):
            for n in names:
                if n in r and pd.notna(r[n]):
                    return int(r[n])
            raise KeyError(names)
        garch_test_kwargs[exog] = {
            "p": _pick(row, "p", "best_p"),
            "o": _pick(row, "o", "best_o"),
            "q": _pick(row, "q", "best_q"),
        }
    garch_test_df = run_test_ecv("garch", evaluate_garch_window,
                                  garch_test_kwargs, cadence=1)
    if not garch_test_df.empty:
        display(garch_test_df[["exog", "rmse", "mae", "qlike",
                               "var_1_exceptions", "var_5_exceptions"]])
        make_test_diagnostic_plots("garch")
"""

TEST_ECV_MSGARCH_CODE = """if RUN_TEST_ECV and MSGARCH_AVAILABLE and not msgarch_grid.empty:
    msgarch_test_kwargs = {}
    for exog in EXOG_VARIANTS:
        match = msgarch_grid[(msgarch_grid["target"] == TARGET) &
                             (msgarch_grid["freq"] == FREQ) &
                             (msgarch_grid["exog"] == exog)]
        if match.empty:
            continue
        row = match.iloc[0]
        msgarch_test_kwargs[exog] = {
            "k": int(row["k"]), "mtype": row["model"], "dist": row["dist"],
        }
    msgarch_test_df = run_test_ecv("msgarch", evaluate_msgarch_window,
                                    msgarch_test_kwargs, cadence=20)
    if not msgarch_test_df.empty:
        display(msgarch_test_df[["exog", "rmse", "mae", "qlike",
                                 "var_1_exceptions", "var_5_exceptions"]])
        make_test_diagnostic_plots("msgarch")
"""

TEST_ECV_LSTM_CODE = """if RUN_TEST_ECV:
    lstm_test_kwargs = {}
    for exog in EXOG_VARIANTS:
        row = lstm_grid[(lstm_grid["target"] == TARGET) &
                        (lstm_grid["freq"] == FREQ) &
                        (lstm_grid["exog"] == exog)].iloc[0]
        cfg = LSTMConfig(
            lookback=int(row["lookback"]),
            hidden_size=int(row["hidden_size"]),
            num_layers=int(row.get("num_layers", 1)),
            dropout=float(row.get("dropout", 0.0)),
            learning_rate=float(row.get("learning_rate", 1e-3)),
            weight_decay=float(row.get("weight_decay", 0.0)),
            batch_size=int(row.get("batch_size", 64)),
            epochs=int(row.get("epochs", 80)),
            patience=int(row.get("patience", 10)),
        )
        lstm_test_kwargs[exog] = {
            "model_cls": VolatilityLSTM, "config": cfg,
            "device": lstm_device, "seed": SEED,
        }
    lstm_test_df = run_test_ecv("lstm_attention", evaluate_seq_window,
                                 lstm_test_kwargs, cadence=20)
    if not lstm_test_df.empty:
        display(lstm_test_df[["exog", "rmse", "mae", "qlike",
                              "var_1_exceptions", "var_5_exceptions"]])
        make_test_diagnostic_plots("lstm_attention")
"""

TEST_ECV_TRANSFORMER_CODE = """if RUN_TEST_ECV:
    tx_test_kwargs = {}
    for exog in EXOG_VARIANTS:
        row = tx_grid[(tx_grid["target"] == TARGET) &
                      (tx_grid["freq"] == FREQ) &
                      (tx_grid["exog"] == exog)].iloc[0]
        cfg = TransformerConfig(
            lookback=int(row["lookback"]),
            d_model=int(row["d_model"]),
            nhead=int(row["nhead"]),
            num_layers=int(row.get("num_layers", 1)),
            dim_feedforward=int(row.get("dim_feedforward", 64)),
            dropout=float(row.get("dropout", 0.1)),
            learning_rate=float(row.get("learning_rate", 1e-3)),
            weight_decay=float(row.get("weight_decay", 0.0)),
            batch_size=int(row.get("batch_size", 64)),
            epochs=int(row.get("epochs", 30)),
            patience=int(row.get("patience", 5)),
        )
        tx_test_kwargs[exog] = {
            "model_cls": VolatilityTransformer, "config": cfg,
            "device": tx_device, "seed": SEED,
        }
    tx_test_df = run_test_ecv("transformer", evaluate_seq_window,
                               tx_test_kwargs, cadence=20)
    if not tx_test_df.empty:
        display(tx_test_df[["exog", "rmse", "mae", "qlike",
                            "var_1_exceptions", "var_5_exceptions"]])
        make_test_diagnostic_plots("transformer")
"""

TEST_ECV_XGB_CODE = """if RUN_TEST_ECV:
    xgb_test_kwargs = {}
    for exog in EXOG_VARIANTS:
        row = xgb_grid[(xgb_grid["target"] == TARGET) &
                       (xgb_grid["freq"] == FREQ) &
                       (xgb_grid["exog"] == exog)].iloc[0]
        cfg = XGBConfig(
            max_depth=int(row["max_depth"]),
            learning_rate=float(row["learning_rate"]),
            n_estimators=int(row["n_estimators"]),
            subsample=float(row.get("subsample", 0.8)),
            colsample_bytree=float(row.get("colsample_bytree", 0.8)),
            min_child_weight=float(row.get("min_child_weight", 1.0)),
            reg_lambda=float(row.get("reg_lambda", 1.0)),
        )
        xgb_test_kwargs[exog] = {"cfg": cfg, "seed": SEED}
    xgb_test_df = run_test_ecv("xgboost", evaluate_xgb_window,
                                xgb_test_kwargs, cadence=20)
    if not xgb_test_df.empty:
        display(xgb_test_df[["exog", "rmse", "mae", "qlike",
                             "var_1_exceptions", "var_5_exceptions"]])
        make_test_diagnostic_plots("xgboost")
"""


def patch():
    nb = json.loads(NB.read_text())
    cells = nb["cells"]

    # ---- A. Move Section 11 RFP diagnostic plots to a separate path -------
    # so they do not collide with Section 13's test-block plots.
    section11_idx = find_md_cell(cells, "## 11. Diagnostic Plots")
    helper_idx = section11_idx + 1
    helper_cell = cells[helper_idx]
    if not replace_in_cell(helper_cell,
        'base_dir = MODEL_DIRS[model_name] / "plots" / TARGET / FREQ',
        'base_dir = (MODEL_DIRS[model_name] / "rfp" / "plots"\n'
        '                / "diagnostics" / TARGET / FREQ)'):
        raise RuntimeError("Section 11 helper not in expected shape.")
    # Update the section 11 markdown explanation to reflect the new path.
    cells[section11_idx]["source"] = "".join(cells[section11_idx]["source"]).replace(
        "saved\nunder `<model>/plots/SPY/daily/<exog>/`",
        "saved\nunder `<model>/rfp/plots/diagnostics/SPY/daily/<exog>/`\n"
        "(Section 13 reuses the legacy `<model>/plots/SPY/daily/<exog>/`\n"
        "path for the chronological test-ECV plots, so the two sections do\n"
        "not overwrite each other)",
    ).splitlines(keepends=True)

    # ---- B. Insert disclaimer markdown cells right after each model header
    # We insert *after* the section markdown cell so the warning appears at
    # the top of the section and before any code cell.
    msg_idx = find_md_cell(cells, "## 6. MS-GARCH (Markov-Switching GARCH)")
    cells.insert(msg_idx + 1, md(MSGARCH_DISCLAIMER_MD))

    tx_idx = find_md_cell(cells, "## 8. Transformer")
    cells.insert(tx_idx + 1, md(TRANSFORMER_DISCLAIMER_MD))

    # ---- C. Append Section 13 (after Section 12 cells) --------------------
    # Find Section 12 header to append after the last code cell of Section 12.
    sec12_idx = find_md_cell(cells, "## 12. RFP Comparison Plots")
    # Walk forward to find the trailing empty code cell (or end of notebook)
    insert_at = len(cells)
    for j in range(sec12_idx + 1, len(cells)):
        if cells[j]["cell_type"] == "code" and not "".join(cells[j]["source"]).strip():
            insert_at = j
            break

    new_cells = [
        md(TEST_ECV_HEADER_MD),
        code(TEST_ECV_HELPER_CODE),
        md("### 13.1 GARCH test ECV (refit cadence = 1 day)\n"),
        code(TEST_ECV_GARCH_CODE),
        md("### 13.2 MS-GARCH test ECV (refit cadence = 20 days)\n"),
        code(TEST_ECV_MSGARCH_CODE),
        md("### 13.3 LSTM-Attention test ECV (refit cadence = 20 days)\n"),
        code(TEST_ECV_LSTM_CODE),
        md("### 13.4 Transformer test ECV (refit cadence = 20 days)\n"),
        code(TEST_ECV_TRANSFORMER_CODE),
        md("### 13.5 XGBoost test ECV (refit cadence = 20 days)\n"),
        code(TEST_ECV_XGB_CODE),
    ]
    cells[insert_at:insert_at] = new_cells

    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1))
    print(f"Patched: {NB}  cells={len(cells)}")


if __name__ == "__main__":
    patch()
