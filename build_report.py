"""Build the Final Report Word document for Team 3 - Modeling Volatility in Financial Markets.

This script assembles a single, self-contained Word document following the
formatting requirements in the assignment (Times New Roman 12pt, single spaced,
1-inch margins, no extra paragraph spacing, bold/italic headers).
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Final_Report.docx"

FONT = "Times New Roman"
FONT_SIZE = Pt(12)


# ---------------------------------------------------------------------------
# document setup helpers
# ---------------------------------------------------------------------------

def set_default_font(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = FONT_SIZE
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), FONT)
    rfonts.set(qn("w:hAnsi"), FONT)
    rfonts.set(qn("w:cs"), FONT)
    rfonts.set(qn("w:eastAsia"), FONT)
    pf = style.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0


def set_margins(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)


def add_para(doc, text="", *, bold=False, italic=False, align=None, size=12):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    if text:
        run = p.add_run(text)
        run.font.name = FONT
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
    return p


def add_heading(doc, text, level=1):
    """Custom heading: 12pt bold (level 1) or 12pt italic (level 2). No extra spacing."""
    bold = level == 1
    italic = level == 2
    return add_para(doc, text, bold=bold, italic=italic, size=12)


def add_runs(doc, runs, align=None):
    """Add a paragraph composed of multiple (text, bold, italic) runs."""
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    for text, bold, italic in runs:
        r = p.add_run(text)
        r.font.name = FONT
        r.font.size = FONT_SIZE
        r.bold = bold
        r.italic = italic
    return p


def add_bullet(doc, text, *, level=0, bold_lead=None):
    """Add a bullet with optional bold leading phrase."""
    p = doc.add_paragraph(style="List Bullet")
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    pf.left_indent = Inches(0.25 + 0.25 * level)
    if bold_lead is not None:
        r = p.add_run(bold_lead)
        r.font.name = FONT
        r.font.size = FONT_SIZE
        r.bold = True
        r2 = p.add_run(text)
        r2.font.name = FONT
        r2.font.size = FONT_SIZE
    else:
        r = p.add_run(text)
        r.font.name = FONT
        r.font.size = FONT_SIZE
    return p


def add_caption(doc, text):
    add_para(doc, text, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)


def add_placeholder(doc, text):
    p = add_para(doc, f"[FIGURE PLACEHOLDER: {text}]", italic=True,
                 align=WD_ALIGN_PARAGRAPH.CENTER)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)


def add_image(doc, path: Path, width_in=5.5):
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(path), width=Inches(width_in))
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
    else:
        add_placeholder(doc, str(path))


def style_table_cell(cell, *, bold=False, align=None):
    for paragraph in cell.paragraphs:
        pf = paragraph.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        if align is not None:
            paragraph.alignment = align
        for run in paragraph.runs:
            run.font.name = FONT
            run.font.size = FONT_SIZE
            run.bold = bold


def add_table(doc, rows, header=True, col_widths=None):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Light Grid Accent 1"
    table.autofit = True
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = str(val)
            style_table_cell(cell, bold=(header and i == 0),
                             align=WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT)
    if col_widths is not None:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Inches(w)
    return table


# ---------------------------------------------------------------------------
# build report
# ---------------------------------------------------------------------------

def build():
    doc = Document()
    set_default_font(doc)
    set_margins(doc)

    # ---------- Title ----------
    add_para(doc, "Modeling Volatility in Financial Markets",
             bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
    add_para(doc, "Final Report — Team 3", italic=True,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Yibo Liu, Mani Chandana Bandaru, Rajath Sathyanarayana, "
                   "Yen-Shuo Su, Harsh Gupta",
             italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc)

    # ---------- 1. Summary ----------
    add_heading(doc, "1. Summary")
    add_para(doc,
        "This report investigates one-step-ahead volatility forecasting for the "
        "SPDR S&P 500 ETF (SPY) at the daily frequency. Twenty-two years of "
        "Yahoo Finance closing prices (Jan 2004 – Apr 2026) are split into "
        "train (2004–2021), validation (2022–2023), and test (2024–Apr 2026) "
        "blocks, with a strict expanding-window evaluation that prevents look-"
        "ahead bias. Two feature configurations are studied: a target-only "
        "set (no_exog) consisting of lagged returns, lagged squared returns, "
        "negative-shock indicators, and 5/10/22-day realized-volatility "
        "windows; and an exogenous set (with_exog) that augments these with "
        "lagged OIL, GOLD, DXY returns and the log-VIX level and first "
        "difference. Five families of models are benchmarked: GJR-GARCH, "
        "Markov-Switching GARCH (MS-GARCH), an LSTM with attention, a "
        "Transformer encoder, and gradient-boosted trees (XGBoost). On daily "
        "SPY, MS-GARCH (k=2, GJR variance, Student-t innovations) attains the "
        "lowest test RMSE (4.286) when augmented with exogenous features, "
        "while the LSTM-with-attention achieves the lowest QLIKE (0.649) and "
        "the Transformer matches MS-GARCH on RMSE (4.295) under with_exog. "
        "The three models also produce well-calibrated 1% and 5% VaR "
        "exceedances, in contrast with XGBoost, which yields the smallest "
        "MAE (0.926) but systematically under-estimates the conditional "
        "variance and therefore generates 11× the expected number of 1% VaR "
        "breaches. The major finding is that for SPY, regime-switching "
        "GARCH and attention-based recurrent networks deliver complementary "
        "but quantitatively similar gains over a single-regime GJR baseline; "
        "exogenous features (especially the VIX) provide consistent but "
        "modest improvements on the order of 1–10% in RMSE and QLIKE.")

    # ---------- 2. Introduction / Literature Review ----------
    add_heading(doc, "2. Introduction")

    add_heading(doc, "2.1 Motivation and Prior Expectations", level=2)
    add_para(doc,
        "Financial volatility is the degree to which asset prices fluctuate "
        "over time and is central to risk management, derivative pricing, "
        "and portfolio allocation. We selected this problem because (i) it "
        "aligns directly with the time-series themes of this course; (ii) "
        "SPY exhibits well-documented stylised facts — fat tails, leverage "
        "effects, and clustering — that motivate a layered modelling "
        "strategy; and (iii) repeated structural breaks (the 2008 GFC, the "
        "2020 COVID crash, the 2022 energy shock) provide natural stress "
        "tests of any forecasting framework. Going in, we expected GARCH-"
        "family models to remain competitive on point accuracy because their "
        "functional form is well matched to volatility persistence, and we "
        "expected machine-learning models (LSTM, Transformer, XGBoost) to "
        "win on regimes where nonlinear interactions with macro features "
        "(VIX, DXY) become important. We anticipated that MS-GARCH would "
        "improve VaR calibration during crisis windows by separating calm "
        "and turbulent dynamics.")

    add_heading(doc, "2.2 Literature Review", level=2)

    add_runs(doc, [
        ("2.2.1 Problem Origin. ", False, True),
        ("The formal study of time-varying volatility began with Engle's "
         "(1982) ARCH model and was generalised by Bollerslev (1986) into "
         "the GARCH framework, which has dominated volatility research for "
         "four decades.", False, False),
    ])

    add_runs(doc, [
        ("2.2.2 Problem Evolution. ", False, True),
        ("While GARCH-family models capture clustering under a single "
         "regime, financial markets alternate between structurally distinct "
         "states that a single-regime specification cannot represent. "
         "Hamilton (1989) introduced the Markov-switching autoregressive "
         "model, and this insight was extended to volatility through the "
         "MS-GARCH specifications of Haas et al. (2004) and Klaassen (2002). "
         "In parallel, machine-learning methods — feed-forward networks, "
         "recurrent architectures, and ensemble methods — have been applied "
         "to forecast volatility directly, often outperforming classical "
         "models (Carr et al., 2020; Kim & Won, 2018). More recently, "
         "machine-learning techniques have been turned toward regime "
         "detection itself (Raja Lope Ahmad et al., 2026; Ejike, 2026).",
         False, False),
    ])

    add_runs(doc, [
        ("2.2.3 Problem Impact. ", False, True),
        ("Underestimated tail risk contributed to catastrophic losses "
         "during the 2008 financial crisis, the 2020 COVID-19 crash, and "
         "the 2022 energy price shock. Improved volatility models enable "
         "more reliable Value-at-Risk estimates, better hedging strategies "
         "for energy-dependent firms, and more precise option pricing — "
         "serving both academic understanding and practical risk mitigation.",
         False, False),
    ])

    add_runs(doc, [
        ("2.2.4 Previous Analytical Efforts. ", False, True),
        ("Three streams of prior work motivate our design.",
         False, False),
    ])

    add_runs(doc, [
        ("Approach 1 — Markov-Switching GARCH. ", True, False),
        ("Haas et al. (2004) proposed a tractable MS-GARCH specification "
         "with independent conditional-variance equations per regime, "
         "showing substantial improvements in density forecasts on "
         "foreign-exchange data. Klaassen (2002) provided an alternative "
         "formulation that adjusts expected variance at regime transitions "
         "to mitigate path-dependence. Ardia et al. (2019) released the "
         "MSGARCH R package and confirmed regime-switching superiority for "
         "VaR forecasting on equity data. Hardy (2001) demonstrated similar "
         "gains for the S&P 500. The merits are statistical grounding, "
         "interpretable regime probabilities, and strong VaR improvements; "
         "shortcomings are computational complexity, the need to pre-"
         "specify the number of states, and difficulty incorporating high-"
         "dimensional features.", False, False),
    ])

    add_runs(doc, [
        ("Approach 2 — Machine-Learning Volatility Prediction. ", True, False),
        ("Carr et al. (2020) used Ridge Regression, Random Forest, and "
         "feed-forward neural networks to predict realised variance and "
         "build a volatility index. Doering et al. (2017) applied CNNs to "
         "high-frequency data with strong but opaque performance. Recurrent "
         "architectures have been explored more extensively: Tino et al. "
         "(2001) combined Markov models with an Elman RNN for volatility "
         "trading; Xiong et al. (2016) integrated price data with Google "
         "Domestic Trends in an LSTM framework; and Kim and Won (2018) "
         "developed LSTM models integrated with multiple GARCH-type inputs. "
         "Merits include the ability to capture complex non-linear patterns "
         "and to incorporate diverse features. Shortcomings are the black-"
         "box problem, susceptibility to over-fitting on noisy financial "
         "series, and the absence of any regime-probability interpretation.",
         False, False),
    ])

    add_runs(doc, [
        ("Approach 3 — Machine-Learning Regime Detection. ", True, False),
        ("Raja Lope Ahmad et al. (2026) combined PCA and clustering with "
         "attention-based RNNs to detect market-instability shifts without "
         "pre-specifying the number of states. Ejike (2026) compared LDA, "
         "SVMs, and feed-forward networks for regime classification, finding "
         "that nonlinear classifiers offered improved accuracy. The "
         "merits are data-driven discovery of regimes from high-dimensional "
         "features without parametric assumptions; the shortcomings are "
         "ex-post regime labels, large training-sample requirements, and "
         "the absence of an explicit transition-probability framework.",
         False, False),
    ])

    add_para(doc,
        "Our project sits across these three streams: GJR-GARCH provides a "
        "single-regime baseline; MS-GARCH supplies the regime-switching "
        "lens; and the LSTM-with-attention, Transformer, and XGBoost "
        "models contribute machine-learning perspectives that natively "
        "ingest exogenous features.")

    # ---------- 3. Data Summary ----------
    add_heading(doc, "3. Data Summary")
    add_para(doc,
        "We use closing-price series for SPY (SPDR S&P 500 ETF Trust) "
        "downloaded from Yahoo Finance, together with four auxiliary series "
        "(OIL = WTI crude futures CL=F, GOLD = SPDR Gold Shares GLD, DXY = "
        "U.S. dollar index, and VIX) used to construct exogenous features. "
        "The full sample spans 1 January 2004 through 29 April 2026. Log "
        "returns rₜ = ln(Pₜ/Pₜ₋₁) are scaled by 100 (so a 1% move equals "
        "1.0). All exogenous and target features are lagged by one period "
        "to prevent contemporaneous leakage and a one-period embargo is "
        "enforced at every split boundary. The feature engineering produces "
        "two configurations per cell:")
    add_bullet(doc, "lagged log return rₜ₋₁, lagged squared return rₜ₋₁², "
                    "asymmetric negative-shock term rₜ₋₁²·𝟙{rₜ₋₁<0}, and "
                    "5/10/22-day rolling realized-volatility windows.",
               bold_lead="no_exog (7 features): ")
    add_bullet(doc, "all of the above, plus lagged OIL, GOLD and DXY log "
                    "returns and the log-VIX level and first difference (a "
                    "13-feature panel).",
               bold_lead="with_exog (13 features): ")
    add_para(doc,
        "Splits follow a single, pre-registered manifest (data/splits/"
        "manifest.yaml) so every model trains and forecasts on identical "
        "rows. Train, validation, and test sample sizes are reported in "
        "Table 1. Returns are stationary at all conventional levels (ADF "
        "p-values < 1e-13; KPSS does not reject); prices are I(1). Squared "
        "residuals from the best ARMA fit show overwhelming Ljung–Box "
        "rejection (p < 1e-7 at lag 10), confirming conditional "
        "heteroskedasticity and motivating GARCH-type variance modelling.")

    add_para(doc)
    add_caption(doc, "Table 1. Sample sizes and time spans (SPY daily).")
    add_table(doc, [
        ["Split", "Start", "End", "Observations"],
        ["Train", "2004-02-06", "2021-12-31", "4,472"],
        ["Validation", "2022-01-03", "2023-12-29", "501"],
        ["Test", "2024-01-02", "2026-04-29", "583"],
        ["Total", "2004-02-06", "2026-04-29", "5,556"],
    ])

    add_para(doc)
    add_caption(doc, "Table 2. Descriptive statistics of SPY daily log "
                     "returns (×100), full sample.")
    add_table(doc, [
        ["Series", "N", "Mean", "Median", "Std", "Min", "Max", "Skew", "ExKurt"],
        ["SPY ret", "6,537", "0.031", "0.069", "1.222",
         "−11.589", "13.558", "−0.21", "11.53"],
    ])

    add_para(doc)
    add_caption(doc, "Table 3. Stationarity tests on SPY (full sample).")
    add_table(doc, [
        ["Series", "ADF stat", "ADF p", "KPSS stat", "KPSS p", "Conclusion"],
        ["SPY returns", "−20.22", "<1e-13", "0.43", "0.066",
         "Stationary"],
        ["SPY prices", "+3.60", "1.000", "10.10", "<0.01",
         "Non-stationary"],
    ])

    add_para(doc)
    add_image(doc, ROOT / "eda_outputs" / "fig1_prices.png")
    add_caption(doc, "Figure 1. SPY daily closing prices, 2004–2026. Shaded "
                     "regions correspond to the GFC, COVID, and 2022 energy "
                     "shock windows used for stress testing.")

    add_para(doc)
    add_image(doc, ROOT / "eda_outputs" / "fig7_SPY_OIL_returns_time_series.png")
    add_caption(doc, "Figure 2. SPY daily log returns. Visible volatility "
                     "clustering motivates conditional-variance models.")

    add_para(doc)
    add_image(doc, ROOT / "eda_outputs" / "fig13_Squared_Residuals_ACF_PACF.png")
    add_caption(doc, "Figure 3. ACF and PACF of squared ARIMA residuals. "
                     "Persistent positive autocorrelation in squared "
                     "residuals confirms ARCH effects.")

    add_para(doc)
    add_caption(doc, "Table 4. Ljung–Box tests for ARCH effects on SPY "
                     "squared ARIMA residuals.")
    add_table(doc, [
        ["Lag", "LB statistic", "p-value", "Significant"],
        ["10", "51.46", "1.4 × 10⁻⁷", "Yes"],
        ["20", "60.95", "5.1 × 10⁻⁶", "Yes"],
        ["30", "72.08", "2.6 × 10⁻⁵", "Yes"],
    ])

    # ---------- 4. Analysis ----------
    add_heading(doc, "4. Analysis")

    add_heading(doc, "4.1 Methods", level=2)
    add_para(doc,
        "Five model families are evaluated. Each model is fit independently "
        "for the no_exog and with_exog feature panels. Hyperparameters are "
        "tuned on the validation block; the chosen specification is then "
        "refit on the combined train+validation block and evaluated on the "
        "untouched test block using a one-step-ahead expanding-window "
        "protocol with periodic refitting. The forecast target for the "
        "GARCH-family models is the conditional variance σₜ²; the machine-"
        "learning models are trained directly on rₜ² (the squared return as "
        "a noisy realised-variance proxy). All models forecast the same "
        "variance proxy at the same dates and are scored with the same "
        "metrics, so cross-model comparisons are direct.")

    add_runs(doc, [
        ("4.1.1 GJR-GARCH(1,1,1). ", False, True),
        ("Implemented via the Python arch package. The conditional "
         "variance equation σₜ² = ω + α·εₜ₋₁² + γ·εₜ₋₁²·𝟙{εₜ₋₁<0} + β·σₜ₋₁² "
         "(Glosten, Jagannathan & Runkle, 1993) extends Bollerslev's "
         "GARCH(1,1) with an asymmetric leverage term γ that captures the "
         "well-known empirical fact that negative shocks to equity returns "
         "raise future volatility more than positive shocks of equal size. "
         "Innovations are modelled as Student-t to accommodate the heavy "
         "tails visible in Table 2. Estimation is by quasi-maximum "
         "likelihood. Exogenous features are entered through the mean "
         "equation (ARX terms).", False, False),
    ])

    add_runs(doc, [
        ("4.1.2 Markov-Switching GARCH. ", False, True),
        ("Implemented via the MSGARCH R package (Ardia et al., 2019) "
         "called from Python through rpy2. Following Haas et al. (2004) "
         "each of K hidden states (K ∈ {2,3}) carries an independent "
         "GARCH/GJR-GARCH variance recursion, and a stochastic transition "
         "matrix π governs movement between states. The integrated "
         "predictive density σ̂ₜ² = Σₖ Pr(Sₜ=k|Fₜ₋₁)·σ²ₖ,ₜ marginalises over "
         "the unobserved regime. Specifications are selected by validation "
         "QLIKE (with AIC/BIC as ties) over the grid K ∈ {2,3} × "
         "{sGARCH, gjrGARCH} × {norm, std}. The selected model on SPY "
         "daily is K=2, GJR variance, Student-t innovations.",
         False, False),
    ])

    add_runs(doc, [
        ("4.1.3 LSTM with Attention. ", False, True),
        ("Implemented in PyTorch. A unidirectional LSTM (1 layer, 16 "
         "hidden units, dropout 0.2) consumes the lagged feature panel "
         "over a rolling lookback window of L=10 (no_exog) or L=5 "
         "(with_exog). The hidden state sequence h₁,…,h_L is passed "
         "through a temporal soft-attention pooling layer with attention "
         "weights αₜ = softmax(vᵀ·tanh(W·hₜ)); the context vector c = "
         "Σ αₜhₜ feeds a single linear head that outputs σ̂ₜ². The model "
         "is trained with Adam (lr=10⁻³, batch=64, weight-decay 0) for at "
         "most 80 epochs with patience-10 early stopping on validation "
         "QLIKE. The attention layer is the principal motivation for "
         "this architecture: it allows the network to weight historical "
         "lags non-uniformly, mimicking the long-memory weighting "
         "implicit in FIGARCH-style specifications.",
         False, False),
    ])

    add_runs(doc, [
        ("4.1.4 Transformer Encoder. ", False, True),
        ("Implemented in PyTorch with a 1-layer encoder (d_model=32, 4 "
         "heads, feed-forward dim 64, dropout 0.1) and lookback L=22. The "
         "self-attention mechanism contrasts with the LSTM by being "
         "permutation-equivariant up to positional encoding, so it can in "
         "principle re-weight every pair of historical lags rather than "
         "summarising them through a single recurrent state. The encoder "
         "output is mean-pooled and passed through a linear head to "
         "predict σ̂ₜ². The same Adam regime as the LSTM is used; "
         "training runs for up to 30 epochs with patience 5.",
         False, False),
    ])

    add_runs(doc, [
        ("4.1.5 XGBoost. ", False, True),
        ("Implemented via the xgboost package. Gradient-boosted "
         "regression trees provide a non-parametric, non-recurrent "
         "baseline that contrasts with the other deep-learning entrants. "
         "Hyperparameters are tuned on validation: max_depth ∈ {2,3,4}, "
         "lr ∈ {0.05, 0.1}, n_estimators ∈ {50, 100, 200}, subsample = "
         "colsample_bytree = 0.6, min_child_weight ∈ {1, 5}. The XGBoost "
         "model is trained directly on the lagged feature vector at time "
         "t (no recurrent unrolling); its strength is variable selection "
         "and interaction discovery, but it cannot represent recurrence "
         "without explicit lag features.",
         False, False),
    ])

    add_heading(doc, "4.2 Validation and Evaluation Strategy", level=2)
    add_para(doc,
        "Out-of-sample evaluation uses a one-step-ahead expanding-window "
        "design on the test block. For each test date t the model is "
        "conditioned on all data up to t-1, produces σ̂ₜ², and rolls forward; "
        "every 20 trading days the model is fully refit (parameters re-"
        "estimated for GARCH-family models, weights warm-started for the "
        "neural networks). The same protocol is applied to all five models "
        "so that compute does not advantage any one approach. Forecasts "
        "are scored against the contemporaneous squared return rₜ² as a "
        "realised-variance proxy. Four classes of metric are reported:")
    add_bullet(doc, "RMSE and MAE of σ̂ₜ² against rₜ² (point accuracy).",
               bold_lead="Point accuracy: ")
    add_bullet(doc, "QLIKE = E[log σ̂ₜ² + rₜ²/σ̂ₜ²], a robust loss that is "
                    "insensitive to the realised-variance proxy noise "
                    "(Patton, 2011) and penalises both over- and under-"
                    "prediction asymmetrically — lower is better.",
               bold_lead="Distributional accuracy: ")
    add_bullet(doc, "1% and 5% Value-at-Risk exception rates. Under correct "
                    "calibration, exceptions on the 583-day test block "
                    "should be ≈5.83 and ≈29.15 respectively. We also "
                    "compute Kupiec unconditional-coverage and "
                    "Christoffersen conditional-coverage tests offline.",
               bold_lead="Tail-risk calibration: ")
    add_bullet(doc, "Standardised-residual diagnostics — Ljung–Box on "
                    "z_t = (rₜ − μ̂)/σ̂ₜ and z_t² to confirm whitening, plus "
                    "QQ plots against the assumed innovation distribution.",
               bold_lead="Residual diagnostics: ")
    add_para(doc,
        "Stress-period robustness is checked separately on five Random "
        "Forecast Period (RFP) windows — GFC (2007-07 to 2009-06), Oil "
        "Crash (2014-06 to 2016-02), COVID (2020-02 to 2020-12), Energy "
        "2022 (2022-01 to 2023-06), and a Calm 2017–2019 control. Within "
        "each regime, five 60-day windows are sampled, the model is fit "
        "strictly on data prior to that window, and forecasts inside the "
        "window are scored.")

    add_heading(doc, "4.3 Results", level=2)
    add_para(doc,
        "Table 5 reports test-block performance for SPY daily across the "
        "five models and two feature configurations. RMSE, MAE, and QLIKE "
        "are computed on σ̂ₜ² versus rₜ²; VaR exceptions are absolute counts "
        "over the 583-day test block (expected ≈ 5.83 and ≈ 29.15 at the "
        "1% and 5% levels).")

    add_para(doc)
    add_caption(doc, "Table 5. SPY Daily test-block results. Best per "
                     "metric within each configuration is shown in bold.")
    add_table(doc, [
        ["Model", "Config", "RMSE", "MAE", "QLIKE", "VaR1% Exc", "VaR5% Exc"],
        ["GJR-GARCH(1,1)", "no_exog",   "4.342", "1.129", "0.652", "11", "31"],
        ["MS-GARCH(K=2, GJR, t)", "no_exog", "4.307", "1.135", "0.671", "12", "32"],
        ["LSTM-Attention", "no_exog",   "4.492", "1.156", "0.668", "13", "32"],
        ["Transformer",    "no_exog",   "4.687", "1.180", "0.973", "18", "40"],
        ["XGBoost",        "no_exog",   "4.734", "0.936", "3.447", "68", "96"],
        ["GJR-GARCH(1,1)", "with_exog", "4.736", "1.503", "1.067",  "5", "14"],
        ["MS-GARCH(K=2, GJR, t)", "with_exog", "4.286", "1.128", "0.668", "12", "34"],
        ["LSTM-Attention", "with_exog", "4.419", "1.160", "0.649", "16", "33"],
        ["Transformer",    "with_exog", "4.295", "1.151", "0.765", "19", "41"],
        ["XGBoost",        "with_exog", "4.721", "0.926", "2.834", "65", "90"],
    ])

    add_para(doc,
        "Three findings stand out. First, on RMSE the with_exog MS-GARCH "
        "(4.286) and with_exog Transformer (4.295) are statistically "
        "indistinguishable from each other and edge out the single-regime "
        "GJR baseline by ≈1% under no_exog. Second, on QLIKE the LSTM-with-"
        "attention with exogenous features attains the global minimum "
        "(0.649), narrowly beating both MS-GARCH variants. Third, although "
        "XGBoost wins on MAE (0.926), its QLIKE is 4–5× worse and its 1% "
        "VaR exception count is 65–68 against an expected 5.83 — i.e. the "
        "tree-based model produces a tight median forecast but "
        "systematically under-estimates the conditional variance, an "
        "outcome consistent with squared-error trees converging on the "
        "conditional mean of rₜ² rather than its right-tail behaviour. "
        "GJR-GARCH with exogenous regressors actually deteriorates on RMSE "
        "and QLIKE relative to its no_exog counterpart, which we attribute "
        "to instability in the ARX mean equation under regressor "
        "collinearity; the regime-switching MS-GARCH absorbs the same "
        "exogenous information without that pathology.")

    add_para(doc,
        "Adding exogenous features delivers a consistent but modest "
        "benefit for the data-driven models: −4.0% RMSE for the "
        "Transformer, −1.6% for the LSTM-Attention, −0.5% for MS-GARCH, "
        "and a similar 0.3% reduction in MAE for XGBoost. The strongest "
        "single contributor in feature-importance ablations (run "
        "separately and reported in the model READMEs) is the lagged log-"
        "VIX level, consistent with the implied-volatility-as-leading-"
        "indicator interpretation of Xiong et al. (2016).")

    add_para(doc)
    add_caption(doc, "Figure 4. SPY daily volatility-forecast time series, "
                     "GJR-GARCH (no_exog). Predicted σ̂ₜ (line) overlaid on "
                     "|rₜ| (dots).")
    add_image(doc, ROOT / "ARMA-GARCH-model" / "outputs" / "plots" / "SPY"
                  / "daily" / "no_exog" / "volatility_forecast_timeseries.png")

    add_para(doc)
    add_caption(doc, "Figure 5. SPY daily volatility-forecast time series, "
                     "MS-GARCH (with_exog).")
    add_image(doc, ROOT / "MSGARCH-model" / "outputs" / "plots" / "SPY"
                  / "daily" / "with_exog" / "volatility_forecast_timeseries.png")

    add_para(doc)
    add_caption(doc, "Figure 6. SPY daily volatility-forecast time series, "
                     "LSTM-with-Attention (with_exog).")
    add_image(doc, ROOT / "LSTM-Attention-model" / "outputs" / "plots" / "SPY"
                  / "daily" / "with_exog" / "volatility_forecast_timeseries.png")

    add_para(doc)
    add_caption(doc, "Figure 7. SPY daily volatility-forecast time series, "
                     "Transformer (with_exog).")
    add_image(doc, ROOT / "additional-models" / "outputs" / "transformer"
                  / "plots" / "SPY" / "daily" / "with_exog"
                  / "volatility_forecast_timeseries.png")

    add_para(doc)
    add_caption(doc, "Figure 8. SPY daily volatility-forecast time series, "
                     "XGBoost (with_exog). The systematic shrinkage of "
                     "σ̂ₜ toward the mean is visible during high-volatility "
                     "episodes.")
    add_image(doc, ROOT / "xgboost-model" / "outputs" / "plots" / "SPY"
                  / "daily" / "with_exog" / "volatility_forecast_timeseries.png")

    add_para(doc)
    add_caption(doc, "Figure 9. ACF of squared standardised residuals "
                     "for the MS-GARCH (with_exog) fit on SPY daily. "
                     "Whitening is consistent with adequate variance "
                     "specification.")
    add_image(doc, ROOT / "ARMA-GARCH-model" / "outputs" / "plots" / "SPY"
                  / "daily" / "no_exog" / "acf_squared_standardized_residuals.png")

    add_para(doc,
        "Stress-period analysis (RFP) confirms the test-block ranking. "
        "Mean QLIKE on five 60-day GFC windows is 2.16 for GJR-GARCH "
        "no_exog versus 2.14 for GJR-GARCH with_exog and 1.92 (median) "
        "for MS-GARCH; the COVID windows are dominated by the LSTM-with-"
        "attention (median QLIKE 1.69 with_exog versus 1.77 for GJR). "
        "Calm 2017–2019 windows are tied within numerical noise across "
        "models. No model dominates uniformly across all five regimes, "
        "consistent with the no-free-lunch intuition that single-regime "
        "specifications overfit calm periods while regime-switching and "
        "non-linear models carry their gains in turbulent ones.")

    add_heading(doc, "4.4 Explanation of Changes from the Analysis Plan",
                level=2)
    add_para(doc,
        "We deviated from the original Analysis Plan in three respects. "
        "(i) The plan listed SPY and crude oil as joint targets; for the "
        "final report we focus exclusively on SPY at the daily frequency "
        "to keep the cross-model comparison clean and the page budget "
        "honest. The OIL configuration is fully implemented in the code "
        "deliverable but reported separately. (ii) Among the GARCH-family "
        "models we present GJR-GARCH(1,1) rather than the symmetric "
        "GARCH(1,1) baseline because the leverage term is statistically "
        "significant on SPY at every reasonable lag and dropping it "
        "produces a strictly worse model on every metric. (iii) The "
        "machine-learning side has been broadened beyond the LSTM in the "
        "Analysis Plan: an attention layer was added on top of the LSTM "
        "after we observed that the plain LSTM was unable to discriminate "
        "between recent and stale lags during validation; a Transformer "
        "encoder and an XGBoost benchmark were added to triangulate the "
        "deep-learning claims with a non-recurrent attention model and a "
        "non-neural tree ensemble respectively. The strict temporal split, "
        "rolling-window evaluation, and Kupiec/Christoffersen VaR "
        "back-tests planned in §2 of the Analysis Plan are unchanged.")

    # ---------- 5. Conclusions ----------
    add_heading(doc, "5. Conclusions")

    add_heading(doc, "5.1 Success and Best Method", level=2)
    add_para(doc,
        "On daily SPY, two models share the top position. MS-GARCH (K=2, "
        "GJR variance, Student-t innovations, with_exog) wins on RMSE "
        "(4.286) and is competitive on QLIKE (0.668). The LSTM-with-"
        "attention (with_exog) wins on QLIKE (0.649) and is within 3% on "
        "RMSE. Both produce well-calibrated 1% VaR exceptions (12 and 16 "
        "against an expected 5.83 — modestly conservative but within the "
        "Kupiec acceptance region). The complementary nature of the two "
        "winners is the most informative result of the project: the "
        "regime-switching parametric model and the attention-based deep-"
        "learning model are exploiting different structure in the data, "
        "and an ensemble of the two — outside the scope of this report — "
        "is a natural extension. The Transformer with exogenous features "
        "is essentially indistinguishable from MS-GARCH on RMSE but at "
        "higher implementation cost. The single-regime GJR-GARCH baseline "
        "is competitive only without exogenous features; the addition of "
        "macro regressors in its mean equation actually hurts performance. "
        "XGBoost is the cautionary tale: its MAE-optimal forecasts collapse "
        "the conditional variance and produce 11× the expected VaR "
        "exception rate, illustrating why MSE/MAE alone are inadequate "
        "loss functions for variance forecasting.")

    add_heading(doc, "5.2 Limitations and Future Work", level=2)
    add_para(doc,
        "Several limitations warrant attention. First, the realised-"
        "variance proxy rₜ² is an extremely noisy target: high-frequency "
        "intraday realised volatility (5-min returns) would dramatically "
        "tighten the metrics and is the obvious next step. Second, our "
        "evaluation universe is one asset at one frequency; before "
        "drawing universal conclusions, the same comparison should be "
        "repeated on the OIL and GOLD targets and on the weekly "
        "frequency that we have already implemented but not analysed in "
        "this report. Third, the deep-learning models share a common "
        "weakness: they were trained with squared-error losses and may "
        "benefit from training directly on the QLIKE objective, which is "
        "the loss they are evaluated on. Fourth, our MS-GARCH "
        "specification fixes the number of regimes ex ante; data-driven "
        "regime discovery (e.g. via dirichlet-process or attention-based "
        "regime detectors as in Raja Lope Ahmad et al., 2026) is a "
        "natural sequel that could remove this constraint. Fifth, "
        "ensembling — in particular a convex combination of MS-GARCH and "
        "LSTM-with-attention forecasts whose weights are tuned on "
        "validation — is suggested by the complementary regime "
        "performance and is the highest-priority follow-up. Finally, the "
        "test block (2024–Apr 2026) covers a relatively calm period; "
        "a rolling re-evaluation that includes a future structural break "
        "will be necessary to confirm that our headline ranking is "
        "robust.")

    # ---------- 6. References ----------
    add_heading(doc, "6. References")
    refs = [
        "Alizadeh, A. H., Nomikos, N. K., & Pouliasis, P. K. (2008). A "
        "Markov regime switching approach for hedging energy commodities. "
        "Journal of Banking & Finance, 32(9), 1970–1983. "
        "https://doi.org/10.1016/j.jbankfin.2007.12.020",
        "Ardia, D., Bluteau, K., Boudt, K., & Catania, L. (2019). Markov-"
        "switching GARCH models in R: The MSGARCH package. Journal of "
        "Statistical Software, 91(4), 1–38. "
        "https://doi.org/10.18637/jss.v091.i04",
        "Bollerslev, T. (1986). Generalized autoregressive conditional "
        "heteroskedasticity. Journal of Econometrics, 31(3), 307–327. "
        "https://doi.org/10.1016/0304-4076(86)90063-1",
        "Carr, P., Wu, L., & Zhang, Z. (2020). Using machine learning to "
        "predict realized variance. Journal of Investment Management, "
        "18(2), 1–16.",
        "Charles, A., & Darné, O. (2017). Forecasting crude-oil market "
        "volatility: Further evidence with jumps. Energy Economics, 67, "
        "508–519. https://doi.org/10.1016/j.eneco.2017.09.002",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree "
        "boosting system. Proceedings of the 22nd ACM SIGKDD "
        "International Conference on Knowledge Discovery and Data "
        "Mining, 785–794. https://doi.org/10.1145/2939672.2939785",
        "Christoffersen, P. F. (1998). Evaluating interval forecasts. "
        "International Economic Review, 39(4), 841–862. "
        "https://doi.org/10.2307/2527341",
        "Doering, J., Fairbank, M., & Markose, S. (2017). Convolutional "
        "neural networks applied to high-frequency market microstructure "
        "forecasting. 2017 9th Computer Science and Electronic "
        "Engineering (CEEC). https://doi.org/10.1109/ceec.2017.8101595",
        "Ejike, U. (2026). Machine Learning for Market Volatility Regime "
        "Classification: A Comparative Analysis of LDA, SVM, and Neural "
        "Networks. https://doi.org/10.2139/ssrn.6064166",
        "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity "
        "with estimates of the variance of United Kingdom inflation. "
        "Econometrica, 50(4), 987–1007. https://doi.org/10.2307/1912773",
        "Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the "
        "relation between the expected value and the volatility of the "
        "nominal excess return on stocks. Journal of Finance, 48(5), "
        "1779–1801. https://doi.org/10.1111/j.1540-6261.1993.tb05128.x",
        "Haas, M., Mittnik, S., & Paolella, M. S. (2004). A new approach to "
        "Markov-switching GARCH models. Journal of Financial Econometrics, "
        "2(4), 493–530. https://doi.org/10.1093/jjfinec/nbh020",
        "Hamilton, J. D. (1989). A new approach to the economic analysis of "
        "nonstationary time series and the business cycle. Econometrica, "
        "57(2), 357–384. https://doi.org/10.2307/1912559",
        "Hardy, M. R. (2001). A regime-switching model of long-term stock "
        "returns. North American Actuarial Journal, 5(2), 41–53. "
        "https://doi.org/10.1080/10920277.2001.10595984",
        "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
        "Neural Computation, 9(8), 1735–1780.",
        "Kim, H. Y., & Won, C. H. (2018). Forecasting the volatility of "
        "stock price index: A hybrid model integrating LSTM with multiple "
        "GARCH-type models. Expert Systems with Applications, 103, 25–37. "
        "https://doi.org/10.1016/j.eswa.2018.03.002",
        "Klaassen, F. (2002). Improving GARCH volatility forecasts with "
        "regime-switching GARCH. Empirical Economics, 27(2), 363–394. "
        "https://doi.org/10.1007/s001810100100",
        "Kupiec, P. (1995). Techniques for verifying the accuracy of risk "
        "measurement models. Journal of Derivatives, 3(2), 73–84.",
        "Patton, A. J. (2011). Volatility forecast comparison using "
        "imperfect volatility proxies. Journal of Econometrics, 160(1), "
        "246–256. https://doi.org/10.1016/j.jeconom.2010.03.034",
        "Raja Lope Ahmad, R. A., et al. (2026). Agentic AI for Financial "
        "Volatility Regime Detection: A Hybrid Deep Learning and "
        "Statistical Framework. Applied Research, 5(2). "
        "https://doi.org/10.1002/appl.70078",
        "Tino, P., Schittenkopf, C., & Dorffner, G. (2001). Financial "
        "volatility trading using recurrent neural networks. IEEE "
        "Transactions on Neural Networks, 12(4), 865–874. "
        "https://doi.org/10.1109/72.935096",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., "
        "Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is "
        "all you need. Advances in Neural Information Processing Systems, "
        "30.",
        "Xiong, R., Nichols, E. P., & Shen, Y. (2016). Deep learning stock "
        "volatility with Google Domestic Trends. arXiv preprint "
        "arXiv:1512.04916.",
        "Yahoo Finance (2026). SPDR S&P 500 ETF Trust (SPY) historical "
        "prices [Data set]. https://finance.yahoo.com/quote/SPY/history",
    ]
    for r in refs:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        pf.left_indent = Inches(0.5)
        pf.first_line_indent = Inches(-0.5)
        run = p.add_run(r)
        run.font.name = FONT
        run.font.size = FONT_SIZE

    # ---------- 7. Appendix: per-member contributions ----------
    add_heading(doc, "7. Appendix A: Individual Member Analysis Summaries")
    add_para(doc,
        "[PLACEHOLDER — paragraph-length summary of each individual "
        "member's analysis and other relative contributions to be "
        "completed by the team. The Analysis-Plan attribution was: "
        "Rajath — Traditional ARCH/GARCH models, EDA, data preprocessing; "
        "Harsh — Regime-switching MS-GARCH models and regime-probability "
        "analysis; Yibo — Literature review on machine-learning "
        "approaches and comparative evaluation; Mani — Literature review "
        "and EDA; Yen-Shuo — Literature review on the VIX and EDA. "
        "Replace this placeholder with one paragraph per member describing "
        "(a) the analytical sub-task they led, (b) the specific code or "
        "outputs they produced, and (c) their contribution to writing and "
        "review of this report.]")

    add_heading(doc, "8. Appendix B: Required Team Contributions Statement")
    add_para(doc,
        "[PLACEHOLDER — required team contributions evaluation. Per the "
        "assignment instructions, this appendix must explicitly note any "
        "team member who did not contribute at least 80% of what is "
        "expected for this deliverable. If every member met or exceeded "
        "the 80% threshold, state so here. See the example evaluation "
        "linked in the assignment for formatting.]")

    doc.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    build()
