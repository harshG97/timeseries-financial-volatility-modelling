"""Registry of model RFP output locations.

Each model script writes its RFP artifacts under its own directory tree (the
``ARMA-GARCH-model/``, ``MSGARCH-model/``, ``additional-models/`` etc. layouts
are kept as-is). This module is the single source of truth for *where* those
artifacts live, so cross-model aggregation / reporting code does not have to
hard-code paths.

Usage
-----
>>> from src.model_registry import MODEL_REGISTRY
>>> garch = MODEL_REGISTRY["garch"]
>>> pd.read_csv(garch.results_csv)
>>> for fc in garch.forecasts_dir.glob("SPY_daily_no_exog_*.csv"):
...     ...
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelOutputs:
    """Canonical layout for a single model's RFP artifacts.

    All four paths are read-only views; nothing here creates directories.
    """

    name: str
    rfp_dir: Path

    @property
    def results_csv(self) -> Path:
        return self.rfp_dir / f"{self.name}_rfp_results.csv"

    @property
    def summary_csv(self) -> Path:
        return self.rfp_dir / f"{self.name}_rfp_summary.csv"

    @property
    def forecasts_dir(self) -> Path:
        return self.rfp_dir / "forecasts"

    @property
    def plots_dir(self) -> Path:
        return self.rfp_dir / "plots"


MODEL_REGISTRY: dict[str, ModelOutputs] = {
    "garch":          ModelOutputs("garch",          ROOT / "ARMA-GARCH-model"     / "outputs" / "rfp"),
    "msgarch":        ModelOutputs("msgarch",        ROOT / "MSGARCH-model"        / "outputs" / "rfp"),
    "lstm":           ModelOutputs("lstm",           ROOT / "lstm-model"           / "outputs" / "rfp"),
    "lstm_attention": ModelOutputs("lstm_attention", ROOT / "LSTM-Attention-model" / "outputs" / "rfp"),
    "transformer":    ModelOutputs("transformer",    ROOT / "additional-models"    / "outputs" / "transformer" / "rfp"),
    "prophet":        ModelOutputs("prophet",        ROOT / "additional-models"    / "outputs" / "prophet"     / "rfp"),
    "orbit":          ModelOutputs("orbit",          ROOT / "additional-models"    / "outputs" / "orbit"       / "rfp"),
    "silverkite":     ModelOutputs("silverkite",     ROOT / "additional-models"    / "outputs" / "silverkite"  / "rfp"),
}
