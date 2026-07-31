"""Tests for the batch scoring CLI (`lotiq.cli`)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from lotiq.cli import score_frame
from lotiq.data.generate import generate_dataset
from lotiq.models.train import train

_TMP = Path(tempfile.mkdtemp(prefix="lotiq_cli_"))
_MODEL: Path | None = None

_RAW_COLS = [
    "lot_id", "commodity", "temperature", "humidity", "moisture_content",
    "co2_level", "oleoresin_content", "colour_score", "storage_days",
    "initial_quality_grade",
]


def _model_path() -> Path:
    global _MODEL
    if _MODEL is None:
        csv = _TMP / "data.csv"
        model = _TMP / "model.pkl"
        metrics = _TMP / "metrics.json"
        generate_dataset(n=600, seed=21).to_csv(csv, index=False)
        train(data_path=csv, model_path=model, metrics_path=metrics)
        _MODEL = model
    return _MODEL


def test_score_frame_appends_columns_and_preserves_rows():
    raw = generate_dataset(n=15, seed=99)[_RAW_COLS]
    out = score_frame(raw, model_path=_model_path(), top_k=3)

    assert len(out) == len(raw)  # rows preserved
    for col in ["risk_score", "risk_level", "recommendation", "explanation",
                "top_factor_1", "top_factor_2", "top_factor_3"]:
        assert col in out.columns, col
    # Original columns still present.
    for col in _RAW_COLS:
        assert col in out.columns, col


def test_scored_values_in_range():
    raw = generate_dataset(n=15, seed=100)[_RAW_COLS]
    out = score_frame(raw, model_path=_model_path(), top_k=3)
    assert out["risk_score"].min() >= 0.0
    assert out["risk_score"].max() <= 100.0
    assert set(out["risk_level"]).issubset({"Healthy", "Moderate", "High", "Critical"})
