"""
Tests for prediction, attribution and preprocessing.

A small model is trained once into a temp directory (using the scikit-learn
backend, so these tests pass without XGBoost installed) and reused across tests.
The helper is deliberately a plain function, not a pytest fixture, so the suite
can also be executed by a minimal runner.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from lotiq.config import MODEL_FEATURES
from lotiq.data.generate import generate_dataset
from lotiq.data.preprocessing import record_to_features
from lotiq.models.predict import load_bundle, predict_score, score_record
from lotiq.models.train import train

_TMP = Path(tempfile.mkdtemp(prefix="lotiq_test_"))
_BUNDLE: dict | None = None

GOOD_LOT = {
    "lot_id": "L-GOOD", "commodity": "chilli", "temperature": 22.0, "humidity": 48.0,
    "moisture_content": 8.5, "co2_level": 550, "oleoresin_content": 8.8,
    "colour_score": 92, "storage_days": 8, "initial_quality_grade": "A",
}
BAD_LOT = {
    "lot_id": "L-BAD", "commodity": "chilli", "temperature": 34.0, "humidity": 82.0,
    "moisture_content": 13.5, "co2_level": 1400, "oleoresin_content": 4.2,
    "colour_score": 55, "storage_days": 70, "initial_quality_grade": "C",
}


def _bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        csv = _TMP / "data.csv"
        model = _TMP / "model.pkl"
        metrics = _TMP / "metrics.json"
        generate_dataset(n=800, seed=11).to_csv(csv, index=False)
        train(data_path=csv, model_path=model, metrics_path=metrics)
        _BUNDLE = load_bundle(str(model))
    return _BUNDLE


def test_record_to_features_shape_and_order():
    x = record_to_features(GOOD_LOT)
    assert list(x.columns) == MODEL_FEATURES
    assert x.shape == (1, len(MODEL_FEATURES))


def test_record_to_features_fills_missing():
    partial = {"temperature": 30.0, "humidity": 70.0, "moisture_content": 12.0,
               "storage_days": 40}
    x = record_to_features(partial)  # no lab values / grade supplied
    assert x.shape == (1, len(MODEL_FEATURES))
    assert not x.isna().any().any()


def test_predict_score_in_range():
    score = predict_score(BAD_LOT, bundle=_bundle())
    assert 0.0 <= score <= 100.0


def test_bad_lot_riskier_than_good_lot():
    good = predict_score(GOOD_LOT, bundle=_bundle())
    bad = predict_score(BAD_LOT, bundle=_bundle())
    assert bad > good
    # And the gap should be substantial, not marginal.
    assert bad - good > 30


def test_score_record_contract():
    out = score_record(BAD_LOT, bundle=_bundle())
    expected_keys = {
        "lot_id", "commodity", "risk_score", "risk_level", "colour", "emoji",
        "recommendation", "risk_factors", "protective_factors", "contributions",
        "explanation",
    }
    assert expected_keys.issubset(out)
    assert isinstance(out["risk_factors"], list)
    assert isinstance(out["contributions"], list) and len(out["contributions"]) >= 1
    assert isinstance(out["explanation"], str) and out["explanation"]


def test_bad_lot_has_risk_factors_good_lot_has_protective():
    bad = score_record(BAD_LOT, bundle=_bundle())
    good = score_record(GOOD_LOT, bundle=_bundle())
    assert len(bad["risk_factors"]) >= 1
    assert len(good["protective_factors"]) >= 1


def test_contribution_signs_are_directional():
    # For the bad lot, moisture is high and should push risk UP (positive).
    out = score_record(BAD_LOT, bundle=_bundle())
    by_feature = {c["feature"]: c for c in out["contributions"]}
    if "moisture_content" in by_feature:
        assert by_feature["moisture_content"]["contribution"] > 0
