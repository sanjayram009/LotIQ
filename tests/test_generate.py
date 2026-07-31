"""Tests for the synthetic data generator."""
from __future__ import annotations

from lotiq.config import CHILLI, MODEL_FEATURES, TARGET
from lotiq.data.generate import generate_dataset


def test_shape_and_required_columns():
    df = generate_dataset(n=300, seed=1)
    assert len(df) == 300
    required = set(MODEL_FEATURES) | {
        "lot_id", "commodity", "initial_quality_grade", TARGET, "risk_level"
    }
    assert required.issubset(df.columns), required - set(df.columns)


def test_features_within_physical_ranges():
    df = generate_dataset(n=800, seed=2)
    for feature, (low, high) in CHILLI.ranges.items():
        assert df[feature].min() >= low - 1e-6, feature
        assert df[feature].max() <= high + 1e-6, feature


def test_risk_score_bounds():
    df = generate_dataset(n=800, seed=3)
    assert df[TARGET].min() >= 0.0
    assert df[TARGET].max() <= 100.0


def test_reproducible_with_seed():
    a = generate_dataset(n=200, seed=7)
    b = generate_dataset(n=200, seed=7)
    # Identical seed => identical data.
    assert a.equals(b)


def test_seed_changes_data():
    a = generate_dataset(n=200, seed=7)
    b = generate_dataset(n=200, seed=8)
    assert not a[TARGET].equals(b[TARGET])


def test_band_spread():
    # A reasonable sample should populate at least three of the four risk bands.
    df = generate_dataset(n=1000, seed=4)
    assert df["risk_level"].nunique() >= 3


def test_no_missing_values():
    df = generate_dataset(n=500, seed=5)
    assert int(df.isna().sum().sum()) == 0
