"""Tests for the risk-band mapping."""
from __future__ import annotations

import pytest

from lotiq.config import band_for_score
from lotiq.models.risk import score_to_risk


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "healthy"),
        (29.9, "healthy"),
        (30.0, "moderate"),
        (59.9, "moderate"),
        (60.0, "high"),
        (79.9, "high"),
        (80.0, "critical"),
        (100.0, "critical"),
    ],
)
def test_band_boundaries(score, expected):
    assert band_for_score(score).name == expected


def test_score_to_risk_clamps_and_describes():
    r = score_to_risk(150.0)  # out of range -> clamped
    assert r.risk_score == 100.0
    assert r.level_key == "critical"
    assert r.recommendation  # non-empty
    assert r.emoji and r.colour.startswith("#")


def test_score_to_risk_low():
    r = score_to_risk(5.0)
    assert r.level_key == "healthy"
    assert r.risk_level == "Healthy"


def test_as_dict_roundtrip():
    d = score_to_risk(72.0).as_dict()
    assert d["risk_level"] == "High"
    assert set(d) == {
        "risk_score", "risk_level", "level_key", "colour", "emoji", "recommendation"
    }
