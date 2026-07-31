"""
Turn raw lot records into the exact numeric feature matrix the model expects.

The model is framework-agnostic: it only ever sees the ordered numeric columns
in ``config.MODEL_FEATURES``. This module is the one place that:

  * converts the categorical ``initial_quality_grade`` (A/B/C) into the ordinal
    ``grade_ordinal`` (0/1/2),
  * clips values into physically plausible ranges (so a typo like humidity=760
    can't blow up the model), and
  * guarantees column order.

Both the training pipeline and the live API go through here, which is what keeps
"training-time" and "serving-time" features identical.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from lotiq.config import (
    COMMODITIES,
    GRADE_TO_ORDINAL,
    MODEL_FEATURES,
)


def grade_to_ordinal(grade: Any) -> int:
    """Map an A/B/C grade to 0/1/2. Unknown or missing grades default to B (1)."""
    if grade is None:
        return GRADE_TO_ORDINAL["B"]
    key = str(grade).strip().upper()
    return GRADE_TO_ORDINAL.get(key, GRADE_TO_ORDINAL["B"])


def clip_to_ranges(df: pd.DataFrame, commodity: str = "chilli") -> pd.DataFrame:
    """Clip each numeric feature into the commodity's plausible physical range."""
    spec = COMMODITIES[commodity]
    out = df.copy()
    for feature, (low, high) in spec.ranges.items():
        if feature in out.columns:
            out[feature] = out[feature].clip(lower=low, upper=high)
    return out


def build_feature_frame(df: pd.DataFrame, commodity: str = "chilli") -> pd.DataFrame:
    """Return a DataFrame with exactly ``MODEL_FEATURES`` columns, in order.

    Accepts a frame that has the raw feature columns plus
    ``initial_quality_grade``. Adds ``grade_ordinal`` if missing and clips.
    """
    work = df.copy()
    if "grade_ordinal" not in work.columns:
        if "initial_quality_grade" not in work.columns:
            raise KeyError(
                "Expected 'initial_quality_grade' or 'grade_ordinal' in the frame."
            )
        work["grade_ordinal"] = work["initial_quality_grade"].map(grade_to_ordinal)

    work = clip_to_ranges(work, commodity)

    missing = [c for c in MODEL_FEATURES if c not in work.columns]
    if missing:
        raise KeyError(f"Missing required model features: {missing}")

    return work[MODEL_FEATURES].astype(float)


def record_to_features(record: Mapping[str, Any], commodity: str = "chilli") -> pd.DataFrame:
    """Convert a single API-style record (a dict) into a 1-row feature frame.

    Missing optional fields are filled with the commodity's midpoint so a partial
    payload still scores rather than erroring -- useful when, say, a lab value
    hasn't been entered yet.
    """
    spec = COMMODITIES[commodity]
    row: dict[str, Any] = {}

    for feature in ["temperature", "humidity", "moisture_content", "co2_level",
                    "oleoresin_content", "colour_score", "storage_days"]:
        if feature in record and record[feature] is not None:
            row[feature] = float(record[feature])
        else:
            low, high = spec.ranges[feature]
            row[feature] = (low + high) / 2.0  # neutral midpoint fallback

    grade = record.get("initial_quality_grade", "B")
    row["initial_quality_grade"] = grade
    row["grade_ordinal"] = grade_to_ordinal(grade)

    return build_feature_frame(pd.DataFrame([row]), commodity)
