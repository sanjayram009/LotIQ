"""
The single scoring entry point used by both the API and the dashboard.

``score_record`` takes a raw record (dict), runs the full pipeline
-- preprocess -> model -> risk band -> attribution -> explanation -- and returns
one tidy dict. Keeping this here means the API and dashboard can never drift
apart in how they score a lot.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from lotiq.config import MODEL_PATH
from lotiq.data.preprocessing import record_to_features
from lotiq.explain.attribution import explain_instance
from lotiq.explain.explainer import (
    build_explanation,
    llm_explanation,
    protective_factors,
    to_risk_factors,
)
from lotiq.models.risk import score_to_risk


@lru_cache(maxsize=4)
def load_bundle(model_path: str = str(MODEL_PATH)) -> dict:
    """Load and cache the model bundle. Raises FileNotFoundError with guidance."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model found at {path}. Train one first:\n"
            f"    python -m lotiq.models.train"
        )
    return joblib.load(path)


def predict_score(record: Mapping[str, Any], bundle: dict | None = None) -> float:
    """Just the number: a 0-100 risk score for a record."""
    bundle = bundle or load_bundle()
    x = record_to_features(record, bundle.get("commodity", "chilli"))
    return float(np.clip(bundle["model"].predict(x)[0], 0, 100))


def score_record(
    record: Mapping[str, Any],
    bundle: dict | None = None,
    top_k: int = 3,
    use_llm: bool = False,
) -> dict:
    """Full scoring: score + band + risk factors + contributions + explanation."""
    bundle = bundle or load_bundle()
    commodity = bundle.get("commodity", "chilli")

    x = record_to_features(record, commodity)
    raw = float(np.clip(bundle["model"].predict(x)[0], 0, 100))
    risk = score_to_risk(raw)

    contributions = explain_instance(bundle, x)  # all features, sorted by impact
    factors = to_risk_factors(contributions, top_k=top_k)
    protective = protective_factors(contributions, top_k=2)

    lot_id = str(record.get("lot_id", "UNKNOWN"))

    explanation = None
    if use_llm:
        explanation = llm_explanation(
            lot_id, risk.risk_score, risk.risk_level, contributions, top_k=top_k
        )
    if not explanation:
        explanation = build_explanation(
            lot_id, risk.risk_score, risk.risk_level, contributions, top_k=top_k
        )

    return {
        "lot_id": lot_id,
        "commodity": commodity,
        "risk_score": risk.risk_score,
        "risk_level": risk.risk_level,
        "colour": risk.colour,
        "emoji": risk.emoji,
        "recommendation": risk.recommendation,
        "risk_factors": factors,
        "protective_factors": protective,
        "contributions": [c.as_dict() for c in contributions[:top_k]],
        "explanation": explanation,
    }
