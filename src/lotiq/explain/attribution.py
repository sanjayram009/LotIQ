"""
Per-lot risk attribution: *why* did this lot get the score it got?

Two backends, one interface. Both return signed contributions in risk-score
points, using the same sign convention:

    contribution > 0  =>  this feature pushes THIS lot's risk ABOVE a typical lot
    contribution < 0  =>  this feature pulls it BELOW a typical lot (protective)

1. SHAP (preferred): when the model is an XGBoost tree ensemble and ``shap`` is
   installed, we use ``shap.TreeExplainer``. SHAP values for a regressor are
   already expressed in output units (points), and the sign convention matches.

2. Ablation (fallback): we start from a "normal" lot (every feature at its
   training median) and ask, one feature at a time, "how much does THIS lot's
   value for the feature move the score?" The contribution is
   ``predict(baseline with feature = lot value) - predict(baseline)``. This
   from-baseline (occlusion) direction is stable and correctly signed even when
   the model saturates at 0 or 100, unlike perturbing down from an already
   extreme prediction. It ignores feature interactions (SHAP doesn't), so it is
   an approximation -- which is exactly why SHAP is the preferred backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lotiq.config import COMMODITIES, FEATURE_LABELS, FEATURE_UNITS, MODEL_FEATURES

# Cache SHAP explainers by model identity so we build them once per process.
_SHAP_CACHE: dict[int, object] = {}


@dataclass
class FeatureContribution:
    feature: str
    label: str
    value: float
    unit: str
    contribution: float        # signed, in risk-score points
    direction: int             # +1 = higher is riskier, -1 = lower is riskier
    threshold: float
    exceeds_threshold: bool    # is the value on the "risky" side of the threshold?

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": round(self.value, 2),
            "unit": self.unit,
            "contribution": round(self.contribution, 2),
            "direction": self.direction,
            "threshold": self.threshold,
            "exceeds_threshold": self.exceeds_threshold,
        }


def _try_shap_values(bundle: dict, x_row: pd.DataFrame) -> np.ndarray | None:
    """Return SHAP values for the row, or None if SHAP/XGBoost isn't usable."""
    if bundle.get("backend") != "xgboost":
        return None
    try:
        import shap  # type: ignore
    except Exception:
        return None
    model = bundle["model"]
    key = id(model)
    explainer = _SHAP_CACHE.get(key)
    if explainer is None:
        explainer = shap.TreeExplainer(model)
        _SHAP_CACHE[key] = explainer
    values = explainer.shap_values(x_row[MODEL_FEATURES])
    return np.asarray(values).reshape(-1)


def _ablation_values(bundle: dict, x_row: pd.DataFrame) -> np.ndarray:
    """From-baseline occlusion attribution (SHAP-free fallback).

    Contribution_i = predict(baseline lot but feature i = this lot's value)
                     - predict(baseline lot).
    Positive => this lot's value for the feature raises risk vs a normal lot.
    """
    import pandas as pd

    model = bundle["model"]
    baseline = bundle["baseline"]

    baseline_row = pd.DataFrame([{f: baseline[f] for f in MODEL_FEATURES}])
    base_pred = float(model.predict(baseline_row[MODEL_FEATURES])[0])

    contribs = np.zeros(len(MODEL_FEATURES))
    for i, feature in enumerate(MODEL_FEATURES):
        probe = baseline_row.copy()
        probe.iloc[0, probe.columns.get_loc(feature)] = float(x_row.iloc[0][feature])
        alt_pred = float(model.predict(probe[MODEL_FEATURES])[0])
        contribs[i] = alt_pred - base_pred
    return contribs


def explain_instance(
    bundle: dict,
    x_row: pd.DataFrame,
    top_k: int | None = None,
) -> list[FeatureContribution]:
    """Explain a single lot. Returns contributions sorted by absolute impact."""
    commodity = bundle.get("commodity", "chilli")
    spec = COMMODITIES[commodity]

    values = _try_shap_values(bundle, x_row)
    if values is None:
        values = _ablation_values(bundle, x_row)

    results: list[FeatureContribution] = []
    for i, feature in enumerate(MODEL_FEATURES):
        raw_value = float(x_row.iloc[0][feature])
        direction = spec.direction.get(feature, +1)
        threshold = spec.thresholds.get(feature, float("nan"))

        if np.isnan(threshold):
            exceeds = False
        elif direction > 0:
            exceeds = raw_value > threshold       # too high
        else:
            exceeds = raw_value < threshold       # too low

        results.append(
            FeatureContribution(
                feature=feature,
                label=FEATURE_LABELS.get(feature, feature),
                value=raw_value,
                unit=FEATURE_UNITS.get(feature, ""),
                contribution=float(values[i]),
                direction=direction,
                threshold=threshold,
                exceeds_threshold=bool(exceeds),
            )
        )

    results.sort(key=lambda c: abs(c.contribution), reverse=True)
    return results[:top_k] if top_k else results
