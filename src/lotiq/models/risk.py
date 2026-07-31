"""
Map a continuous risk score (0-100) onto the operational language a QC manager
actually uses: a level, a colour, an emoji, and a recommended action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from lotiq.config import band_for_score

# What the QC team should do at each band. Kept here so the API, dashboard and
# explanations all recommend the same thing.
_RECOMMENDATIONS: dict[str, str] = {
    "healthy": "Routine monitoring. No action needed.",
    "moderate": "Watch this lot. Re-check sensor and lab values within the week.",
    "high": "Prioritise for QC inspection in the next 24-48 hours.",
    "critical": "Immediate QC inspection. Consider isolating or expediting this lot.",
}


@dataclass(frozen=True)
class RiskResult:
    risk_score: float
    risk_level: str      # short status word, e.g. "Critical"
    level_key: str       # machine key, e.g. "critical"
    colour: str          # hex
    emoji: str
    recommendation: str

    def as_dict(self) -> dict:
        return asdict(self)


def score_to_risk(score: float) -> RiskResult:
    """Turn a 0-100 score into a fully-described RiskResult."""
    score = float(max(0.0, min(100.0, score)))
    band = band_for_score(score)
    return RiskResult(
        risk_score=round(score, 1),
        risk_level=band.label,
        level_key=band.name,
        colour=band.colour,
        emoji=band.emoji,
        recommendation=_RECOMMENDATIONS[band.name],
    )
