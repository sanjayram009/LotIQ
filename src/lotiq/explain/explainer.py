"""
Turn raw feature contributions into language a QC manager can act on.

Two layers:

* ``to_risk_factors`` / ``build_explanation`` -- a deterministic, dependency-free
  template. It always works, costs nothing, and is what the tests pin against.
* ``llm_explanation`` -- an OPTIONAL enhancement that sends the SHAP summary to
  the Anthropic API for a more fluent paragraph. It is only used when the
  ``anthropic`` package is installed AND an ``ANTHROPIC_API_KEY`` is set AND the
  caller opts in. On any problem it silently returns None so the template is used
  instead -- the product never depends on a network call to render a score.
"""
from __future__ import annotations

import os

from lotiq.explain.attribution import FeatureContribution

_DEFAULT_MODEL = os.environ.get("LOTIQ_LLM_MODEL", "claude-haiku-4-5-20251001")

# Features displayed as whole numbers rather than one decimal place.
_INTEGER_FEATURES = {"storage_days", "co2_level"}


def _fmt(value: float, feature: str, unit: str) -> str:
    """Format 'value + unit' with correct spacing.

    Alphabetic units (days, ppm) get a leading space ('79 days'); symbol units
    (%, /100) and degrees do not ('13.2%', '34\u00b0C').
    """
    num = f"{value:.0f}" if feature in _INTEGER_FEATURES else f"{value:.1f}"
    if unit and unit[0].isalpha():
        return f"{num} {unit}"
    return f"{num}{unit}"


def _phrase(fc: FeatureContribution) -> str:
    """Short human phrase for one risk-increasing feature, e.g. 'High moisture (13.2%)'."""
    qualifier = "High" if fc.direction > 0 else "Low"
    return f"{qualifier} {fc.label.lower()} ({_fmt(fc.value, fc.feature, fc.unit)})"


def to_risk_factors(
    contributions: list[FeatureContribution],
    top_k: int = 3,
) -> list[str]:
    """Short bullet phrases for the top risk-INCREASING factors."""
    risky = [c for c in contributions if c.contribution > 0]
    risky.sort(key=lambda c: c.contribution, reverse=True)
    return [_phrase(c) for c in risky[:top_k]]


def protective_factors(
    contributions: list[FeatureContribution],
    top_k: int = 2,
) -> list[str]:
    """Short phrases for the strongest risk-REDUCING factors (the good news)."""
    good = [c for c in contributions if c.contribution < 0]
    good.sort(key=lambda c: c.contribution)  # most negative first
    return [f"{c.label} ({_fmt(c.value, c.feature, c.unit)})" for c in good[:top_k]]


def build_explanation(
    lot_id: str,
    risk_score: float,
    risk_level: str,
    contributions: list[FeatureContribution],
    top_k: int = 3,
) -> str:
    """Deterministic plain-language explanation (no external calls)."""
    factors = to_risk_factors(contributions, top_k=top_k)
    lead = f"Lot {lot_id} has a risk score of {risk_score:.0f}/100 ({risk_level})."

    if not factors:
        return lead + " No individual factor stands out as elevating risk; the lot looks stable."

    detail_bits = []
    for c in [x for x in contributions if x.contribution > 0][:top_k]:
        val = _fmt(c.value, c.feature, c.unit)
        thr = _fmt(c.threshold, c.feature, c.unit)
        if c.exceeds_threshold:
            # The value genuinely crossed the QC threshold.
            side = "above" if c.direction > 0 else "below"
            limit = "safe limit" if c.direction > 0 else "expected level"
            detail_bits.append(f"{c.label.lower()} is {val}, {side} the {limit} of {thr}")
        else:
            # Within the threshold, but still worse than a typical lot.
            side = "higher" if c.direction > 0 else "lower"
            detail_bits.append(f"{c.label.lower()} is {val}, {side} than a typical lot")

    body = "The main drivers are " + "; ".join(detail_bits) + "."
    return f"{lead} {body}"


def _build_llm_prompt(
    lot_id: str,
    risk_score: float,
    risk_level: str,
    contributions: list[FeatureContribution],
    top_k: int,
) -> str:
    lines = [
        f"Lot {lot_id} has been assigned a spoilage-risk score of "
        f"{risk_score:.0f}/100 ({risk_level}).",
        "Top contributing factors (SHAP), with the lot's value and the QC threshold:",
    ]
    for c in contributions[:top_k]:
        side = "above" if (c.direction > 0 and c.exceeds_threshold) else (
            "below" if (c.direction < 0 and c.exceeds_threshold) else "near"
        )
        lines.append(
            f"- {c.label}: {c.value:.1f}{c.unit} (threshold {c.threshold:.1f}{c.unit}, {side}); "
            f"impact {c.contribution:+.1f} points"
        )
    lines.append(
        "\nWrite 2-3 short sentences a warehouse QC manager can act on. "
        "Plain English, no jargon, no markdown. Explain why the lot is at this "
        "risk level and what to check first."
    )
    return "\n".join(lines)


def llm_explanation(
    lot_id: str,
    risk_score: float,
    risk_level: str,
    contributions: list[FeatureContribution],
    top_k: int = 3,
    model: str | None = None,
) -> str | None:
    """Optional Anthropic-generated explanation. Returns None if unavailable.

    Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY`` env var.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # type: ignore
    except Exception:
        return None

    try:
        client = anthropic.Anthropic()
        prompt = _build_llm_prompt(lot_id, risk_score, risk_level, contributions, top_k)
        message = client.messages.create(
            model=model or _DEFAULT_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
        text = " ".join(p.strip() for p in parts if p).strip()
        return text or None
    except Exception:
        # Never let an API hiccup break scoring -- fall back to the template.
        return None
