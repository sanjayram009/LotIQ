"""
LotIQ REST API (FastAPI).

Thin transport layer: request validation in, tidy JSON out. All scoring logic
lives in ``lotiq.models.predict.score_record`` (which is unit-tested), so the
API and the dashboard behave identically.

Run it:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make `lotiq` importable even if the package wasn't `pip install -e .`-ed.
try:
    from lotiq import __version__
    from lotiq.models.predict import load_bundle, score_record
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lotiq import __version__
    from lotiq.models.predict import load_bundle, score_record

app = FastAPI(
    title="LotIQ API",
    version=__version__,
    description="AI spoilage-risk scoring for spice-warehouse lots (chilli pilot).",
)

# Loaded once at import; reused across requests.
_BUNDLE: dict | None = None


def _get_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_bundle()
    return _BUNDLE


class LotRequest(BaseModel):
    """A single lot to score. Sensor + lab + intake readings.

    Optional fields default to the commodity midpoint if omitted, so a partial
    payload (e.g. lab values not entered yet) still returns a score.
    """

    lot_id: str = Field("UNKNOWN", examples=["L-0007"])
    commodity: str = Field("chilli", examples=["chilli"])
    temperature: Optional[float] = Field(None, description="deg C", examples=[31.5])
    humidity: Optional[float] = Field(None, description="% RH", examples=[72.0])
    moisture_content: Optional[float] = Field(None, description="%", examples=[12.1])
    co2_level: Optional[float] = Field(None, description="ppm", examples=[980])
    oleoresin_content: Optional[float] = Field(None, description="%", examples=[5.4])
    colour_score: Optional[float] = Field(None, description="/100", examples=[66])
    storage_days: Optional[float] = Field(None, description="days", examples=[52])
    initial_quality_grade: str = Field("B", examples=["A", "B", "C"])
    use_llm: bool = Field(False, description="Use the optional LLM explanation if configured.")


class Contribution(BaseModel):
    feature: str
    label: str
    value: float
    unit: str
    contribution: float
    direction: int
    threshold: float
    exceeds_threshold: bool


class LotResponse(BaseModel):
    lot_id: str
    commodity: str
    risk_score: float
    risk_level: str
    colour: str
    emoji: str
    recommendation: str
    risk_factors: list[str]
    protective_factors: list[str]
    contributions: list[Contribution]
    explanation: str


@app.get("/health")
def health() -> dict:
    """Liveness + model status."""
    try:
        bundle = _get_bundle()
        return {
            "status": "ok",
            "model_loaded": True,
            "backend": bundle.get("backend"),
            "commodity": bundle.get("commodity"),
            "version": __version__,
        }
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False, "version": __version__}


@app.post("/predict-risk", response_model=LotResponse)
def predict_risk(lot: LotRequest) -> dict:
    """Score a single lot and explain the result."""
    try:
        bundle = _get_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    record = lot.model_dump()
    use_llm = bool(record.pop("use_llm", False))
    return score_record(record, bundle=bundle, use_llm=use_llm)


# Convenience alias.
@app.post("/predict", response_model=LotResponse, include_in_schema=False)
def predict_alias(lot: LotRequest) -> dict:
    return predict_risk(lot)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
