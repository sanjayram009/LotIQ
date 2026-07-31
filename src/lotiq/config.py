"""
Central configuration for LotIQ.

This module is the single source of truth for:
  * filesystem paths (robust to the current working directory),
  * the feature schema (which columns are sensor / lab / intake features),
  * the domain thresholds for the pilot commodity (chilli),
  * the risk-band definitions used across the API and dashboard, and
  * the coefficients of the synthetic data-generating process.

Keeping all of this in one place means the generator, the model, the API and
the dashboard never disagree about what a feature is or where a risk band
starts. If you want to tune the prototype, tune it here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# config.py lives at <repo>/src/lotiq/config.py, so the repo root is 3 levels up.
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR: Path = DATA_DIR / "synthetic"
MODELS_DIR: Path = ROOT_DIR / "models"

# Default artefact locations for the pilot commodity.
SYNTHETIC_CSV: Path = SYNTHETIC_DATA_DIR / "chilli_synthetic.csv"
WAREHOUSE_SNAPSHOT_CSV: Path = SYNTHETIC_DATA_DIR / "warehouse_snapshot.csv"
MODEL_PATH: Path = MODELS_DIR / "lotiq_chilli.pkl"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"

RANDOM_SEED: int = 42


# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #
# Sensor features arrive continuously (every ~15 min in the production design).
SENSOR_FEATURES: list[str] = ["temperature", "humidity", "moisture_content", "co2_level"]

# Lab features are measured periodically by the QC team and entered by hand.
LAB_FEATURES: list[str] = ["oleoresin_content", "colour_score"]

# Intake features are fixed when a lot enters the warehouse. `storage_days`
# is derived from the intake date and therefore lives with the intake group.
INTAKE_FEATURES: list[str] = ["storage_days", "initial_quality_grade"]

# `initial_quality_grade` is categorical (A/B/C). We one-hot encode it into
# an ordinal helper `grade_ordinal` for the model so the whole feature vector
# is numeric and framework-agnostic.
CATEGORICAL_FEATURES: list[str] = ["initial_quality_grade"]

# The exact, ordered list of columns fed to the model. Order matters because
# some explainers key contributions by position.
MODEL_FEATURES: list[str] = [
    "storage_days",
    "temperature",
    "humidity",
    "moisture_content",
    "co2_level",
    "oleoresin_content",
    "colour_score",
    "grade_ordinal",
]

TARGET: str = "risk_score"

# Human-readable labels for explanations and dashboards.
FEATURE_LABELS: dict[str, str] = {
    "storage_days": "Storage duration",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "moisture_content": "Moisture content",
    "co2_level": "CO2 level",
    "oleoresin_content": "Oleoresin content",
    "colour_score": "Colour score",
    "grade_ordinal": "Initial quality grade",
}

FEATURE_UNITS: dict[str, str] = {
    "storage_days": "days",
    "temperature": "\u00b0C",
    "humidity": "%",
    "moisture_content": "%",
    "co2_level": "ppm",
    "oleoresin_content": "%",
    "colour_score": "/100",
    "grade_ordinal": "",
}


# --------------------------------------------------------------------------- #
# Commodity domain specification (pilot: chilli)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CommoditySpec:
    """Domain knowledge for a single commodity.

    `thresholds` encode the value at which a feature starts to *worry* a QC
    manager. For "higher is worse" features (moisture, temperature, humidity,
    co2, storage_days) the threshold is an upper limit. For "higher is better"
    features (oleoresin, colour_score) it is a lower limit. `direction` records
    which way is risky so the explanation layer can phrase things correctly.
    """

    name: str
    thresholds: dict[str, float]
    # +1 means "higher value => higher risk", -1 means "higher value => lower risk".
    direction: dict[str, int]
    # Plausible physical ranges, used to clip generated / user-supplied values.
    ranges: dict[str, tuple[float, float]]


CHILLI = CommoditySpec(
    name="chilli",
    thresholds={
        "moisture_content": 11.0,   # % — above this, mould risk climbs (ICAR guidance ~11%)
        "temperature": 28.0,        # deg C — cold-chain style comfort ceiling for dry storage
        "humidity": 65.0,           # % RH
        "co2_level": 900.0,         # ppm — proxy for respiration / microbial activity
        "storage_days": 45.0,       # days — quality erosion becomes material
        "oleoresin_content": 6.0,   # % — below this, quality/value has degraded
        "colour_score": 70.0,       # /100 — below this, ASTA colour has faded
    },
    direction={
        "moisture_content": +1,
        "temperature": +1,
        "humidity": +1,
        "co2_level": +1,
        "storage_days": +1,
        "oleoresin_content": -1,
        "colour_score": -1,
    },
    ranges={
        "moisture_content": (6.0, 20.0),
        "temperature": (12.0, 42.0),
        "humidity": (30.0, 95.0),
        "co2_level": (400.0, 2500.0),
        "storage_days": (0.0, 120.0),
        "oleoresin_content": (2.0, 12.0),
        "colour_score": (20.0, 100.0),
    },
)

COMMODITIES: dict[str, CommoditySpec] = {"chilli": CHILLI}

GRADE_TO_ORDINAL: dict[str, int] = {"A": 0, "B": 1, "C": 2}
ORDINAL_TO_GRADE: dict[int, str] = {v: k for k, v in GRADE_TO_ORDINAL.items()}


# --------------------------------------------------------------------------- #
# Risk bands
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskBand:
    name: str
    lower: float          # inclusive
    upper: float          # exclusive (except the top band, which is inclusive)
    colour: str           # hex, for charts
    emoji: str
    label: str            # short status word for tables


RISK_BANDS: list[RiskBand] = [
    RiskBand("healthy", 0.0, 30.0, "#2e9e5b", "\U0001F7E2", "Healthy"),
    RiskBand("moderate", 30.0, 60.0, "#e6b800", "\U0001F7E1", "Moderate"),
    RiskBand("high", 60.0, 80.0, "#e67300", "\U0001F7E0", "High"),
    RiskBand("critical", 80.0, 100.01, "#d13438", "\U0001F534", "Critical"),
]


def band_for_score(score: float) -> RiskBand:
    """Return the RiskBand a 0-100 score falls into."""
    for band in RISK_BANDS:
        if band.lower <= score < band.upper:
            return band
    # Fallback for out-of-range values (should not happen after clipping).
    return RISK_BANDS[-1] if score >= 100 else RISK_BANDS[0]


# --------------------------------------------------------------------------- #
# Synthetic data-generating process
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenerationConfig:
    """Coefficients for the domain-informed synthetic generator.

    The label is deliberately NOT a hard threshold rule on the features. It is a
    smooth (logistic) function of standardised, domain-weighted drivers, plus an
    UNOBSERVED latent factor and Gaussian noise. That design matters:

      * If the risk label were a deterministic function of the visible features,
        a tree model would simply memorise the rule and SHAP would just echo it
        back -- impressive looking, but circular and scientifically empty.
      * By injecting an unobserved `latent_sigma` term (think: supplier-specific
        microbial seeding we never measure) and `noise_sigma`, we create a
        genuine, learnable-but-imperfect signal. The model has to *generalise*,
        evaluation error is non-trivial and honest, and SHAP attributions vary
        smoothly across lots the way they would on real data.

    `weights` are on standardised drivers (see generate.py). Positive weight =>
    pushes risk up; negative => pushes risk down.
    """

    n_samples: int = 2500

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "moisture_content": 1.35,
            "humidity": 0.85,
            "temperature": 0.80,
            "storage_days": 0.70,
            "co2_level": 0.55,
            "oleoresin_content": -0.75,   # higher oleoresin => fresher => lower risk
            "colour_score": -0.65,        # higher colour => better => lower risk
        }
    )

    # Spread of the unobserved latent risk factor (in logit units).
    latent_sigma: float = 0.55
    # Observation noise on the logit (in logit units).
    noise_sigma: float = 0.40
    # Logistic steepness and centre applied to the driver sum before scaling to
    # 0-100. `scale` < 1 compresses the extremes (fewer 0s and 100s); a positive
    # `centre` shifts the baseline toward lower risk, so most lots read healthy
    # and only a minority are critical -- as in a real, reasonably-run warehouse.
    logistic_scale: float = 0.60
    logistic_centre: float = 0.35


GENERATION = GenerationConfig()
