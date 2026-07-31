"""
Domain-informed synthetic data generator for the chilli pilot.

WHY SYNTHETIC (and why not SDV)
-------------------------------
LotIQ has no seed operational data yet, so there is nothing for a "learn the
distribution then sample" tool like SDV to learn *from*. What we can encode is
domain knowledge: how chilli degrades in storage. So this generator is a small,
transparent, rule-*informed* simulator, not a black box.

WHY THE LABEL IS NOT A THRESHOLD RULE
-------------------------------------
The risk score is a smooth logistic function of standardised, domain-weighted
drivers, PLUS an unobserved latent factor and Gaussian noise (see
``config.GenerationConfig``). This is deliberate. If ``risk_score`` were, say,
``100 if moisture > 12 else 0``, a tree model would memorise that rule and SHAP
would merely recite it back -- a circular, meaningless demo. By making the label
depend partly on something the model can never see, we create a genuine signal
with irreducible error, so evaluation numbers are honest and explanations vary
smoothly across lots, the way they would on real sensor data.

The physical structure (moisture rises with humidity and time; colour and
oleoresin degrade with heat and time; CO2 tracks respiration) also induces
realistic correlations between features rather than independent noise.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from lotiq.config import (
    CHILLI,
    GENERATION,
    RANDOM_SEED,
    SYNTHETIC_CSV,
    band_for_score,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _standardise(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def generate_dataset(
    n: int | None = None,
    seed: int | None = None,
    commodity: str = "chilli",
) -> pd.DataFrame:
    """Generate a synthetic chilli warehouse dataset.

    Returns a DataFrame with the raw features, ``initial_quality_grade``,
    ``grade_ordinal``, the continuous ``risk_score`` (0-100) target, and a
    convenience ``risk_level`` band label.
    """
    if commodity != "chilli":
        raise NotImplementedError(
            "The pilot generator is specified for chilli only. "
            "Add a CommoditySpec + baselines to extend it."
        )

    n = int(n or GENERATION.n_samples)
    rng = np.random.default_rng(RANDOM_SEED if seed is None else seed)
    spec = CHILLI

    # -- 1. Intake: initial quality grade drives starting colour / oleoresin ----
    grades = rng.choice(["A", "B", "C"], size=n, p=[0.45, 0.40, 0.15])
    grade_num = np.select([grades == "A", grades == "B", grades == "C"], [0, 1, 2])

    colour_base = np.select(
        [grades == "A", grades == "B", grades == "C"],
        [92.0, 82.0, 71.0],
    ) + rng.normal(0, 3.0, n)
    oleoresin_base = np.select(
        [grades == "A", grades == "B", grades == "C"],
        [8.6, 7.2, 5.8],
    ) + rng.normal(0, 0.5, n)
    moisture_base = np.select(
        [grades == "A", grades == "B", grades == "C"],
        [8.8, 9.6, 10.4],
    ) + rng.normal(0, 0.4, n)

    # -- 2. Time in storage ----------------------------------------------------
    storage_days = rng.gamma(shape=2.0, scale=18.0, size=n)
    storage_days = np.clip(storage_days, *spec.ranges["storage_days"])

    # -- 3. Warehouse micro-climate (each lot sits in its own micro-zone) -------
    humidity = rng.normal(60.0, 9.0, n) + rng.normal(0, 3.0, n)
    humidity = np.clip(humidity, *spec.ranges["humidity"])

    temperature = rng.normal(26.0, 3.5, n) + rng.normal(0, 1.5, n)
    temperature = np.clip(temperature, *spec.ranges["temperature"])

    # -- 4. Moisture ingress: grows with humidity exposure over time -----------
    moisture = (
        moisture_base
        + 0.9 * ((humidity - 55.0) / 100.0) * np.sqrt(storage_days)
        + rng.normal(0, 0.5, n)
    )
    moisture = np.clip(moisture, *spec.ranges["moisture_content"])

    # -- 5. Respiration proxy: CO2 rises with time, heat and moisture ----------
    co2 = (
        500.0
        + 6.0 * storage_days * (0.5 + (moisture - 9.0) / 12.0)
        + 14.0 * (temperature - 24.0)
        + rng.normal(0, 60.0, n)
    )
    co2 = np.clip(co2, *spec.ranges["co2_level"])

    # -- 6. Colour fades with time, heat and humidity --------------------------
    colour = (
        colour_base
        - 0.22 * storage_days
        - 0.8 * np.maximum(0.0, temperature - 26.0)
        - 0.05 * np.maximum(0.0, humidity - 60.0)
        - rng.normal(0, 2.0, n)
    )
    colour = np.clip(colour, *spec.ranges["colour_score"])

    # -- 7. Oleoresin (volatiles) degrade with time and heat -------------------
    oleoresin = (
        oleoresin_base
        - 0.020 * storage_days
        - 0.06 * np.maximum(0.0, temperature - 26.0)
        - rng.normal(0, 0.3, n)
    )
    oleoresin = np.clip(oleoresin, *spec.ranges["oleoresin_content"])

    frame = pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "moisture_content": moisture,
            "co2_level": co2,
            "oleoresin_content": oleoresin,
            "colour_score": colour,
            "storage_days": storage_days,
            "initial_quality_grade": grades,
            "grade_ordinal": grade_num,
        }
    )

    # -- 8. Latent risk: smooth function of standardised drivers + noise -------
    driver_sum = np.zeros(n)
    for feature, weight in GENERATION.weights.items():
        driver_sum += weight * _standardise(frame[feature].to_numpy())

    latent = rng.normal(0.0, GENERATION.latent_sigma, n)   # unobserved factor
    noise = rng.normal(0.0, GENERATION.noise_sigma, n)     # measurement noise
    logit = (
        GENERATION.logistic_scale * (driver_sum - GENERATION.logistic_centre)
        + latent
        + noise
    )
    risk_score = 100.0 * _sigmoid(logit)

    frame["risk_score"] = np.round(risk_score, 1)
    frame["risk_level"] = [band_for_score(s).name for s in frame["risk_score"]]
    frame.insert(0, "commodity", commodity)
    frame.insert(0, "lot_id", [f"L-{i:04d}" for i in range(1, n + 1)])

    return frame


def summarise(df: pd.DataFrame) -> str:
    """Return a short human-readable summary of a generated dataset."""
    counts = df["risk_level"].value_counts()
    lines = [
        f"Rows: {len(df)}",
        f"risk_score  mean={df['risk_score'].mean():.1f}  "
        f"std={df['risk_score'].std():.1f}  "
        f"min={df['risk_score'].min():.1f}  max={df['risk_score'].max():.1f}",
        "Risk-band distribution:",
    ]
    for level in ["healthy", "moderate", "high", "critical"]:
        c = int(counts.get(level, 0))
        lines.append(f"  {level:<9} {c:>5}  ({100 * c / len(df):4.1f}%)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic chilli data.")
    parser.add_argument("-n", "--n-samples", type=int, default=None,
                        help="Number of lots to generate.")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed (defaults to config.RANDOM_SEED).")
    parser.add_argument("-o", "--output", type=str, default=str(SYNTHETIC_CSV),
                        help="Output CSV path.")
    args = parser.parse_args()

    df = generate_dataset(n=args.n_samples, seed=args.seed)

    from pathlib import Path
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(summarise(df))
    print(f"\nWrote {len(df)} rows -> {out}")
    print("NOTE: this is SYNTHETIC prototype data, not real operational data.")


if __name__ == "__main__":
    main()
