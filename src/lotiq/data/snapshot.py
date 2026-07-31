"""
Build a 'warehouse snapshot': a set of lots, each scored by the trained model,
written to CSV. The dashboard reads this to render the warehouse health view
without needing to re-score on every page load.

This is separate from the training data on purpose -- it's a fresh, unseen draw
from the same generator, so the dashboard shows the model generalising to lots
it has never trained on.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lotiq.config import SENSOR_FEATURES, WAREHOUSE_SNAPSHOT_CSV
from lotiq.data.generate import generate_dataset
from lotiq.models.predict import load_bundle, score_record


def build_snapshot(n: int = 40, seed: int = 7) -> pd.DataFrame:
    """Generate `n` fresh lots and attach model predictions."""
    lots = generate_dataset(n=n, seed=seed)
    bundle = load_bundle()

    rows = []
    for _, lot in lots.iterrows():
        record = {
            "lot_id": lot["lot_id"],
            "commodity": lot["commodity"],
            "temperature": lot["temperature"],
            "humidity": lot["humidity"],
            "moisture_content": lot["moisture_content"],
            "co2_level": lot["co2_level"],
            "oleoresin_content": lot["oleoresin_content"],
            "colour_score": lot["colour_score"],
            "storage_days": lot["storage_days"],
            "initial_quality_grade": lot["initial_quality_grade"],
        }
        scored = score_record(record, bundle=bundle)
        factors = scored["risk_factors"]
        rows.append(
            {
                **record,
                "risk_score": scored["risk_score"],
                "risk_level": scored["risk_level"],
                "emoji": scored["emoji"],
                "recommendation": scored["recommendation"],
                "top_factor_1": factors[0] if len(factors) > 0 else "",
                "top_factor_2": factors[1] if len(factors) > 1 else "",
                "top_factor_3": factors[2] if len(factors) > 2 else "",
                "explanation": scored["explanation"],
            }
        )

    df = pd.DataFrame(rows)
    # Round sensor columns for a tidy display.
    for col in SENSOR_FEATURES + ["oleoresin_content", "colour_score", "storage_days"]:
        df[col] = df[col].round(1)
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the warehouse snapshot CSV.")
    parser.add_argument("-n", "--n-lots", type=int, default=40)
    parser.add_argument("-s", "--seed", type=int, default=7)
    parser.add_argument("-o", "--output", type=str, default=str(WAREHOUSE_SNAPSHOT_CSV))
    args = parser.parse_args()

    df = build_snapshot(n=args.n_lots, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    counts = df["risk_level"].value_counts()
    print(f"Wrote {len(df)} lots -> {out}")
    print("Risk levels:", {k: int(v) for k, v in counts.items()})


if __name__ == "__main__":
    main()
