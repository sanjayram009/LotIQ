"""
Batch scoring CLI.

Score a whole CSV of lots at once and write an annotated copy. Useful for
scoring an exported warehouse manifest without standing up the API.

    lotiq-score lots.csv                 # -> lots_scored.csv
    lotiq-score lots.csv -o out.csv --top-k 3

The input CSV needs the feature columns (temperature, humidity, moisture_content,
storage_days at minimum; co2_level, oleoresin_content, colour_score,
initial_quality_grade, lot_id, commodity are used if present). Missing optional
fields fall back to a neutral midpoint, exactly like the API.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lotiq.config import MODEL_PATH
from lotiq.models.predict import load_bundle, score_record

# Columns we will pass through to the scorer if they exist in the input.
_KNOWN = [
    "lot_id", "commodity", "temperature", "humidity", "moisture_content",
    "co2_level", "oleoresin_content", "colour_score", "storage_days",
    "initial_quality_grade",
]


def score_frame(df: pd.DataFrame, model_path: str | Path = MODEL_PATH,
                top_k: int = 3) -> pd.DataFrame:
    """Return a copy of `df` with risk columns appended."""
    bundle = load_bundle(str(model_path))
    present = [c for c in _KNOWN if c in df.columns]

    out_rows = []
    for _, row in df.iterrows():
        record = {c: row[c] for c in present}
        scored = score_record(record, bundle=bundle, top_k=top_k)
        factors = scored["risk_factors"]
        enriched = {
            "risk_score": scored["risk_score"],
            "risk_level": scored["risk_level"],
            "recommendation": scored["recommendation"],
            "explanation": scored["explanation"],
        }
        for i in range(top_k):
            enriched[f"top_factor_{i + 1}"] = factors[i] if i < len(factors) else ""
        out_rows.append(enriched)

    enriched_df = pd.DataFrame(out_rows, index=df.index)
    return pd.concat([df, enriched_df], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score a CSV of lots.")
    parser.add_argument("input", type=str, help="Input CSV of lots.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV (default: <input>_scored.csv).")
    parser.add_argument("-k", "--top-k", type=int, default=3,
                        help="Number of top risk factors to include.")
    parser.add_argument("-m", "--model", type=str, default=str(MODEL_PATH),
                        help="Path to the trained model bundle.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    scored = score_frame(df, model_path=args.model, top_k=args.top_k)

    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem + "_scored.csv"
    )
    scored.to_csv(out_path, index=False)

    counts = scored["risk_level"].value_counts()
    print(f"Scored {len(scored)} lots -> {out_path}")
    print("Risk levels:", {k: int(v) for k, v in counts.items()})


if __name__ == "__main__":
    main()
