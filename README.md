# LotIQ 🌶️ — Spoilage-Risk Scoring for Spice Warehouses

[![CI](https://github.com/sanjayram009/LotIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjayram009/LotIQ/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

> Replace `<your-username>/<your-repo>` in the CI badge URL after you push.

LotIQ is a prototype **software intelligence layer** for spice warehouses. Given a
lot's sensor readings (temperature, humidity, moisture, CO₂) and lab/intake data
(oleoresin, colour grade, storage time), it does three things:

1. **Scores** the lot's spoilage risk on a 0–100 scale,
2. **Explains** *why* the score is what it is, in plain language a QC manager can act on, and
3. **Surfaces** it all through a REST API and a QC dashboard.

The pilot commodity is **chilli**. The whole pipeline — data → model → API →
dashboard → tests — runs end to end on your machine after a one-line install.

> ⚠️ **Data disclaimer.** All data in this repository is **synthetic prototype
> data**, produced by a domain-informed simulator (`src/lotiq/data/generate.py`).
> It is **not** real operational data and is **not** proprietary to any company.
> The model demonstrates the *approach*; it must be retrained on real sensor data
> before it means anything operationally. See [Honest limitations](#honest-limitations).

---

## What it looks like

```
LOT-0020   Risk 97/100   🔴 CRITICAL
  Top factors:  ↑ Moisture 11.5%   ↓ Oleoresin 3.8%   ↑ Storage 57 days
  Action:       Immediate QC inspection. Consider isolating or expediting this lot.
  Why:          Moisture is 11.5%, above the safe limit of 11.0%; oleoresin is
                3.8%, below the expected level of 6.0%; storage duration is 57 days ...
```

---

## Architecture

```
 Sensor + lab + intake data
            │
            ▼
   Preprocessing  (clip ranges, encode grade, order features)
            │
            ▼
   XGBoost regressor  ──►  Risk score (0–100)
            │
            ▼
   SHAP attribution  ──►  Top risk drivers per lot
            │
            ▼
   Explanation layer  (plain-language template, optional LLM)
            │
     ┌──────┴───────┐
     ▼              ▼
  FastAPI       Streamlit
   API          QC dashboard
```

A rendered version of this diagram lives in [`docs/architecture.md`](docs/architecture.md).

**Model backend.** The intended production stack is **XGBoost + SHAP**. If those
wheels aren't available in your environment, LotIQ automatically falls back to
scikit-learn's `HistGradientBoostingRegressor` plus a transparent ablation-based
explainer, so the pipeline always runs. The active backend is recorded in
`models/metrics.json`.

---

## Quickstart

Requires Python 3.10+.

```bash
# 1. Clone and enter
git clone https://github.com/sanjayram009/LotIQ.git  LotIQ && cd LotIQ

# 2. Install (editable) + dev tools
python -m pip install -e ".[dev]"

# 3. The repo ships with a trained model and demo data, so you can go straight to:
uvicorn api.main:app --reload        # API  -> http://127.0.0.1:8000/docs
# ...or, in another terminal:
streamlit run dashboard/app.py       # Dashboard -> http://localhost:8501
```

To regenerate everything from scratch (data → model → warehouse snapshot):

```bash
python -m lotiq.data.generate     # writes data/synthetic/chilli_synthetic.csv
python -m lotiq.models.train      # writes models/lotiq_chilli.pkl + metrics.json
python -m lotiq.data.snapshot     # writes data/synthetic/warehouse_snapshot.csv
```

With `make` installed, the shortcuts are `make setup`, `make pipeline`,
`make api`, `make dashboard`, and `make test`.

### Run with Docker

The repo ships a `Dockerfile` and `docker-compose.yml` that run the API and
dashboard together (the trained model and demo data are baked into the image):

```bash
docker compose up --build
# API docs   -> http://localhost:8000/docs
# Dashboard  -> http://localhost:8501
```

### Batch scoring (CLI)

Score a whole CSV of lots without the API:

```bash
lotiq-score lots.csv -o lots_scored.csv
```

It appends `risk_score`, `risk_level`, `recommendation`, the top risk factors, and
an explanation to each row. Missing optional fields fall back to a neutral midpoint,
just like the API.

---

## The API

`POST /predict-risk` (alias `POST /predict`):

**Request**
```json
{
  "lot_id": "L-1842",
  "commodity": "chilli",
  "temperature": 29.4,
  "humidity": 76,
  "moisture_content": 12.8,
  "storage_days": 51,
  "co2_level": 1100,
  "oleoresin_content": 5.2,
  "colour_score": 64,
  "initial_quality_grade": "B"
}
```

**Response** (abridged)
```json
{
  "lot_id": "L-1842",
  "risk_score": 97.9,
  "risk_level": "Critical",
  "recommendation": "Immediate QC inspection. Consider isolating or expediting this lot.",
  "risk_factors": ["High moisture content (12.8%)", "Low oleoresin content (5.2%)", "High humidity (76.0%)"],
  "contributions": [
    {"feature": "moisture_content", "value": 12.8, "contribution": 29.25, "threshold": 11.0, "exceeds_threshold": true}
  ],
  "explanation": "Lot L-1842 has a risk score of 98/100 (Critical). The main drivers are moisture content is 12.8%, above the safe limit of 11.0%; oleoresin content is 5.2%, below the expected level of 6.0%; humidity is 76.0%, above the safe limit of 65.0%."
}
```

Sensor + storage fields are required; lab fields and grade are optional (a missing
field falls back to a neutral midpoint, so a partial reading still scores). Set
`"use_llm": true` to route the explanation through the Anthropic API if you've
configured it (see [Optional LLM explanations](#optional-llm-explanations)).

---

## What the model predicts

**Target:** `risk_score`, a continuous 0–100 spoilage-risk value.

**Features**

| Group   | Feature              | Unit  | Risk direction        |
|---------|----------------------|-------|-----------------------|
| Sensor  | temperature          | °C    | higher → riskier      |
| Sensor  | humidity             | %     | higher → riskier      |
| Sensor  | moisture_content     | %     | higher → riskier      |
| Sensor  | co2_level            | ppm   | higher → riskier      |
| Lab     | oleoresin_content    | %     | lower → riskier       |
| Lab     | colour_score         | /100  | lower → riskier       |
| Intake  | storage_days         | days  | higher → riskier      |
| Intake  | initial_quality_grade| A/B/C | worse grade → riskier |

Risk bands: **Healthy** 0–30 · **Moderate** 30–60 · **High** 60–80 · **Critical** 80–100.

---

## Model performance

Latest run on held-out synthetic test data (backend recorded in `models/metrics.json`):

| Metric                    | Value        |
|---------------------------|--------------|
| MAE                       | ~9.3 points  |
| RMSE                      | ~12.7 points |
| R²                        | ~0.85        |
| 5-fold CV MAE             | ~9.1 ± 0.4   |
| Risk-band accuracy        | ~71%         |

**Why R² is ~0.85 and not ~1.0 — on purpose.** The synthetic label is *not* a
deterministic function of the visible features. It's a smooth function of
domain-weighted drivers **plus an unobserved latent factor and noise**. If the
label were a hard rule on the features, a tree model would simply memorise it and
SHAP would parrot it back — an impressive-looking but circular demo. The injected
irreducible error forces the model to genuinely generalise, which is what makes
the evaluation numbers and the explanations meaningful. (More in
[`docs/architecture.md`](docs/architecture.md).)

---

## Explainability

Every score comes with per-lot attributions:

- **SHAP** (`shap.TreeExplainer`) when the XGBoost backend is active — exact,
  fast, additive contributions in risk-score points.
- **Ablation fallback** otherwise — "starting from a typical lot, how much does
  this lot's value for each feature move the score?" Transparent and easy to
  explain, used when SHAP/XGBoost aren't installed.

Both feed the same explanation layer, which is **threshold-aware**: it only says a
value "crossed the safe limit" when it actually did, otherwise it says the value
is merely higher/lower than a typical lot.

### Optional LLM explanations

By default explanations are generated by a deterministic template — no API key, no
network, no cost. If you set `ANTHROPIC_API_KEY` and install the `llm` extra
(`pip install -e ".[llm]"`), passing `use_llm=true` routes the SHAP summary through
the Anthropic API for a more fluent paragraph. Any failure silently falls back to
the template, so scoring never depends on a network call.

---

## Project structure

```
LotIQ/
├── src/lotiq/
│   ├── config.py            # single source of truth: schema, thresholds, bands
│   ├── cli.py               # batch scoring CLI (lotiq-score)
│   ├── data/
│   │   ├── generate.py      # domain-informed synthetic generator
│   │   ├── preprocessing.py # raw record -> model feature vector
│   │   └── snapshot.py      # build the warehouse snapshot for the dashboard
│   ├── models/
│   │   ├── train.py         # XGBoost (or sklearn) training + metrics
│   │   ├── predict.py       # one-stop scoring entry point
│   │   └── risk.py          # score -> band, colour, recommendation
│   └── explain/
│       ├── attribution.py   # SHAP or ablation contributions
│       └── explainer.py     # contributions -> plain language (+ optional LLM)
├── api/main.py              # FastAPI service
├── dashboard/app.py         # Streamlit QC dashboard
├── notebooks/               # data exploration + model evaluation
├── tests/                   # pytest suite (20 tests)
├── data/synthetic/          # committed synthetic demo data
├── models/                  # committed pilot model + metrics
├── docs/                    # architecture.md, model_card.md, screenshots/
├── Dockerfile               # + docker-compose.yml for API + dashboard
└── .github/workflows/ci.yml # tests + pipeline smoke test on push
```

---

## Testing

```bash
pytest -q
```

The suite covers the generator (shape, ranges, reproducibility, band spread), the
risk-band boundaries, preprocessing, and prediction/attribution sanity (a clearly
bad lot must score much higher than a clearly good one, contribution signs must be
directional, etc.).

---

## Honest limitations

This is a prototype. Known gaps, kept explicit on purpose:

- **Synthetic data.** The model has never seen a real chilli lot. Performance on
  real, noisy sensor data is unknown and will differ.
- **The labelling problem is unsolved.** Real spoilage is confirmed days/weeks
  after the early sensor signals. How to label real data is an open question this
  prototype does not answer.
- **Stale lab features.** Lab values update weekly; sensors every 15 minutes. A
  production system needs a strategy for gracefully stale inputs.
- **Single commodity.** Only chilli is modelled. The schema generalises, but each
  new commodity needs its own domain spec, data and (probably) model.
- **Illustrative risk trend.** The dashboard's per-lot trend line is a seeded
  simulation, not a real time series — a placeholder until sensor history exists.

---

## Roadmap

- Per-commodity specs and models (turmeric, pepper, …) behind the same interface.
- Real sensor ingestion + a defensible labelling strategy.
- Calibrated probabilities and confidence intervals on the score.
- A React dashboard for production (Streamlit is the prototype UI).

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — pipeline diagram, the
  data-generating process, and the model/explainability design rationale.
- [`docs/model_card.md`](docs/model_card.md) — intended use, metrics, ethical
  considerations, and limitations.
- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — the living project context.

## License

MIT — see [LICENSE](LICENSE).
