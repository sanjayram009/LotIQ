# LotIQ — Project Context

Living context document for the LotIQ prototype. Keep it updated as the project
evolves (real data collected, labelling strategy chosen, new commodity added).

> This is an improved version of the original planning doc. Two substantive
> corrections were made to keep the project scientifically honest — see
> [Corrections](#corrections-from-the-original-plan) at the end.

## What LotIQ is

A **software intelligence layer** for spice warehouses. It sits on top of sensor
data and lab records and assigns every lot a live spoilage-risk score (0–100),
tells the QC manager which lots are at risk, and explains why in plain language.
It is **not** a hardware product.

## The problem

Indian spice warehouses lose inventory to preventable spoilage. Post-harvest loss
figures (ICAR-CIPHET) are material — e.g. chilli ≈ 6% — and current monitoring is
largely manual (handheld moisture meters, visual inspection). Even a modest
reduction in spoilage on a large chilli inventory is financially significant.
LotIQ's thesis: **early, explainable, per-lot risk scoring lets QC teams act
before spoilage is locked in.**

## Scope of *this repository*

The repo proves one specific claim end to end:

> *Given warehouse sensor + lot intake data, LotIQ can estimate spoilage risk for
> each lot, explain the risk, and surface it through a usable QC dashboard.*

### Implemented here (chilli pilot)

- Domain-informed **synthetic dataset** generator.
- **XGBoost** risk-scoring model (with a scikit-learn fallback so it always runs).
- **SHAP** explainability (with an ablation fallback), plus a plain-language layer
  and an *optional* Anthropic-LLM explanation.
- **FastAPI** inference service (`/predict-risk`).
- **Streamlit** QC dashboard (warehouse overview + per-lot detail + live scoring).
- A **batch scoring CLI** (`lotiq-score`) for scoring a CSV of lots.
- **Docker** (Dockerfile + compose for API + dashboard) and **GitHub Actions CI**.
- **Tests**, docs (architecture notes + model card), and an architecture diagram.

### Designed but NOT in this repo (production vision)

- Physical sensing: DHT22 (temp/humidity), moisture probe, CO₂ sensor on an
  ESP32 edge device, transmitting over LoRaWAN to a gateway.
- Cloud hosting, authentication, alerting, and a **React** production dashboard.
- Multiple commodities, each with its own model.

These are deliberately out of scope for the prototype — the point is a convincing
end-to-end MVP, not a full production build.

## Commodities and features (domain reference)

The pilot models **chilli**. The broader design anticipates per-commodity feature
sets:

| Commodity | Key quality features (besides moisture/temp/humidity)   |
|-----------|---------------------------------------------------------|
| Chilli    | Colour score, oleoresin content, CO₂                    |
| Turmeric  | Curcumin content, light exposure                        |
| Black pepper | Piperine content, oil content, CO₂                   |
| Ginger    | Gingerol content, ethylene, ventilation                 |
| Cumin     | Essential-oil content, microbial load                   |
| Coriander | Volatile-oil content, foreign matter %                  |

Feature taxonomy used by the model:

- **Sensor** (real-time): temperature, humidity, moisture content, CO₂.
- **Lab** (periodic, entered by QC): oleoresin, colour score (and per-commodity
  analogues like curcumin, piperine).
- **Intake** (fixed at intake): lot ID, commodity, storage time, initial grade.

## Model

- **Current:** one **XGBoost** regressor for chilli. Input = sensor + lab + intake
  features; output = risk score 0–100. L1/L2 regularisation; SHAP for per-lot
  drivers. Falls back to scikit-learn `HistGradientBoostingRegressor` +
  ablation attribution when XGBoost/SHAP aren't installed.
- **Future:** as real per-commodity data grows, a shared-trunk network with
  commodity-specific heads becomes viable. Not needed at prototype scale.

## Data plan

- **Stage 1 (now):** domain-informed **synthetic** data for chilli
  (`src/lotiq/data/generate.py`). ~2,500 samples. Continuous 0–100 label.
- **Stage 2 (real deployment):** collect real temperature/humidity/moisture/CO₂
  from a warehouse partner; QC enters lab values. **Labelling is the open problem**
  (ground-truth spoilage lags the sensor signals — see below).
- **Stage 3:** retrain on real data once enough samples exist per commodity; phase
  out synthetic data; expand to a second commodity.

## Explainability layer

`XGBoost prediction → SHAP top drivers → plain-language explanation`. The
explanation is generated deterministically by default (no API key needed). An
optional path sends the SHAP summary to the Anthropic API for a more fluent
paragraph, e.g.:

```
"Lot 14 has a risk score of 78/100. Moisture is 13.2% (limit 11%) and temperature
 is 34°C (limit 28°C), while oleoresin has dropped to 4.1% (expected >6%).
 Check this lot first: high moisture with heat is the classic mould-risk pattern."
```

## Dashboard

Warehouse health metrics; a colour-coded, risk-sorted lot list; per-lot detail
(risk gauge, sensor readings, top risk factors, recommended action, illustrative
risk trend); and a live "score a new lot" form. Streamlit for the prototype; React
is the eventual production target.

## Known open problems (kept explicit)

1. **Labelling.** Ground truth for risk is confirmed spoilage, which lags the
   early sensor signals by days/weeks. The labelling strategy for real data is
   unsolved.
2. **Stale lab features.** Lab values update weekly; sensors every 15 minutes. The
   model must tolerate stale lab inputs.
3. **Sensor drift.** Cheap sensors degrade in dusty warehouses; recalibration
   strategy is undefined.
4. **No warehouse partner confirmed.** Deployment assumptions are research-based.
5. **Synthetic-to-real gap.** A model trained on synthetic data may transfer
   poorly to real noisy data.
6. **Adoption.** QC managers may not trust an AI score over experience;
   explainability is partly there to address this.

## Corrections from the original plan

Two changes were made to keep the prototype defensible:

1. **Synthetic data is rule-based, not SDV.** SDV *learns a distribution from
   existing data and samples from it* — but there is no seed data to learn from
   yet. The right tool is a transparent, domain-informed **simulator** (encoding
   how chilli actually degrades in storage), which is what
   `src/lotiq/data/generate.py` implements.

2. **The risk label is intentionally not a hard threshold rule.** The original
   sketch derived risk directly from feature thresholds (e.g. "moisture > 12 ⇒
   high risk"). If the label is a deterministic function of the visible features,
   a tree model just memorises the rule and SHAP parrots it back — a circular,
   meaningless result. LotIQ instead makes risk a smooth function of domain-
   weighted drivers **plus an unobserved latent factor and noise**, so the model
   has to genuinely generalise (test R² ≈ 0.85, not 1.0) and the explanations are
   meaningful. See `docs/architecture.md`.
