# Model Card — LotIQ Chilli Spoilage-Risk Model

A short, structured summary of the pilot model, following the spirit of the
[Model Cards](https://arxiv.org/abs/1810.03993) framework. It is intended to make
the model's scope and limits explicit — especially important because the training
data is synthetic.

## Model details

- **Purpose:** estimate a 0–100 spoilage-risk score for a chilli lot from sensor,
  lab and intake features, and attribute the score to specific features.
- **Type:** gradient-boosted regression tree ensemble.
  - Primary backend: **XGBoost** (`XGBRegressor`).
  - Fallback backend: scikit-learn `HistGradientBoostingRegressor` (used when
    XGBoost isn't installed). The active backend is recorded in `models/metrics.json`.
- **Version:** 0.1.0 (prototype).
- **Output:** continuous `risk_score` in [0, 100], mapped to bands
  Healthy (0–30), Moderate (30–60), High (60–80), Critical (80–100).
- **Explainability:** per-lot SHAP values (or an ablation fallback), surfaced as
  ranked risk factors and a plain-language explanation.

## Intended use

- **Intended:** a decision-support prototype that helps a QC manager triage which
  chilli lots to inspect first, and understand why. A demonstration of the LotIQ
  approach end to end.
- **Out of scope:** any real operational or safety decision. This model has never
  seen real data and must not be used to accept, reject, price, or certify actual
  inventory. It is a prototype, not a validated instrument.

## Factors (features)

| Group  | Feature | Direction |
|--------|---------|-----------|
| Sensor | temperature, humidity, moisture_content, co2_level | higher → riskier |
| Lab    | oleoresin_content, colour_score | lower → riskier |
| Intake | storage_days | higher → riskier |
| Intake | initial_quality_grade (A/B/C) | worse → riskier |

Thresholds encoding "when a feature starts to worry a QC manager" live in
`lotiq.config.CHILLI`.

## Training & evaluation data

- **Source:** fully **synthetic**, produced by a domain-informed simulator
  (`src/lotiq/data/generate.py`). Not real, not proprietary.
- **Size:** ~2,500 lots; 80/20 train/test split; 5-fold CV on the training split.
- **Label construction:** the risk score is a smooth (logistic) function of
  domain-weighted, standardised drivers **plus an unobserved latent factor and
  Gaussian noise**. The unobserved component is intentional — it prevents the
  model from trivially inverting a threshold rule and makes evaluation and
  explanations meaningful (see `docs/architecture.md`).

## Metrics

On held-out synthetic test data (representative run, scikit-learn backend):

| Metric              | Value        |
|---------------------|--------------|
| MAE                 | ~9.3 points  |
| RMSE                | ~12.7 points |
| R²                  | ~0.85        |
| 5-fold CV MAE       | ~9.1 ± 0.4   |
| Risk-band accuracy  | ~71%         |

The gap between CV MAE (~9.1) and test MAE (~9.3) is small, indicating no serious
overfitting. R² is intentionally well below 1.0 because of the injected irreducible
error; a near-1.0 R² on this data would indicate a circular label, not a good model.

## Ethical considerations

- **Automation bias.** A confident-looking score can crowd out human expertise.
  The explanation layer and risk factors exist partly to keep a human in the loop
  rather than deferring blindly to a number.
- **Distribution shift.** Real warehouses, sensors and suppliers will differ from
  the simulator. Deploying without re-training and re-validation on real data
  would be misleading.
- **Consequential errors.** In a real setting, a false "Healthy" could let a
  spoiling lot through, and a false "Critical" could waste inspection effort or
  unfairly flag a supplier. Threshold choices should be revisited with domain
  experts and real cost trade-offs.

## Caveats & limitations

- Trained only on synthetic chilli data; real-world performance is unknown.
- Single commodity. Other commodities need their own specs, data and models.
- The labelling problem for real data (spoilage confirmed days after early signals)
  is unsolved and out of scope for this prototype.
- Lab features are assumed current; a production system must handle stale inputs.
- The dashboard's per-lot risk trend is an illustrative simulation, not a real
  time series.

## Maintenance

Retrain with `python -m lotiq.models.train` after regenerating or replacing the
dataset. Update this card whenever the data, backend, or metrics change.
