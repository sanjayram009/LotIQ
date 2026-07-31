"""
Train the chilli spoilage-risk regressor.

Design notes
------------
* The intended production model is **XGBoost** (great on small tabular data,
  L1/L2 regularisation to fight overfitting, and native, exact SHAP via
  TreeExplainer). If ``xgboost`` is installed we use it.
* If ``xgboost`` isn't importable (e.g. a locked-down environment where the
  wheel won't build), we transparently fall back to scikit-learn's
  ``HistGradientBoostingRegressor`` -- also a regularised gradient-boosted tree
  ensemble, so the pipeline, metrics and explanations still work end to end.
  The chosen backend is recorded in the saved bundle.
* We save a *bundle* (model + feature order + backend name + training-set
  medians for the explainer baseline + the commodity) so that prediction and
  explanation never have to guess how the model was built.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from lotiq.config import (
    MODEL_FEATURES,
    MODEL_PATH,
    METRICS_PATH,
    RANDOM_SEED,
    SYNTHETIC_CSV,
    TARGET,
    band_for_score,
)
from lotiq.data.preprocessing import build_feature_frame


def _build_model() -> tuple[Any, str]:
    """Return (estimator, backend_name). Prefer XGBoost, fall back to sklearn."""
    try:
        from xgboost import XGBRegressor  # type: ignore

        model = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.5,      # L1
            reg_lambda=2.0,     # L2
            min_child_weight=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            objective="reg:squarederror",
        )
        return model, "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=2.0,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        )
        return model, "sklearn_histgbr"


def _level_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of lots whose predicted risk *band* matches the true band.

    This is the metric a QC manager cares about: not the exact number, but
    whether we put the lot in the right bucket (healthy/moderate/high/critical).
    """
    true_bands = [band_for_score(s).name for s in y_true]
    pred_bands = [band_for_score(s).name for s in y_pred]
    return float(np.mean([t == p for t, p in zip(true_bands, pred_bands)]))


def train(
    data_path: Path = SYNTHETIC_CSV,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    commodity: str = "chilli",
) -> dict:
    """Train, evaluate and persist the model. Returns the metrics dict."""
    df = pd.read_csv(data_path)
    X = build_feature_frame(df, commodity)
    y = df[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model, backend = _build_model()

    # Light cross-validation on the training split for an honest generalisation
    # estimate (negative MAE -> positive).
    cv_mae = -cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )

    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, 100)

    metrics = {
        "backend": backend,
        "commodity": commodity,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": round(float(mean_absolute_error(y_test, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "cv_mae_mean": round(float(cv_mae.mean()), 3),
        "cv_mae_std": round(float(cv_mae.std()), 3),
        "risk_band_accuracy": round(_level_accuracy(y_test, pred), 4),
    }

    # Baseline medians from the FULL feature set: used by the ablation explainer
    # as the "normal lot" reference when SHAP isn't available.
    baseline = {c: float(np.median(X[c])) for c in MODEL_FEATURES}

    bundle = {
        "model": model,
        "backend": backend,
        "features": MODEL_FEATURES,
        "commodity": commodity,
        "baseline": baseline,
        "target": TARGET,
        "version": "0.1.0",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


def _print_metrics(m: dict) -> None:
    print(f"Backend            : {m['backend']}")
    print(f"Train / test rows  : {m['n_train']} / {m['n_test']}")
    print(f"MAE (test)         : {m['mae']:.2f} risk points")
    print(f"RMSE (test)        : {m['rmse']:.2f} risk points")
    print(f"R^2 (test)         : {m['r2']:.3f}")
    print(f"CV MAE (train)     : {m['cv_mae_mean']:.2f} +/- {m['cv_mae_std']:.2f}")
    print(f"Risk-band accuracy : {m['risk_band_accuracy']*100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the chilli risk model.")
    parser.add_argument("-d", "--data", type=str, default=str(SYNTHETIC_CSV))
    parser.add_argument("-o", "--model-out", type=str, default=str(MODEL_PATH))
    args = parser.parse_args()

    metrics = train(data_path=Path(args.data), model_path=Path(args.model_out))
    _print_metrics(metrics)
    print(f"\nSaved model  -> {args.model_out}")
    print(f"Saved metrics-> {METRICS_PATH}")


if __name__ == "__main__":
    main()
