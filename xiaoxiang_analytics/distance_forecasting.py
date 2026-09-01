"""Five-distance-band order forecasting and artifact persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .constants import (
    COUNT_FEATURES,
    DISTANCE_FORECAST_FEATURES,
    DISTANCE_ORDER_TARGETS,
    PERCENTAGE_FEATURES,
)
from .data import normalize_known_numeric_columns, parse_number, require_columns


def prepare_forecasting_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and clean features and targets used by the distance models."""

    required = [*DISTANCE_FORECAST_FEATURES, *DISTANCE_ORDER_TARGETS]
    require_columns(frame, required)
    cleaned = normalize_known_numeric_columns(
        frame,
        [*COUNT_FEATURES, *PERCENTAGE_FEATURES, *DISTANCE_ORDER_TARGETS],
        PERCENTAGE_FEATURES,
    )
    cleaned = cleaned.dropna(subset=required).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No complete rows remain after numeric cleaning.")
    return cleaned[DISTANCE_FORECAST_FEATURES], cleaned[DISTANCE_ORDER_TARGETS]


def train_distance_forecaster(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 50,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train one XGBoost regressor per delivery-distance target."""

    from xgboost import XGBRegressor

    x, y = prepare_forecasting_data(frame)
    if len(x) < 5:
        raise ValueError("At least five complete rows are required for train/test evaluation.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = []
    predictions = []
    for target in DISTANCE_ORDER_TARGETS:
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="rmse",
            objective="reg:squarederror",
            n_jobs=1,
        )
        model.fit(x_train_scaled, y_train[target])
        models.append(model)
        predictions.append(model.predict(x_test_scaled))

    matrix = np.column_stack(predictions)
    per_target = mean_squared_error(y_test, matrix, multioutput="raw_values")
    metrics = {
        "overall_mse": float(mean_squared_error(y_test, matrix)),
        "per_target_mse": {
            target: float(value)
            for target, value in zip(DISTANCE_ORDER_TARGETS, per_target, strict=True)
        },
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    bundle = {
        "artifact_version": 1,
        "features": list(DISTANCE_FORECAST_FEATURES),
        "targets": list(DISTANCE_ORDER_TARGETS),
        "scaler": scaler,
        "models": models,
    }
    return bundle, metrics


def predict_orders(bundle: Mapping[str, Any], input_data: Mapping[str, object]) -> dict[str, int]:
    """Predict rounded, non-negative orders for one prospective store."""

    features = list(bundle["features"])
    missing = [feature for feature in features if feature not in input_data]
    if missing:
        raise ValueError("Missing prediction features: " + ", ".join(missing))

    values = []
    percentage_set = set(PERCENTAGE_FEATURES)
    for feature in features:
        values.append(parse_number(input_data[feature], percentage=feature in percentage_set))
    input_frame = pd.DataFrame([values], columns=features)
    scaled = bundle["scaler"].transform(input_frame)
    raw = np.array([model.predict(scaled)[0] for model in bundle["models"]])
    rounded = np.maximum(np.rint(raw), 0).astype(int)
    result = {
        target: int(value)
        for target, value in zip(bundle["targets"], rounded, strict=True)
    }
    result["站日均订单量(预测总和)"] = int(rounded.sum())
    return result


def save_forecaster(bundle: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(bundle), destination)
    return destination


def load_forecaster(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact}")
    bundle = joblib.load(artifact)
    if bundle.get("artifact_version") != 1:
        raise ValueError("Unsupported model artifact version.")
    return bundle
