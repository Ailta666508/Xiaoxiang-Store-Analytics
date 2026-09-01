from __future__ import annotations

from xiaoxiang_analytics.constants import (
    DISTANCE_FORECAST_FEATURES,
    DISTANCE_ORDER_TARGETS,
)
from xiaoxiang_analytics.distance_forecasting import (
    load_forecaster,
    predict_orders,
    save_forecaster,
)
from xiaoxiang_analytics.webapp import create_app


def test_distance_forecaster_predicts_all_bands(forecast_bundle, project_frame) -> None:
    bundle, metrics = forecast_bundle
    input_data = project_frame.iloc[0][DISTANCE_FORECAST_FEATURES].to_dict()
    predictions = predict_orders(bundle, input_data)

    assert metrics["overall_mse"] >= 0
    assert set(metrics["per_target_mse"]) == set(DISTANCE_ORDER_TARGETS)
    assert set(DISTANCE_ORDER_TARGETS).issubset(predictions)
    assert predictions["站日均订单量(预测总和)"] == sum(
        predictions[target] for target in DISTANCE_ORDER_TARGETS
    )


def test_saved_artifact_drives_web_form(
    forecast_bundle, project_frame, tmp_path
) -> None:
    bundle, _ = forecast_bundle
    artifact = save_forecaster(bundle, tmp_path / "forecaster.joblib")
    assert load_forecaster(artifact)["artifact_version"] == 1

    app = create_app(artifact)
    client = app.test_client()
    assert client.get("/").status_code == 200

    form = {
        feature: str(project_frame.iloc[0][feature])
        for feature in DISTANCE_FORECAST_FEATURES
    }
    response = client.post("/", data=form)
    assert response.status_code == 200
    assert "Predicted daily orders" in response.get_data(as_text=True)
