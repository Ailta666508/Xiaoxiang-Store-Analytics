from __future__ import annotations

from xiaoxiang_analytics.demand_drivers import (
    compare_driver_models,
    fit_random_forest_baseline,
)


def test_driver_model_comparison(project_frame) -> None:
    result = compare_driver_models(project_frame, k_best=6, n_estimators=12)

    assert result.best_model in {"random_forest", "gradient_boosting"}
    assert set(result.cross_validated_mse) == {"random_forest", "gradient_boosting"}
    assert result.rows_used == len(project_frame)
    assert 1 <= len(result.feature_importance) <= 6
    assert result.feature_importance["importance"].sum() > 0


def test_random_forest_baseline(project_frame) -> None:
    result = fit_random_forest_baseline(project_frame, n_estimators=12)

    assert result.test_mse >= 0
    assert result.rows_used == len(project_frame)
    assert not result.feature_importance.empty
