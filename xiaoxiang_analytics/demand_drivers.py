"""Store-demand driver analysis for the first business task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .constants import BASELINE_EXCLUSIONS, DRIVER_ANALYSIS_EXCLUSIONS, TOTAL_DAILY_ORDERS
from .data import coerce_numeric_like_columns, require_columns


@dataclass
class DriverAnalysisResult:
    estimator: Pipeline
    best_model: str
    cross_validated_mse: dict[str, float]
    training_metrics: dict[str, float]
    feature_importance: pd.DataFrame
    rows_used: int
    outliers_removed: int


@dataclass
class BaselineResult:
    estimator: Pipeline
    test_mse: float
    feature_importance: pd.DataFrame
    rows_used: int


def filter_iqr_outliers(
    frame: pd.DataFrame,
    column: str = TOTAL_DAILY_ORDERS,
    threshold: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into retained rows and target outliers using the IQR rule."""

    require_columns(frame, [column])
    numeric_target = pd.to_numeric(frame[column], errors="coerce")
    working = frame.assign(**{column: numeric_target}).dropna(subset=[column])
    q1 = working[column].quantile(0.25)
    q3 = working[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    mask = working[column].between(lower, upper)
    return working.loc[mask].reset_index(drop=True), working.loc[~mask].reset_index(drop=True)


def _prepare_xy(
    frame: pd.DataFrame,
    exclusions: list[str],
    *,
    remove_outliers: bool,
) -> tuple[pd.DataFrame, pd.Series, int]:
    require_columns(frame, [TOTAL_DAILY_ORDERS])
    if remove_outliers:
        cleaned, outliers = filter_iqr_outliers(frame)
    else:
        cleaned = frame.dropna(subset=[TOTAL_DAILY_ORDERS]).reset_index(drop=True)
        outliers = frame.iloc[0:0]

    feature_columns = [column for column in cleaned.columns if column not in exclusions]
    if not feature_columns:
        raise ValueError("No candidate features remain after applying the exclusion list.")

    x = coerce_numeric_like_columns(cleaned[feature_columns])
    y = pd.to_numeric(cleaned[TOTAL_DAILY_ORDERS], errors="raise")
    return x, y, len(outliers)


def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def _selected_importance(estimator: Pipeline) -> pd.DataFrame:
    names = estimator.named_steps["preprocess"].get_feature_names_out()
    selector = estimator.named_steps.get("select")
    if selector is not None:
        names = names[selector.get_support()]
    model = estimator.named_steps["model"]
    importance = model.feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compare_driver_models(
    frame: pd.DataFrame,
    *,
    k_best: int = 10,
    n_estimators: int = 100,
    random_state: int = 42,
) -> DriverAnalysisResult:
    """Compare RF and GB with leakage-safe LOOCV, then fit the better model."""

    x, y, outlier_count = _prepare_xy(
        frame, DRIVER_ANALYSIS_EXCLUSIONS, remove_outliers=True
    )
    if len(x) < 4:
        raise ValueError("At least four retained rows are required for model comparison.")
    selected_count = min(k_best, x.shape[1])

    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=random_state,
        ),
    }

    pipelines: dict[str, Pipeline] = {}
    scores: dict[str, float] = {}
    for name, model in candidates.items():
        pipeline = Pipeline(
            [
                ("preprocess", _build_preprocessor(x)),
                ("select", SelectKBest(score_func=f_regression, k=selected_count)),
                ("model", model),
            ]
        )
        cv_scores = cross_val_score(
            pipeline,
            x,
            y,
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
            n_jobs=None,
        )
        pipelines[name] = pipeline
        scores[name] = float(-cv_scores.mean())

    best_name = min(scores, key=scores.get)
    best = pipelines[best_name].fit(x, y)
    predictions = best.predict(x)
    mse = float(mean_squared_error(y, predictions))
    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }
    return DriverAnalysisResult(
        estimator=best,
        best_model=best_name,
        cross_validated_mse=scores,
        training_metrics=metrics,
        feature_importance=_selected_importance(best),
        rows_used=len(x),
        outliers_removed=outlier_count,
    )


def fit_random_forest_baseline(
    frame: pd.DataFrame,
    *,
    n_estimators: int = 500,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BaselineResult:
    """Fit the original task's broad random-forest importance baseline."""

    x, y, _ = _prepare_xy(frame, BASELINE_EXCLUSIONS, remove_outliers=False)
    if len(x) < 5:
        raise ValueError("At least five rows are required for a train/test baseline.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    estimator = Pipeline(
        [
            ("preprocess", _build_preprocessor(x)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    ).fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    return BaselineResult(
        estimator=estimator,
        test_mse=float(mean_squared_error(y_test, predictions)),
        feature_importance=_selected_importance(estimator),
        rows_used=len(x),
    )
