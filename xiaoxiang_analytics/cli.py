"""Command-line entry points for analysis, training, and serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data import read_project_sheet
from .demand_drivers import compare_driver_models, fit_random_forest_baseline
from .distance_forecasting import save_forecaster, train_distance_forecaster
from .webapp import create_app


def _data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="Path to the source .xlsx workbook")
    parser.add_argument("--sheet", default="Sheet2", help="Worksheet name (default: Sheet2)")
    parser.add_argument("--output-dir", default="results", help="Directory for generated artifacts")


def _save_importance_plot(table, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font_candidates = [
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    cjk_font_path = next((candidate for candidate in font_candidates if candidate.is_file()), None)
    cjk_font = FontProperties(fname=str(cjk_font_path)) if cjk_font_path else None

    figure_height = max(5, min(14, 0.38 * len(table)))
    figure, axis = plt.subplots(figsize=(11, figure_height))
    ordered = table.sort_values("importance", ascending=True)
    axis.barh(ordered["feature"], ordered["importance"], color="#4C9FD8")
    axis.set_xlabel("Feature importance")
    axis.set_ylabel("Feature")
    axis.set_title(title)
    if cjk_font is not None:
        for label in axis.get_yticklabels():
            label.set_fontproperties(cjk_font)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def analyze_demand_drivers_main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare random forest and gradient boosting for store demand drivers."
    )
    _data_arguments(parser)
    parser.add_argument("--k-best", type=int, default=10)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = read_project_sheet(args.data, args.sheet)
    result = compare_driver_models(frame, k_best=args.k_best)

    result.feature_importance.to_csv(output / "demand_driver_importance.csv", index=False)
    _save_importance_plot(
        result.feature_importance,
        output / "demand_driver_importance.png",
        "Store demand-driver importance",
    )
    joblib.dump(result.estimator, output / "demand_driver_model.joblib")
    summary = {
        "best_model": result.best_model,
        "cross_validated_mse": result.cross_validated_mse,
        "training_metrics": result.training_metrics,
        "rows_used": result.rows_used,
        "outliers_removed": result.outliers_removed,
        "selected_features": result.feature_importance["feature"].tolist(),
    }
    (output / "demand_driver_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def rank_feature_importance_main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the broad random-forest feature-importance baseline."
    )
    _data_arguments(parser)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = read_project_sheet(args.data, args.sheet)
    result = fit_random_forest_baseline(frame)

    result.feature_importance.to_csv(output / "baseline_feature_importance.csv", index=False)
    _save_importance_plot(
        result.feature_importance,
        output / "baseline_feature_importance.png",
        "Random-forest feature-importance baseline",
    )
    joblib.dump(result.estimator, output / "baseline_random_forest.joblib")
    summary = {"test_mse": result.test_mse, "rows_used": result.rows_used}
    (output / "baseline_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def train_distance_forecaster_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train five XGBoost models for delivery-distance order bands."
    )
    _data_arguments(parser)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = read_project_sheet(args.data, args.sheet)
    bundle, metrics = train_distance_forecaster(frame)
    artifact = save_forecaster(bundle, output / "distance_order_forecaster.joblib")
    (output / "distance_forecast_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**metrics, "artifact": str(artifact)}, ensure_ascii=False, indent=2))


def serve_prediction_dashboard_main() -> None:
    parser = argparse.ArgumentParser(description="Serve the order-prediction dashboard.")
    parser.add_argument(
        "--model",
        default="results/distance_order_forecaster.joblib",
        help="Path to the trained forecasting artifact",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.model)
    app.run(host=args.host, port=args.port, debug=args.debug)
