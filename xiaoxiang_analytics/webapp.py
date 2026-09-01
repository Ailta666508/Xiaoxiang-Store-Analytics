"""Flask application factory for interactive order prediction."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from .distance_forecasting import load_forecaster, predict_orders


def create_app(model_path: str | Path) -> Flask:
    """Create an app around one persisted distance-band forecasting bundle."""

    app = Flask(__name__)
    app.config["FORECAST_BUNDLE"] = load_forecaster(model_path)

    @app.route("/", methods=["GET", "POST"])
    def index():
        bundle = app.config["FORECAST_BUNDLE"]
        features = bundle["features"]
        if request.method == "GET":
            return render_template("index.html", features=features)

        try:
            input_data = {feature: request.form[feature] for feature in features}
            predictions = predict_orders(bundle, input_data)
        except (KeyError, TypeError, ValueError) as exc:
            return render_template("index.html", features=features, error=str(exc)), 400
        return render_template(
            "result.html", predictions=predictions, input_data=input_data
        )

    return app
