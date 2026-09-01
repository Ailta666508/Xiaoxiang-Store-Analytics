"""Analytics tools for the Xiaoxiang Supermarket business-case project."""

from .distance_forecasting import load_forecaster, predict_orders

__all__ = ["load_forecaster", "predict_orders"]
__version__ = "0.1.0"
