# Release scope and file mapping

The supplied material consisted of four Python scripts and a project report. No two Python files were byte-identical.

Two scripts addressed related feature-importance questions but were not duplicates:

- `第一问（新）.py` performed IQR filtering, SelectKBest feature selection, leave-one-out comparison of random forest and gradient boosting, and final importance analysis.
- `关键因素.py` fit a broader random-forest train/test baseline and ranked all retained features.

They are therefore retained as distinct primary and baseline tasks.

## Task-oriented renaming

| Supplied filename | Public task filename | Purpose |
| --- | --- | --- |
| `第一问（新）.py` | `scripts/analyze_store_demand_drivers.py` | Select factors and compare RF/GB under small-sample validation |
| `关键因素.py` | `scripts/rank_store_feature_importance.py` | Produce a broad random-forest importance baseline |
| `第二问分析.py` | `scripts/train_distance_band_order_forecaster.py` | Train five distance-band XGBoost regressors |
| `网页.py` | `scripts/serve_order_prediction_dashboard.py` | Serve forecasts through a Flask interface |

Reusable logic now lives in `xiaoxiang_analytics/`; the four public scripts are clear command-line entry points.

## Curation changes

- Replaced hard-coded Windows workbook and font paths with command-line options and platform font fallback.
- Moved all training and artifact loading behind functions and explicit entry points.
- Added schema validation and parsing for comma-separated counts and percent strings.
- Fit the scaler on training rows only in the XGBoost task.
- Moved feature selection inside each cross-validation fold to avoid validation leakage.
- Saved one versioned forecasting bundle containing feature order, targets, scaler, and models.
- Added missing HTML templates and converted the Flask script to an application factory.
- Disabled Flask debug mode unless explicitly requested.
- Rounded forecast counts and clipped negative values to zero for dashboard output.
- Added synthetic-data tests and release-scope checks.

## Excluded material

- The source workbook was not supplied and may contain non-public business data.
- The project report is used only to describe project goals and report-recorded findings. It is not redistributed in this code-focused release.
- Generated model artifacts and result files are ignored because they depend on the private workbook.

The curated Git history begins at publication time. It does not reconstruct development commits from March–June 2025.
