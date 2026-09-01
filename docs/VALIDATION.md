# Release validation

Validation was completed on 2026-09-01 before publication.

## Tested environment

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| pandas | 3.0.5 |
| NumPy | 2.5.2 |
| scikit-learn | 1.9.0 |
| XGBoost | 3.4.1 |
| Flask | 3.1.3 |
| Matplotlib | 3.11.1 |
| openpyxl | 3.1.5 |
| pytest | 9.1.1 |

On macOS, XGBoost also required the Homebrew `libomp` runtime. Other operating systems may provide OpenMP differently.

## Automated checks

```text
6 passed
Release verification passed (4 source mappings; 16 Python files).
```

The checks cover:

- small-sample demand-driver model comparison;
- random-forest baseline ranking;
- five-band XGBoost training and metrics;
- saved-artifact round trip and single-store inference;
- Flask GET and POST routes;
- absence of private workbooks, reports, generated models, hard-coded Windows paths, and unconditional debug mode;
- one-to-one source-file mapping and no byte-identical Python files in the curated release.

## Command-line smoke run

All three data-processing entry points were executed against a 24-row synthetic workbook using worksheet `Sheet2`. They produced:

- demand-driver metrics, selected-feature CSV, plot, and fitted pipeline;
- baseline test metrics, full importance CSV, plot, and fitted pipeline;
- overall and per-target forecasting MSE plus a versioned five-model artifact.

The generated importance plot was visually inspected to confirm that the cross-platform CJK font fallback renders the original Chinese feature names.

Synthetic metrics are test artifacts and are not included in the repository or presented as business results. The original report's results were not rerun because the source workbook was not supplied.
