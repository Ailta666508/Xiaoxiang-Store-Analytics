from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xiaoxiang_analytics.constants import DISTANCE_ORDER_TARGETS
from xiaoxiang_analytics.distance_forecasting import train_distance_forecaster


@pytest.fixture(scope="session")
def project_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 24
    coverage = np.arange(1200, 1200 + 35 * rows, 35)
    active = np.arange(900, 900 + 28 * rows, 28)
    distance_15 = np.arange(420, 420 + 12 * rows, 12)
    distance_21 = np.arange(310, 310 + 9 * rows, 9)
    distance_28 = np.arange(240, 240 + 7 * rows, 7)
    competition = np.tile([2, 3, 4, 5, 6, 7], 4)
    l4 = np.linspace(0.18, 0.42, rows)
    l5 = np.linspace(0.04, 0.14, rows)
    residence = np.linspace(0.30, 0.58, rows)
    work = np.linspace(0.12, 0.28, rows)

    total = (
        180
        + 0.15 * coverage
        + 0.18 * active
        + 230 * residence
        - 4 * competition
        + rng.normal(0, 4, rows)
    )
    shares = np.array([0.38, 0.24, 0.18, 0.13, 0.07])
    targets = total[:, None] * shares + rng.normal(0, 2, (rows, 5))

    frame = pd.DataFrame(
        {
            "门店id": [f"store-{index:02d}" for index in range(rows)],
            "站日均订单量": total,
            "覆盖户数": [f"{value:,}" for value in coverage],
            "距离1.5km以内户数": distance_15,
            "距离1.5~2.1km户数": distance_21,
            "距离2.1~2.8km户数": distance_28,
            "距离2.8~3.8km户数": np.arange(180, 180 + 5 * rows, 5),
            "距离3.8km以上户数": np.arange(90, 90 + 3 * rows, 3),
            "美团月活跃用户数": active,
            "L4占比": l4,
            "L5占比": l5,
            "L3及以下占比": 1 - l4 - l5,
            "仅居住占比": residence,
            "仅工作占比": work,
            "美团月活跃用户数_仅居住占比": residence,
            "美团月活跃用户数_仅工作占比": work,
            "美团月活跃用户数_居住&工作占比": 1 - residence - work,
            "门店覆盖范围内超市数": competition,
            "门店覆盖范围内小型超市数": np.maximum(competition - 1, 0),
        }
    )
    for index, target in enumerate(DISTANCE_ORDER_TARGETS):
        frame[target] = targets[:, index]
    return frame


@pytest.fixture(scope="session")
def forecast_bundle(project_frame):
    bundle, metrics = train_distance_forecaster(
        project_frame, test_size=0.25, n_estimators=8
    )
    return bundle, metrics
