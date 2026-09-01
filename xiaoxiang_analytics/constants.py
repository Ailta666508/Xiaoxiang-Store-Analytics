"""Column definitions shared by the analysis and prediction tasks."""

STORE_ID = "门店id"
TOTAL_DAILY_ORDERS = "站日均订单量"

DISTANCE_ORDER_TARGETS = [
    "距离1.5km以下日均订单量",
    "距离1.5~2.1km日均订单量",
    "距离2.1~2.8km日均订单量",
    "距离2.8~3.8km日均订单量",
    "距离3.8km以上日均订单量",
]

DISTANCE_FORECAST_FEATURES = [
    "覆盖户数",
    "距离1.5km以内户数",
    "距离1.5~2.1km户数",
    "距离2.1~2.8km户数",
    "美团月活跃用户数",
    "L4占比",
    "L5占比",
    "仅居住占比",
    "仅工作占比",
    "门店覆盖范围内超市数",
]

COUNT_FEATURES = DISTANCE_FORECAST_FEATURES[:5] + ["门店覆盖范围内超市数"]
PERCENTAGE_FEATURES = ["L4占比", "L5占比", "仅居住占比", "仅工作占比"]

DRIVER_ANALYSIS_EXCLUSIONS = [
    TOTAL_DAILY_ORDERS,
    STORE_ID,
    *DISTANCE_ORDER_TARGETS,
    "美团月活跃用户数",
    "覆盖户数",
    "距离1.5~2.1km户数",
    "距离2.8~3.8km户数",
]

BASELINE_EXCLUSIONS = [
    TOTAL_DAILY_ORDERS,
    STORE_ID,
    *DISTANCE_ORDER_TARGETS,
    "美团月活跃用户数_仅居住占比",
    "美团月活跃用户数_仅工作占比",
    "美团月活跃用户数_居住&工作占比",
]
