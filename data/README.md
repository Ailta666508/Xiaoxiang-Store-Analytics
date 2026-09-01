# Private workbook setup

The source workbook is not redistributed. Use only data that you are authorized to access.

By default, the command-line tools read worksheet `Sheet2`. Pass another name with `--sheet` when needed.

## Demand-driver analysis

The main target is:

- `站日均订单量`

The scripts automatically treat all columns outside their documented exclusion sets as candidate predictors. See `xiaoxiang_analytics/constants.py` for the exact lists.

## Distance-band forecasting

The five model inputs are built from these ten fields:

- `覆盖户数`
- `距离1.5km以内户数`
- `距离1.5~2.1km户数`
- `距离2.1~2.8km户数`
- `美团月活跃用户数`
- `L4占比`
- `L5占比`
- `仅居住占比`
- `仅工作占比`
- `门店覆盖范围内超市数`

Targets:

- `距离1.5km以下日均订单量`
- `距离1.5~2.1km日均订单量`
- `距离2.1~2.8km日均订单量`
- `距离2.8~3.8km日均订单量`
- `距离3.8km以上日均订单量`

Count fields may contain commas or spaces. Percentage fields may be decimal values or strings with a `%` suffix. Rows missing any forecasting input or target are excluded from model training.
