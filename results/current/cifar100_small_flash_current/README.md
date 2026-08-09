# CIFAR-100 Small Flash-current evaluation

This directory contains a deterministic evaluation of the published
CIFAR-100 Small Hybrid and Physical checkpoints using the Flash-EKV operator.
Current is measured before TIA scaling on fixed test samples 0–3; accuracy and
cross-entropy use the complete 10,000-image CIFAR-100 test set.

The two `*_layers.csv` files contain one row for every Flash-projection
position. Each row records the block/layer identity, tensor shape, current
count, signed and absolute-current statistics, finite rate, and TIA saturation
rate. The JSON files retain the same records plus run metadata. `comparison.*`
provides model-level and matched-position summaries.

| Variant | Top-1 | mean absolute current | p99 absolute current | max absolute current | finite | TIA saturation |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid | 70.98% | 0.776719 µA | 2.900553 µA | 9.703725 µA | 100% | 0% |
| Physical | 71.97% | 4.168495 µA | 22.067599 µA | 46.341072 µA | 100% | 2.744942% |

The formatted Excel workbook containing the overview, matched FC1 positions,
all Hybrid positions, all Physical positions, and raw source tables is attached
to the GitHub release.
