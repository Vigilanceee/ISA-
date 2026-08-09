# FG50 24-state measured-codebook deployment

This experiment deploys the trained CIFAR-100 Physical ViT-S FFN onto a
24-state empirical codebook derived from 1,040 measured cells.  It is separate
from `isa.device_sweeps`: the sweep package approximates analytic device
equations for MLP/VGG8 training, whereas this experiment performs discrete
verify/write assignment against measured I-V curves and then compensates the
fixed assignment.

## Method

1. Build a compact runtime from all 32,240 raw curves (1,040 cells × 31
   programmed levels).  Currents are floored at 0.39 nA, converted to monotone
   envelopes, scored by inverse-log-current horizontal shift, and stably
   partitioned into the published 24 state counts.
2. Evaluate the independent Q-point assignment.
3. Search four activation-aware assignments.  Top-K refers to candidate
   **24×24 center-state pairs**; it does not grow with the 32,240 raw member
   curves.  Calibration tokens are collected per layer and a residual-aware
   coordinate pass chooses the fixed positive/negative state pair for every
   physical cell.
4. Enforce the assignment gate, then run fixed-assignment compensation.  The
   measured forward remains fixed while the selected digital parameters are
   trained.  The best reported recipe trains attention, fc2, LayerNorm, and
   the classifier head; the discrete fc1 assignment is never refreshed.
5. Evaluate the deterministic center curves and run 200 empirical Monte Carlo
   realizations.  Each realization independently samples one complete raw
   member curve for every positive/negative physical cell and holds it fixed
   for the entire validation set.

The runtime mapping is reconstructed deterministically from the published
state counts and centers.  It is not presented as an unreleased original
cell-to-state assignment.

## Input format

Raw measurement files are not committed.  Generate three inputs locally:

- `fg50_members.npz`
  - `voltage_v`: `[61]`
  - `curve_ids`: `[1040, 31]`
  - `cell_ids`: `[1040]`
  - `program_states`: `[31]`
  - `raw_current_na`: `[1040, 31, 61]`
- `fg50_24state_centers.npz`
  - `voltages`: `[61]`
  - `curves`: `[24, 61]` in nA
- `fg50_24state_table.json`: 24 records containing at least `curve_count`,
  `shift_center_V`, `best_verify_V`, `verify_I10`, `verify_IMedian`, and
  `verify_I90`.

Build the runtime:

```bash
python -m isa.measured_deployment.fg50_runtime \
  --member-archive /data/fg50_members.npz \
  --center-codebook /data/fg50_24state_centers.npz \
  --state-table /data/fg50_24state_table.json \
  --output-dir outputs/fg50_runtime
```

The builder writes the runtime NPZ, the reconstructed member mapping, a
manifest with SHA-256 input identities, and numerical audit files.

## Two-GPU reproducible pipeline

```bash
bash experiments/fg50_24state/run_pipeline_2gpu.sh \
  --runtime outputs/fg50_runtime/fg50_24state_runtime.npz \
  --checkpoint /checkpoints/physical_vit_s/best_checkpoint.pth \
  --data /datasets/cifar100 \
  --output outputs/fg50_24state \
  --gpus 0,1
```

Use `--stages assignment`, `--stages compensation`, or `--stages mc` to run a
subset.  Every stage is resumable and refuses to advance when its accuracy or
seed-completeness gate fails.

## Reproduced result

| Stage | CIFAR-100 top-1 |
|---|---:|
| Continuous Physical ViT-S checkpoint | 72.19% |
| Independent Q-point assignment | 54.00% |
| Activation-aware assignment (Top-128, 8,192 tokens/layer, block 2) | 65.44% |
| Fixed-assignment compensation | 71.46% |
| Deterministic center-curve evaluation | 71.35% |
| 200-seed raw-member Monte Carlo mean | 66.1091% |

The Monte Carlo standard deviation is 1.1268 percentage points and the 95%
confidence interval for the mean is 65.9529–66.2653%.  Seeds are exactly
10000–10199 with no gaps or duplicates.  Machine-readable summaries and all
200 per-seed rows are in [`results/`](results/).

## Performance path

`isa.measured_deployment.kernels.measured_lut_triton` implements exact
piecewise-linear measured lookup and input-dimension reduction in one Triton
kernel.  Center-state assignment uses compact 24×24 pair coefficients; Monte
Carlo member mode gathers positive/negative coefficients from the raw curve
table.  For large token batches, the backend can retain the segmented-GEMM
path for higher throughput; the threshold is configurable through the
documented `FG50_*_TOKEN_THRESHOLD` variables in `fused_backend.py`.
