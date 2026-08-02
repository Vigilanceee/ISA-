# Cross-device prediction trajectories

This experiment asks whether five device-specific VGG8 models follow a common
low-dimensional prediction geometry on CIFAR-10. It intentionally uses ordinary
PCA, cosine similarity, and normalized Euclidean distance; it does not implement
information-geometric distances, geodesics, InPCA, or weight-space analysis.

## Protocol

- Devices: ReRAM, PCM, STT-MRAM, FeFET, and Flash.
- Seeds: 0, 1, and 2.
- Training: 200 epochs with one shared optimizer, learning rate, scheduler,
  regularization recipe, and batch size.
- Device-specific searched hyperparameters: only `init_center`,
  `init_half_width`, and `tia_r`; the physical equations remain device-specific.
- Probe: the same stratified 1,000-image subset of the CIFAR-10 test set for
  every run.
- Snapshots: validation metrics and `[1000, 10]` float32 probabilities at epoch
  0 and every 5 epochs.
- Resume: each run has an atomic last checkpoint and an independent completion
  marker. A restarted matrix skips completed runs and resumes incomplete runs.

## Local smoke test

The smoke test runs one device and one seed for 10 epochs and validates all three
probe snapshots. The PCA/statistics implementation is covered independently by
the unit tests and a synthetic complete-matrix integration test:

```bash
bash experiments/prediction_trajectory/run_pipeline_2gpu.sh \
  --data /datasets/cifar10 \
  --output-root artifacts/prediction_trajectory_smoke \
  --gpus 0 \
  --smoke
```

The data directory must already contain `cifar-10-batches-py`; the experiment
never downloads a dataset inside a compute job.

## Full two-GPU run

```bash
bash experiments/prediction_trajectory/run_pipeline_2gpu.sh \
  --data /datasets/cifar10 \
  --output-root artifacts/prediction_trajectory \
  --gpus 0,1
```

The launcher dynamically assigns the next unfinished `(device, seed)` run to
the first available GPU. After all 15 runs finish, it writes:

```text
artifacts/prediction_trajectory/
├── raw/
├── analysis/
├── figures/
├── tables/
├── reports/
└── completed.json
```

The primary outputs are joint PCA coordinates and explained variance, milestone
cosine-similarity matrices, same-accuracy and same-epoch effect ratios with
bootstrap intervals, source CSV files, and publication exports in PNG, PDF,
SVG, and TIFF.

## Interpretation boundary

PCA supports a claim about a common low-dimensional prediction geometry, not a
proof that all devices share a mathematically identical nonlinear manifold.
The bootstrap intervals summarize the finite three-seed experiment and should
not be interpreted as population-level biological confidence intervals.

## Figure contract

- Core conclusion: distinct device parameterizations retain common dominant
  prediction directions, while speed-aligned and accuracy-aligned comparisons
  test for device-specific residual paths without presupposing the outcome.
- Archetype and backend: quantitative grid, generated only with Python and
  Matplotlib.
- Final size: Figure 3b is 89 mm wide; Figure 3c is 183 mm wide.
- Panel map: 3b shows joint-PCA seed and device-mean trajectories plus cumulative
  variance; 3c shows cross-device cosine similarity and the device/seed ratio.
- Hero evidence: the joint trajectories and mean cosine curve. Validation
  evidence: top-3 variance, `k90`, and same-epoch versus same-accuracy `R`.
- Statistics: `n=3` independent training seeds per device; curves report the
  arithmetic mean and deterministic 95% bootstrap intervals. No hypothesis test
  or multiple-comparison correction is used.
- Source data: every quantitative element is traceable to the CSV files under
  `analysis/`; the train/validation/test split and metric definitions are fixed
  in `config.yaml` and the generated methods report.
- Interpretation constraints: three seeds limit interval interpretation; PCA
  establishes a shared low-dimensional subspace rather than a strict nonlinear
  manifold; milestone interpolation is only valid inside the common
  achieved-accuracy range.
