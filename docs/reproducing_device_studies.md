# Reproducing the five-device studies

The device collection contains five MLP/MNIST and five VGG8/CIFAR-10 selected
checkpoints, plus two PCM packages tied to the manuscript panels. Each package
contains `model.pth`, `config.json`, `metrics.json`, `evaluation.json`, and the
complete `training_curve.csv`. Code and released checkpoints use the MIT License.

## Install, download and inspect

Install ISA as described in the root README. PyTorch 2.3/CUDA 12.4 and an H100
were used for release verification; the physical kernels require CUDA.

```bash
# List all twelve immutable checkpoint entries; this does not load PyTorch.
python -m isa.device_sweeps.checkpoints --list

# Download a checkpoint and its configuration, and verify the weight SHA256.
python -m isa.device_sweeps.checkpoints \
  --entry mnist-mlp-fefet --download-only --download-dir checkpoints/downloaded
```

The download command prints the local paths. To inspect tensors without running
the model:

```python
import torch
state = torch.load('checkpoints/downloaded/mnist-mlp/fefet/model.pth',
                   map_location='cpu', weights_only=True)
for name, tensor in state.items():
    print(name, tuple(tensor.shape), tensor.dtype)
```

Use `checkpoints/device_studies.json` to obtain the exact Hugging Face revision,
relative paths, byte sizes and checksums. Do not substitute a moving `main`
revision when archiving results. The download helper refuses unverified entries
and checks SHA256 after download.

To download the entire device collection at that exact revision:

```python
import json
from huggingface_hub import snapshot_download
manifest = json.load(open('checkpoints/device_studies.json'))
snapshot_download(manifest['repository'], revision=manifest['revision'],
                  local_dir='checkpoints/device_download')
```

## Full deterministic evaluation

Provide existing torchvision MNIST/CIFAR-10 data directories. The evaluator
does not download data, augment images, change weights, or discard state keys.
It uses the full official test split, deterministic ordering, and strict state
loading. Input encoding, quantization, TIA bounds and calibration are read from
the checkpoint configuration rather than inferred from a generic default.

```bash
python -m isa.device_sweeps.checkpoints \
  --entry mnist-mlp-fefet --data /datasets \
  --output artifacts/mnist_fefet_evaluation.json

python -m isa.device_sweeps.checkpoints \
  --entry cifar10-vgg8-flash --data /datasets \
  --output artifacts/cifar10_flash_evaluation.json
```

Each test split contains 10,000 images. These splits were used for validation
and checkpoint selection during training; the reported values are therefore
not independent held-out estimates. Small floating-point/backend differences
can change a few predictions: the release re-evaluations differ from the
selected training records by at most 0.01 percentage point.

Training-curve loss uses the recorded label smoothing; the evaluation JSON's
`cross_entropy` is ordinary unsmoothed cross-entropy.

The existing MNIST gzip files also match all four official checksums recorded
by torchvision. Their decompressed bytes match the raw IDX files (60,000
training and 10,000 test images). The checksum and class-count audit is in
`results/source_data/device_studies/mnist_data_validation.json`.

| Device | MNIST selected / re-evaluated (%) | CIFAR-10 selected / re-evaluated (%) |
|---|---:|---:|
| ReRAM | 97.95 / 97.95 | 87.50 / 87.50 |
| PCM | 97.99 / 97.99 | 92.03 / 92.02 |
| STT | 97.18 / 97.18 | 89.22 / 89.22 |
| Flash | 96.69 / 96.69 | 90.14 / 90.15 |
| FeFET | 97.58 / 97.58 | 88.70 / 88.70 |

## Device operators and architecture

All models use independent positive and negative state tensors and subtract
their physical currents. Currents are in microamperes; the TIA converts them
to volts using `I * 1e-6 * R`. A checkpoint must be loaded with its configuration.

- ReRAM and STT use their configured analytic-response kernels.
- PCM uses its configured state mapping and response operator. The selected
  packages use `chi=tanh(theta)` and the order-7 sinh expansion. The manuscript
  S9/S10 packages use the fitted state-dependent-current configuration stored
  in their separate configuration files; their accuracies are 97.32%/89.71%.
- Flash uses the fitted EKV response. The MNIST package uses its recorded
  response lookup table; VGG8 uses the degree-5 polynomial response operator
  with BF16 matrix products. These numerical settings are part of the model.
- FeFET uses the fitted LK/EKV forward response with a 512-point lookup table.
  Training uses a surrogate derivative, fixed gate calibration and output gain.
  The surrogate affects the backward derivative, not the stored forward table.
  `models/peripherals.py` and `kernels/device_sweep/lut_triton.py` implement the
  calibration and derivative tables. Omitting calibration or changing the
  derivative is a different experiment, not a reproduction of these weights.

MLP is `784 -> 128 -> 10`, with a TIA between the two physical layers. VGG8
contains six 3x3 convolutions with channels 128/128/256/256/512/512, three
max-pooling stages and two physical FC layers of width 1024 and 10. TIA/BN
stages follow the first seven physical layers. The configured TIA resistance
is scaled by inverse square root of fan-in for VGG8. Calibration slots are
automatically checked against checkpoint keys without renaming tensors.

## Retrain one selected configuration

This bypasses parameter search and preserves the recorded device backend,
calibration, batch size, learning rate, weight decay, label smoothing, voltage
bounds and quantization settings. It uses AdamW, cosine learning-rate decay,
gradient clipping at 1.0 and physical-state clamping. The initialization jitter
half-width is `0.1*w_init_max`, matching the selected studies. CIFAR-10 training
uses random crop and horizontal flip; MNIST uses no augmentation.

```bash
python -m isa.device_sweeps.retrain --entry mnist-mlp-fefet \
  --data /datasets --output artifacts/retrain_mnist_fefet --seed 42

# Continue exactly the saved run configuration and schedule.
python -m isa.device_sweeps.retrain --entry mnist-mlp-fefet \
  --data /datasets --output artifacts/retrain_mnist_fefet --seed 42 --resume
```

The original initialization RNG states were not saved, so retraining is a new
seeded realization of the protocol, not a promise of bitwise-identical training
or the identical selected accuracy. New runs save seed/RNG states, the last
optimizer/scheduler checkpoint, best weights and every training epoch. A
diagnostic `--max-steps 1 --epochs 1` run is explicitly marked as diagnostic.

## Figure source data

`results/device_studies.csv` links each selected checkpoint to its complete
curve and full evaluation. Small source-data files are mirrored under
`results/source_data/device_studies/`. No interpolation or smoothing is applied.
The updated Figure S2 uses `mnist-mlp-flash` (96.69%, epoch 23), not a curve
from another checkpoint. Recreate it with:

```bash
python scripts/plot_device_training_curve.py \
  --source results/source_data/device_studies/mnist-mlp/flash/training_curve.csv \
  --output artifacts/figure_s2_flash_mnist
```

This exports editable SVG/PDF and 600-dpi PNG/TIFF. The plot is one selected
training run, not a multi-seed uncertainty estimate. The two additional PCM
entries are `mnist-mlp-pcm-fig-s9` and `cifar10-vgg8-pcm-fig-s10`.

## Submission bundle

For offline code inspection, use the source ZIP at the GitHub release tag or a
ZIP exported from an exact commit. Include the source-data tables and the
checkpoint manifest. The large weights can be retrieved separately from the
manifest-pinned Hugging Face revision; they need not be embedded in the code
ZIP. A repository revision is not a DOI. If the selected journal requests a
DOI archival deposit, archive the final exact release separately and update
the data/code availability statement with the issued DOI.
