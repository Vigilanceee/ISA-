# ISA: In-Synapse Activation

ISA is a research implementation of physical activation and compute-in-memory
feed-forward networks for vision and language models. The repository contains:

- Digital, Hybrid, and Physical Vision Transformers at three model sizes;
- Digital, Hybrid, and Physical GPT-style language models at three sizes;
- fused Triton/CUDA kernels for the physical Transformer FFN;
- ReRAM, PCM, STT, FeFET, and Flash-transistor device studies with MLP and
  VGG8;
- a measured 24-state verify/write deployment pipeline with activation-aware
  assignment, fixed-assignment post-training, and cell-wise Monte Carlo;
- one launcher for single-GPU, 4-GPU, and 8-GPU training, automatic resume,
  and experiment-matrix execution.

## Model variants

The three Transformer variants share the same attention and embedding
architecture. They differ only in the FFN projections:

| Variant | Up projection | Down projection |
|---|---|---|
| Digital | `Linear + GELU` | `Linear` |
| Hybrid | Flash-EKV `CIMLinear` | Digital `Linear` |
| Physical | Flash-EKV `CIMLinear` | Flash-EKV `CIMLinear` |

All physical projections in the Hybrid and Physical ViT/GPT FFNs use the
fitted Flash-transistor EKV parameter set in
[`configs/devices/flash_transistor_ekv.yaml`](configs/devices/flash_transistor_ekv.yaml).
The current is evaluated with a one-dimensional LUT over
`ΔV = VGS - Vth`, followed by fused Triton forward and CUDA backward kernels.

This Transformer path is distinct from the low-rank planar approximation used
in the multi-device MLP/VGG8 study:

| Device | Device-sweep backend | Rank |
|---|---|---:|
| ReRAM | factorized | — |
| PCM | low-rank planar | 2 |
| STT | factorized | — |
| FeFET | low-rank planar | 16 |
| Flash transistor | low-rank planar | 8 |

`Flash transistor` refers to the physical device model. PyTorch Flash SDPA
refers to the digital attention implementation; the two are independent.

## Installation

The optimized kernels require a CUDA-capable PyTorch installation. Python
3.10–3.12 is recommended.

```bash
git clone https://github.com/Vigilanceee/ISA-.git
cd ISA-

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The CUDA backward extension is compiled on first use. It can also be built
before a run:

```bash
python -c \
  "from isa.kernels.transformer_ffn.cuda_backend import prebuild; prebuild()"
```

## Data

The default data layout and preparation commands are documented in
[`data/README.md`](data/README.md).

```bash
# CIFAR-10 and CIFAR-100
python data/prepare_cifar.py --dataset all

# OpenWebText
python data/prepare_openwebtext.py \
  --output-dir data/language/openwebtext

# TinyStories validation
python data/prepare_tinystories.py \
  --output data/language/benchmarks/tinystories_valid.txt
```

ImageNet images are not redistributed. The ImageNet-200 loader expects
`data/imagenet200/train/<class-id>/` and
`data/imagenet200/val/<class-id>/`.

## Unified launcher

Scientific settings are stored in experiment YAML files. GPU count is selected
independently through a resource profile. The launcher keeps the configured
global batch size fixed and derives the per-GPU batch size.

List the experiments in a matrix:

```bash
python -m isa list --config configs/vision/imagenet200.yaml
```

Run one experiment:

```bash
python -m isa train \
  --config configs/vision/imagenet200.yaml \
  --experiment physical_l \
  --profile profiles/8gpu.yaml
```

Run a complete matrix sequentially:

```bash
python -m isa matrix \
  --config configs/vision/cifar100.yaml \
  --profile profiles/4gpu.yaml \
  --resume auto
```

The same experiment can be launched with a different GPU count without
creating another training script:

```bash
# 4 GPUs
python -m isa train \
  --config configs/language/openwebtext.yaml \
  --experiment physical_m \
  --profile profiles/4gpu.yaml

# 8 GPUs
python -m isa train \
  --config configs/language/openwebtext.yaml \
  --experiment physical_m \
  --profile profiles/8gpu.yaml
```

Use `--data` and `--output-root` to override paths:

```bash
python -m isa matrix \
  --config configs/vision/imagenet200.yaml \
  --profile profiles/8gpu.yaml \
  --data /datasets/imagenet200 \
  --output-root /checkpoints/isa/imagenet200
```

`--resume auto` skips completed experiments and resumes from the configured
last checkpoint when one exists. Every launch writes a machine-readable
manifest under `<output-root>/manifests/`.

## Experiment matrices

### Vision

```bash
# CIFAR-100: 3 variants × 3 sizes
python -m isa matrix \
  --config configs/vision/cifar100.yaml \
  --profile profiles/4gpu.yaml

# ImageNet-200: 3 variants × 3 sizes
python -m isa matrix \
  --config configs/vision/imagenet200.yaml \
  --profile profiles/8gpu.yaml
```

Vision size mapping: S/M/L = `d=192/256/384`.

### Language

```bash
# OpenWebText training: 3 variants × 3 sizes
python -m isa matrix \
  --config configs/language/openwebtext.yaml \
  --profile profiles/4gpu.yaml

# OWT, TinyStories, and BLiMP evaluation for all nine checkpoints
python -m isa evaluate \
  --config configs/language/openwebtext.yaml \
  --profile profiles/4gpu.yaml
```

Language size mapping: S/M/L = `d=192/384/768`.

### Multi-device study

The search space includes learning rate, initialization center,
initialization half-width, and TIA resistance for every device/model pair.

```bash
python -m isa matrix \
  --config configs/device_sweeps/all_devices.yaml \
  --profile profiles/4gpu.yaml \
  --resume auto
```

This runs MLP/MNIST and VGG8/CIFAR-10 for ReRAM, PCM, STT, FeFET, and Flash
transistor.

### Cross-device prediction trajectories

The VGG8 prediction-trajectory experiment tests whether the five physical
device implementations retain a common low-dimensional task geometry. It uses
three seeds, one shared stratified 1,000-image CIFAR-10 probe, joint PCA,
prediction-direction cosine similarity, and a device/seed distance ratio.

```bash
bash experiments/prediction_trajectory/run_pipeline_2gpu.sh \
  --data /path/to/cifar10 \
  --output-root artifacts/prediction_trajectory \
  --gpus 0,1
```

The launcher is resumable at each `(device, seed)` run, saves probabilities
every five epochs, refuses to download CIFAR-10 inside a compute job, and
generates source tables plus publication-ready Figure 3b/3c exports. See
[`experiments/prediction_trajectory/README.md`](experiments/prediction_trajectory/README.md)
for the complete protocol and interpretation boundary.

The default FeFET entry intentionally uses the older, poorer measured-data fit
selected for the device comparison (`A_lk=0.001369`, `B_lk=1.29224`,
`I_S=11.0725`, `n=1.112106`).  The later clean fit is retained as
[`configs/device_sweeps/device_params_fefet_latest_fit.yaml`](configs/device_sweeps/device_params_fefet_latest_fit.yaml).
The selected historical results and the later controlled LR search are kept
separately in [`results/device_sweep_selected.csv`](results/device_sweep_selected.csv)
and [`results/device_sweep_latest_lr_search.csv`](results/device_sweep_latest_lr_search.csv).

### Measured 24-state deployment

The FG50 path is a deployment experiment, not another analytic device backend.
It starts from the 72.19% Physical ViT-S checkpoint, performs discrete
activation-aware verify/write assignment, applies fixed-assignment
compensation training, and evaluates 200 cell-wise raw-curve Monte Carlo
realizations.

```bash
bash experiments/fg50_24state/run_pipeline_2gpu.sh \
  --runtime /data/fg50_24state_runtime.npz \
  --checkpoint /checkpoints/physical_vit_s/best_checkpoint.pth \
  --data /datasets/cifar100 \
  --gpus 0,1
```

The full data schema, runtime builder, resume gates, selected parameters, and
results are documented in
[`experiments/fg50_24state/README.md`](experiments/fg50_24state/README.md).

## Reference results

| Benchmark | Metric | Digital S/M/L | Hybrid S/M/L | Physical S/M/L |
|---|---|---|---|---|
| CIFAR-100 test | Top-1 ↑ | 67.98 / 69.45 / 69.30 | 67.52 / 69.54 / 71.36 | 67.70 / 69.61 / 72.19 |
| ImageNet-200 val. | Top-1 ↑ | 72.43 / 72.57 / 73.64 | 69.33 / 72.63 / 76.71 | 72.71 / 75.80 / 77.19 |
| OpenWebText val. | PPL ↓ | 77.18 / 69.40 / 62.85 | 75.51 / 68.64 / 54.47 | 68.44 / 66.10 / 54.25 |
| TinyStories val. | PPL ↓ | 57.25 / 48.45 / 45.36 | 64.70 / 47.68 / 39.22 | 49.22 / 48.88 / 40.84 |
| BLiMP | Accuracy ↑ | 70.06 / 70.39 / 70.53 | 69.12 / 70.77 / 71.45 | 69.64 / 71.48 / 72.63 |

Machine-readable copies are available in
[`results/reference_results.csv`](results/reference_results.csv) and
[`results/reference_results.json`](results/reference_results.json).

The language protocol uses the exact OpenWebText validation blocks,
a deterministic 999,999-token prefix of the official TinyStories validation
split with context length 128 and stride 64, and all 67,000 BLiMP minimal pairs.

## Repository layout

```text
src/isa/
├── device_models/       # physical equations and fitted parameters
├── approximations/      # exact, LUT, node-planar, and low-rank paths
├── operators/           # CIMLinear and physical/hybrid FFNs
├── kernels/             # Transformer and device-study Triton/CUDA kernels
├── vision/              # ViT models, training, evaluation, data
├── language/            # GPT models, training, evaluation
├── device_sweeps/       # MLP/VGG8 Optuna experiments
├── prediction_trajectory/ # cross-device output-space trajectory analysis
├── measured_deployment/ # empirical codebook, assignment, post-training, MC
└── cli/                 # unified launcher and resume scheduler
```

Configuration is kept outside the source tree:

```text
configs/
├── devices/
├── vision/
├── language/
└── device_sweeps/
```

## Validation

```bash
python -m compileall -q src
pytest -q

python -m isa matrix \
  --config configs/vision/cifar100.yaml \
  --profile profiles/4gpu.yaml \
  --dry-run
```
