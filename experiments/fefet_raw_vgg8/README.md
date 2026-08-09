# FeFET exact-operator VGG8 health study

This experiment runs five deterministic CIFAR-10/VGG8 seeds with the original
FeFET L-K/EKV response, six Newton iterations, and analytical gradients.  The
training hyperparameters are fixed in `manifest.yaml`; only the exact optimized
kernel implementation is selected through the device configuration.

## Run

Use an existing CIFAR-10 directory.  The launcher does not require a dataset
download when `cifar-10-batches-py` is already present.

```bash
bash experiments/fefet_raw_vgg8/run_5seed.sh \
  --data-dir /path/to/cifar10 \
  --output-root /path/to/results
```

For two GPUs, run independent seeds concurrently:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash experiments/fefet_raw_vgg8/run_5seed.sh \
  --data-dir /path/to/cifar10 \
  --output-root /path/to/results \
  --parallel-seeds 2
```

The launcher is resumable.  A seed with a terminal `COMPLETE` or `PRUNED`
record is skipped, while a non-terminal Optuna checkpoint resumes from its last
committed epoch.

## Health decisions

- Any non-finite loss or accuracy stops immediately.
- At epoch 8, a best validation accuracy below 15% is pruned.
- At epoch 20, a best validation accuracy below 35% is pruned when the trailing
  five-epoch gain is also below two percentage points.

Every seed stores its command and process output in `launcher.log`, epoch
metrics in `trial_0.csv`, resolved physical/training configuration in
`trial_0_config.json`, checkpoints in the seed run directory, and the terminal
decision in `terminal.json`.  The configuration record includes the exact
FeFET kernel SHA-256 for provenance.
