# Five-device exact-formula sweep

This launcher runs the MLP/MNIST and VGG8/CIFAR-10 searches for ReRAM, PCM,
STT, FeFET, and Flash. Each device/model study terminates after five Optuna
trials. Two GPUs execute independent studies concurrently; a single study
never uses DDP.

The scheduler is resumable. Completed studies are detected from their summary
tables, while an interrupted study resumes its SQLite Optuna state and latest
trial checkpoint.

```bash
bash experiments/exact_formula_sweep/run_2gpu.sh \
  --data /path/to/existing/device_sweep_data \
  --output-root /path/to/exact_formula_sweep_results
```

The launcher does not download datasets. VGG8 applies the configured health
checks at epochs 8 and 20 so failed trials do not consume all 200 epochs.

If a five-run exact VGG8/FeFET health study has already completed, reuse it
instead of spending GPU time on identical trials:

```bash
bash experiments/exact_formula_sweep/run_2gpu.sh \
  --data /path/to/existing/device_sweep_data \
  --output-root /path/to/exact_formula_sweep_results \
  --reuse-vgg8-fefet-root /path/to/completed_fefet_vgg8_5run_results
```

The scheduler validates all five terminal records, writes their provenance to
`reused_vgg8_fefet.json`, and schedules only FeFET/MLP plus the remaining four
devices for both networks.
