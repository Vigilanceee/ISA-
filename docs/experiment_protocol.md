# Experiment protocol

## Main model matrix

The main study contains 27 training configurations:

- CIFAR-100: Digital/Hybrid/Physical × S/M/L;
- ImageNet-200: Digital/Hybrid/Physical × S/M/L;
- OpenWebText: Digital/Hybrid/Physical × S/M/L.

Digital baselines are trained first. Each Hybrid or Physical model uses the
matching Digital size as its initialization and FFN-matching target.

## Language evaluation

Each of the nine language checkpoints is evaluated on:

- OpenWebText validation perplexity using the exact training-validation blocks;
- TinyStories validation perplexity on a deterministic 999,999-token prefix,
  with GPT-2 BPE, context length 128, and stride 64;
- BLiMP forced-choice accuracy over all 67,000 minimal pairs.

Evaluation results are written after each completed task. Re-running the same
command with resume enabled skips completed tasks.

## Distributed execution

Experiment YAML files define scientific settings. Resource profiles define GPU
count. The launcher derives per-GPU batch size from the configured global batch
size and records the resolved command in a run manifest.
