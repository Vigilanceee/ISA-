# Pretrained checkpoints

Validated model weights are hosted outside GitHub. The Transformer collection
alone is approximately 2.6 GiB; the device collection is hosted separately:

- [Vision](https://huggingface.co/Viligance/ISA-Vision)
- [Language](https://huggingface.co/Viligance/ISA-Language)
- [Device MLP/MNIST and VGG8/CIFAR-10](https://huggingface.co/Viligance/ISA-Device-Checkpoints)

`manifest.json` records the file layout, byte size, SHA256 checksum, model
variant, dataset, scale, and validation evidence for all 18 checkpoints.
`release_validation.json` summarizes the release-wide integrity checks.

`device_studies.json` records all twelve verified device packages: five
MLP/MNIST, five VGG8/CIFAR-10, and two manuscript PCM panel packages. It pins
an immutable Hugging Face revision and records weight sizes/SHA256, matching
configurations, complete curves and full 10,000-image evaluations.
`device_manifest.json` is the VGG8-only subset for existing consumers.

```bash
python -m isa.device_sweeps.checkpoints --list
python -m isa.device_sweeps.checkpoints \
  --entry mnist-mlp-flash --download-only --download-dir checkpoints/downloaded
```

To inspect a weight, use `torch.load(path, map_location='cpu', weights_only=True)`.
For inference, use the matching configuration through the strict evaluator;
TIA ranges, input encoding and FeFET calibration must not be guessed. See
[`reproduction instructions`](../docs/reproducing_device_studies.md).

Device weights use the MIT License. Accuracy records distinguish the selected
training-validation value from full re-evaluation. The official test split was
used for checkpoint selection, not retained as an independent final holdout.

The relative `file` field in the manifest is rooted at the corresponding model
repository. For example, `vision/cifar100/hybrid/small/model.pt` maps to
`cifar100/hybrid/small/model.pt` in ISA-Vision.
