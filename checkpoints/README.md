# Pretrained checkpoints

Validated model weights are hosted outside GitHub because the complete set is
approximately 2.6 GiB:

- [Vision](https://huggingface.co/Viligance/ISA-Vision)
- [Language](https://huggingface.co/Viligance/ISA-Language)
- [Device VGG8](https://huggingface.co/Viligance/ISA-Device-Checkpoints)

`manifest.json` records the file layout, byte size, SHA256 checksum, model
variant, dataset, scale, and validation evidence for all 18 checkpoints.
`release_validation.json` summarizes the release-wide integrity checks.

`device_manifest.json` separately records the selected VGG8/CIFAR-10 device
checkpoints and their training-validation metrics.

The relative `file` field in the manifest is rooted at the corresponding model
repository. For example, `vision/cifar100/hybrid/small/model.pt` maps to
`cifar100/hybrid/small/model.pt` in ISA-Vision.
