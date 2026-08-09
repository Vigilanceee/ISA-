# Pretrained checkpoints

Validated model weights are hosted outside GitHub because the complete set is
approximately 2.6 GiB:

- [Vision](https://huggingface.co/Viligance/ISA-Vision)
- [Language](https://huggingface.co/Viligance/ISA-Language)

`manifest.json` records the file layout, byte size, SHA256 checksum, model
variant, dataset, scale, and validation evidence for all 18 checkpoints.
`release_validation.json` summarizes the release-wide integrity checks.

The relative `file` field in the manifest is rooted at the corresponding model
repository. For example, `vision/cifar100/hybrid/small/model.pt` maps to
`cifar100/hybrid/small/model.pt` in ISA-Vision.
