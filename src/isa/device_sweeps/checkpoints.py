"""Strict loading and deterministic evaluation of released device checkpoints.

Example::

    python -m isa.device_sweeps.checkpoints --config config.json \
        --checkpoint model.pth --data data --output evaluation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Mapping
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_state(payload):
    """Read common checkpoint wrappers without silently dropping parameters."""
    import torch

    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must contain a tensor state dictionary")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in payload and isinstance(payload[key], Mapping):
            payload = payload[key]
            break
    if not payload or not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError("State dictionary contains missing or non-tensor entries")
    keys = list(payload)
    if all(key.startswith("module.") for key in keys):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    if any(not torch.isfinite(value).all().item() for value in payload.values()):
        raise ValueError("Checkpoint contains non-finite tensors")
    return dict(payload)


def infer_calibration_slots(architecture: str, state: Mapping) -> bool:
    """Infer the sequential-module layout without changing checkpoint tensors."""
    architecture = architecture.lower()
    prefix = "classifier" if architecture == "mlp" else "features"
    if architecture not in {"mlp", "vgg8"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    compact = f"{prefix}.2.theta_pos" in state
    calibrated = f"{prefix}.3.theta_pos" in state
    if compact == calibrated:
        raise ValueError(f"Cannot identify the {architecture} checkpoint module layout")
    return calibrated


def build_checkpoint_model(config: dict, state: Mapping):
    from isa.device_sweeps.models.mlp import MLP
    from isa.device_sweeps.models.vgg8 import VGG8

    architecture = str(config["architecture"]).lower()
    device = str(config["device"]).lower()
    if device not in {"reram", "pcm", "stt", "fefet", "flash"}:
        raise ValueError(f"Unsupported device: {device}")
    params = dict(config["device_parameters"])
    slots = infer_calibration_slots(architecture, state)
    if "calibration_slots" in params and bool(params["calibration_slots"]) != slots:
        raise ValueError("Configuration calibration_slots disagrees with checkpoint keys")
    params["calibration_slots"] = slots
    model_class = {"mlp": MLP, "vgg8": VGG8}[architecture]
    model = model_class(device, params, **config["model_parameters"])
    model.load_state_dict(state, strict=True)
    return model


def evaluation_dataset(config: dict, data: str | Path):
    from torchvision import datasets, transforms

    dataset = str(config["dataset"]).lower().replace("-", "")
    transistor = str(config["device"]).lower() in {"flash", "fefet"}
    channels = 1 if dataset == "mnist" else 3
    mean, std = ((0.0, 0.25) if transistor else (0.5, 1.0))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,) * channels, (std,) * channels),
    ])
    dataset_class = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}.get(dataset)
    if dataset_class is None:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    return dataset_class(root=str(data), train=False, download=False, transform=transform)


def evaluate(config: dict, checkpoint: Path, data: Path, *, batch_size=128, device="cuda") -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from isa.device_sweeps.models.quantization import set_quantization_config

    expected_sha = config.get("checkpoint_sha256")
    actual_sha = sha256_file(checkpoint)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"Checkpoint SHA256 mismatch: {actual_sha}")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable PyTorch installation is required by the device kernels")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    quant = config.get("quantization", {})
    set_quantization_config(
        quant.get("enabled", True), quant.get("input_bits", 8), quant.get("output_bits", 8)
    )
    state = checkpoint_state(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = build_checkpoint_model(config, state).to(device).eval()
    dataset = evaluation_dataset(config, data)
    if len(dataset) != 10000:
        raise ValueError(f"Expected the complete 10,000-image test split, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct, count, loss_sum = 0, 0, 0.0
    started = time.monotonic()
    with torch.inference_mode():
        for batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs).float()
            if not torch.isfinite(logits).all().item():
                raise ValueError(f"Non-finite logits in evaluation batch {batch}")
            correct += logits.argmax(1).eq(labels).sum().item()
            count += labels.numel()
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            if batch % 20 == 0:
                print(json.dumps({"examples": count, "accuracy": correct / count}), flush=True)
    if count != len(dataset):
        raise RuntimeError(f"Incomplete evaluation: {count}/{len(dataset)} examples")
    reference = config.get("training_validation_accuracy")
    return {
        "architecture": config["architecture"], "dataset": config["dataset"],
        "device": config["device"], "checkpoint_sha256": actual_sha,
        "strict_load": True, "examples": count, "correct": correct,
        "accuracy": correct / count, "cross_entropy": loss_sum / count,
        "training_validation_accuracy": reference,
        "difference_percentage_points": None if reference is None else 100 * (correct / count - reference),
        "dataset_split": "official_test",
        "used_for_checkpoint_selection": config.get("evaluation_split_used_for_selection", True),
        "elapsed_seconds": time.monotonic() - started,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else "cpu"},
        "configuration": config,
    }


def download_entry(manifest: dict, entry_id: str, output: Path | None = None):
    """Download an immutable published checkpoint and its configuration."""
    from huggingface_hub import hf_hub_download

    matches = [row for row in manifest["checkpoints"] if row["id"] == entry_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint entry for {entry_id!r}; found {len(matches)}")
    entry = matches[0]
    if entry.get("status") != "verified":
        raise ValueError(f"Checkpoint {entry_id!r} has not passed release verification")
    revision = entry.get("revision", manifest.get("revision", ""))
    if (not isinstance(revision, str) or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)):
        raise ValueError("The checkpoint manifest must pin an immutable repository revision")
    kwargs = {"repo_id": manifest["repository"], "revision": revision, "repo_type": "model"}
    if output is not None:
        kwargs["local_dir"] = str(output)
    config = Path(hf_hub_download(filename=entry["configuration_file"], **kwargs))
    checkpoint = Path(hf_hub_download(filename=entry["file"], **kwargs))
    if sha256_file(checkpoint) != entry["sha256"]:
        raise ValueError(f"Downloaded checkpoint SHA256 mismatch: {entry_id}")
    return config, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("checkpoints/device_studies.json"))
    parser.add_argument("--entry", help="Manifest checkpoint identifier, e.g. mnist-mlp-fefet")
    parser.add_argument("--list", action="store_true", help="List the manifest checkpoint entries")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.list:
        manifest = json.loads(args.manifest.read_text())
        for row in manifest["checkpoints"]:
            print(f"{row['id']}\t{row['status']}\t{row.get('training_validation_accuracy', '')}")
        return
    if args.entry:
        if args.config or args.checkpoint:
            parser.error("Use --entry or explicit --config/--checkpoint, not both")
        args.config, args.checkpoint = download_entry(
            json.loads(args.manifest.read_text()), args.entry, args.download_dir
        )
    if args.config is None or args.checkpoint is None:
        parser.error("Provide --entry, or both --config and --checkpoint")
    if args.download_only:
        print(json.dumps({"config": str(args.config), "checkpoint": str(args.checkpoint),
                          "sha256": sha256_file(args.checkpoint)}, indent=2))
        return
    if args.data is None or args.output is None:
        parser.error("Evaluation requires --data and --output")
    config = json.loads(args.config.read_text())
    result = evaluate(config, args.checkpoint, args.data, batch_size=args.batch_size, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
