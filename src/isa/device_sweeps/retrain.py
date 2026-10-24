"""Retrain a released device configuration without an additional parameter search.

The original initialization RNG states were not recorded. A new run records its
explicit seed and resumable RNG/optimizer states; it is not a bitwise replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path


def main():
    import numpy as np
    import torch
    from torch import nn

    from isa.device_sweeps.checkpoints import download_entry
    from isa.device_sweeps.models.mlp import MLP
    from isa.device_sweeps.models.quantization import set_quantization_config
    from isa.device_sweeps.models.vgg8 import VGG8
    from isa.device_sweeps.utils.data_loader import get_cifar10_loaders, get_mnist_loaders

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--entry")
    parser.add_argument("--manifest", type=Path, default=Path("checkpoints/device_studies.json"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Limit batches per epoch for a diagnostic run")
    args = parser.parse_args()
    if args.entry:
        args.config, _ = download_entry(json.loads(args.manifest.read_text()), args.entry)
    config = json.loads(args.config.read_text())
    hp = config["training"]["hyperparameters"]
    epochs = args.epochs or int(config["training"]["epochs"])
    if epochs <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error("epochs and max-steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The device kernels require a CUDA-capable PyTorch installation")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Output directory is not empty; use --resume or a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    quant = config["quantization"]
    set_quantization_config(quant["enabled"], quant["input_bits"], quant["output_bits"])
    params = dict(config["device_parameters"])
    # The selected studies sampled w_init_max, then used 0.1*w_init_max as jitter.
    params["init_half_width_multiplier"] = config["training"].get("init_half_width_multiplier", 0.1)
    model_class = {"MLP": MLP, "VGG8": VGG8}[config["architecture"]]
    model = model_class(config["device"].lower(), params, **config["model_parameters"]).cuda()
    loaders = get_mnist_loaders if config["architecture"] == "MLP" else get_cifar10_loaders
    train_loader, val_loader = loaders(
        batch_size=config["training"]["batch_size"], data_dir=str(args.data),
        num_workers=0, device_type=config["device"].lower(), download=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])
    last = args.output / "last.pt"
    start, best = 1, -1.0
    records = []
    if args.resume:
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["seed"] != args.seed or checkpoint["epochs"] != epochs:
            raise ValueError("Resume configuration, seed or schedule differs from the saved run")
        if checkpoint["max_steps"] != args.max_steps:
            raise ValueError("Resume batch limit differs from the saved run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        random.setstate(checkpoint["python_rng"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best"]
        records = checkpoint["records"]
    run_metadata = {"configuration": config, "seed": args.seed, "epochs": epochs,
                    "init_half_width_multiplier": params["init_half_width_multiplier"],
                    "max_steps": args.max_steps, "diagnostic": args.max_steps is not None,
                    "original_initialization_rng_recorded": False,
                    "split": "official test used for validation and selection"}
    (args.output / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")

    def phase(loader, training):
        model.train(training)
        count, correct, total = 0, 0, 0.0
        with torch.set_grad_enabled(training):
            for index, (images, labels) in enumerate(loader):
                if args.max_steps is not None and index >= args.max_steps:
                    break
                images, labels = images.cuda(), labels.cuda()
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss).item():
                    raise ValueError("Non-finite loss")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    with torch.no_grad():
                        for layer in model.modules():
                            if hasattr(layer, "theta_pos"):
                                layer.theta_pos.clamp_(layer.w_min, layer.w_max)
                                layer.theta_neg.clamp_(layer.w_min, layer.w_max)
                n = labels.numel()
                count += n
                correct += logits.argmax(1).eq(labels).sum().item()
                total += loss.item() * n
        return total / count, correct / count

    for epoch in range(start, epochs + 1):
        started = time.monotonic()
        train_loss, train_acc = phase(train_loader, True)
        val_loss, val_acc = phase(val_loader, False)
        scheduler.step()
        record = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                      val_loss=val_loss, val_acc=val_acc, lr=scheduler.get_last_lr()[0])
        records.append(record)
        if val_acc > best:
            best = val_acc
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                       args.output / "model.pth")
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), epoch=epoch, best=best,
                        records=records, config=config, seed=args.seed, epochs=epochs,
                        max_steps=args.max_steps, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate()), last)
        with (args.output / "training_curve.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps({**record, "best_val_acc": best,
                          "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
