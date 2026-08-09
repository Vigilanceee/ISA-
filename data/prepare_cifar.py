#!/usr/bin/env python3
"""Download CIFAR-10 and/or CIFAR-100 with torchvision."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "cifar100", "all"),
        default="all",
    )
    args = parser.parse_args()
    root = Path(args.root)

    if args.dataset in {"cifar10", "all"}:
        target = root / "device_sweeps"
        CIFAR10(root=str(target), train=True, download=True)
        CIFAR10(root=str(target), train=False, download=True)
        print(f"[CIFAR-10] {target}")

    if args.dataset in {"cifar100", "all"}:
        target = root / "cifar100"
        CIFAR100(root=str(target), train=True, download=True)
        CIFAR100(root=str(target), train=False, download=True)
        print(f"[CIFAR-100] {target}")


if __name__ == "__main__":
    main()
