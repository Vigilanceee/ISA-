#!/usr/bin/env python3
"""Download the official TinyStories validation text once for offline evaluation."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file():
        source = hf_hub_download(
            repo_id="roneneldan/TinyStories",
            filename="TinyStories-valid.txt",
            repo_type="dataset",
        )
        temporary = output.with_suffix(output.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(output)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"[TinyStories] path={output} bytes={output.stat().st_size} sha256={digest}")


if __name__ == "__main__":
    main()
