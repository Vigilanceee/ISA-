#!/usr/bin/env python3
"""Download/stream OpenWebText into the text files used by ISA training.

The training code in this repo expects ordinary UTF-8 text files. This script
streams an OpenWebText-style Hugging Face dataset and writes two files:
  - owt_train.txt
  - owt_valid.txt

It is designed for CPU download jobs and works with HF mirrors via HF_ENDPOINT.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare OpenWebText text files for ISA")
    p.add_argument("--dataset", default="Skylion007/openwebtext",
                   help="Hugging Face dataset id (default: Skylion007/openwebtext)")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--output-dir", default="data/language/openwebtext")
    p.add_argument("--validation-every", type=int, default=1000,
                   help="Put every Nth document into validation; 0 disables validation split")
    p.add_argument("--max-docs", type=int, default=0, help="Stop after this many docs; 0 means no limit")
    p.add_argument("--max-bytes", type=int, default=0, help="Stop after this many written bytes; 0 means no limit")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", ""),
                   help="Optional HF mirror endpoint, e.g. https://hf-mirror.com")
    p.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE", ""))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-every", type=int, default=1000)
    return p.parse_args()


def byte_limit(value: str) -> int:
    s = value.strip().lower()
    if not s:
        return 0
    units = [("gb", 1024 ** 3), ("g", 1024 ** 3), ("mb", 1024 ** 2), ("m", 1024 ** 2), ("kb", 1024), ("k", 1024)]
    for suffix, mul in units:
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mul)
    return int(s)


def iter_rows(dataset: Any) -> Iterable[dict[str, Any]]:
    for row in dataset:
        if isinstance(row, dict):
            yield row


def main() -> None:
    args = parse_args()
    if isinstance(args.max_bytes, str):
        args.max_bytes = byte_limit(args.max_bytes)

    # The cluster image often sets HF_HUB_OFFLINE=1. A download job must override it.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    from datasets import load_dataset

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "owt_train.txt"
    val_path = out_dir / "owt_valid.txt"
    meta_path = out_dir / "prepare_meta.json"

    if not args.overwrite and (train_path.exists() or val_path.exists()):
        raise SystemExit(
            f"Refusing to overwrite existing files. Use --overwrite or change "
            f"--output-dir. Existing: {train_path}, {val_path}"
        )

    print(f"[prepare_owt] dataset={args.dataset} split={args.split} streaming={args.streaming}", flush=True)
    print(f"[prepare_owt] output train={train_path}", flush=True)
    print(f"[prepare_owt] output valid={val_path}", flush=True)
    if args.hf_endpoint:
        print(f"[prepare_owt] HF_ENDPOINT={args.hf_endpoint}", flush=True)

    ds_kwargs: dict[str, Any] = {"split": args.split, "streaming": args.streaming}
    if args.cache_dir:
        ds_kwargs["cache_dir"] = args.cache_dir
    if args.trust_remote_code:
        ds_kwargs["trust_remote_code"] = True
    ds = load_dataset(args.dataset, **ds_kwargs)

    start = time.time()
    docs = 0
    train_docs = 0
    val_docs = 0
    bytes_written = 0

    with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
        for row in iter_rows(ds):
            text = row.get(args.text_field)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            docs += 1
            target_is_val = bool(args.validation_every and docs % args.validation_every == 0)
            f = val_f if target_is_val else train_f
            f.write(text)
            f.write("\n<|endoftext|>\n")
            n = len(text.encode("utf-8")) + len("\n<|endoftext|>\n")
            bytes_written += n
            if target_is_val:
                val_docs += 1
            else:
                train_docs += 1

            if args.report_every > 0 and docs % args.report_every == 0:
                elapsed = max(time.time() - start, 1e-6)
                mb = bytes_written / (1024 ** 2)
                print(f"[prepare_owt] docs={docs} train={train_docs} val={val_docs} bytes={mb:.1f}MiB rate={mb/elapsed:.2f}MiB/s", flush=True)

            if args.max_docs and docs >= args.max_docs:
                break
            if args.max_bytes and bytes_written >= args.max_bytes:
                break

    elapsed = max(time.time() - start, 1e-6)
    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "text_field": args.text_field,
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "docs": docs,
        "train_docs": train_docs,
        "validation_docs": val_docs,
        "bytes_written": bytes_written,
        "elapsed_sec": elapsed,
        "mib_per_sec": bytes_written / (1024 ** 2) / elapsed,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[prepare_owt] done docs={docs} bytes={bytes_written / (1024 ** 2):.1f}MiB rate={meta['mib_per_sec']:.2f}MiB/s", flush=True)
    print(f"[prepare_owt] meta={meta_path}", flush=True)


if __name__ == "__main__":
    main()
