#!/usr/bin/env python3
"""Evaluate custom GPT-2 CIM checkpoints on language-model benchmarks.

Metrics:
  * OpenWebText, WikiText-2, PTB, and TinyStories: token-level perplexity.
  * BLiMP: forced-choice grammatical acceptability accuracy.
  * LAMBADA: exact greedy continuation accuracy for the final word.
  * HellaSwag, PIQA, ARC-Easy: length-normalized choice log-likelihood accuracy.

Each requested task is saved immediately to a JSON result file. Re-running with
--resume skips tasks already present, so a long evaluation is restartable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import GPT2TokenizerFast

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from isa.language.models import create_model


TASKS = (
    "owt",
    "tinystories",
    "wikitext2",
    "ptb",
    "blimp",
    "lambada",
    "hellaswag",
    "piqa",
    "arc_easy",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--owt-valid", default="data/openwebtext/owt_valid.txt")
    p.add_argument("--benchmark-data-dir", default="benchmark_data")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--ppl-stride", type=int, default=64)
    p.add_argument(
        "--tinystories-max-tokens",
        type=int,
        default=1_000_000,
        help="Evaluate a deterministic prefix of this many GPT-2 tokens; 0 uses the full validation split.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Debug-only example limit; 0 means full set.")
    return p.parse_args()


def atomic_json_dump(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(path: str, device: torch.device):
    state = torch.load(path, map_location="cpu")
    cfg = state.get("model_config")
    if cfg is None or "model_state" not in state:
        raise ValueError(f"Invalid checkpoint: {path}")
    if isinstance(cfg, dict):
        from isa.language.config import ModelConfig

        cfg = ModelConfig(**cfg)
    model = create_model(cfg)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    return model, cfg, state


def encode(tokenizer: GPT2TokenizerFast, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def encode_pair(tokenizer: GPT2TokenizerFast, context: str, continuation: str) -> Tuple[List[int], List[int]]:
    """Tokenize a context/continuation pair without losing boundary whitespace."""
    trailing = len(context) - len(context.rstrip())
    if trailing:
        continuation = context[-trailing:] + continuation
        context = context[:-trailing]
    ctx = encode(tokenizer, context)
    whole = encode(tokenizer, context + continuation)
    if whole[: len(ctx)] != ctx:
        # Rare BPE merge across the boundary: use the longest common prefix.
        common = 0
        for a, b in zip(ctx, whole):
            if a != b:
                break
            common += 1
        ctx = whole[:common]
    cont = whole[len(ctx) :]
    if not ctx:
        ctx = [tokenizer.eos_token_id]
    if not cont:
        raise ValueError("Continuation tokenized to an empty sequence")
    return ctx, cont


@torch.no_grad()
def continuation_scores(
    model,
    pairs: Sequence[Tuple[List[int], List[int]]],
    max_length: int,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    """Return total and per-token mean log-likelihood for each pair."""
    seqs: List[List[int]] = []
    starts: List[int] = []
    for ctx, cont in pairs:
        max_ctx = max(1, max_length - len(cont))
        ctx = ctx[-max_ctx:]
        full = (ctx + cont)[-max_length:]
        start = max(0, len(full) - len(cont) - 1)
        seqs.append(full)
        starts.append(start)

    width = max(len(x) for x in seqs) - 1
    inputs = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
    targets = torch.zeros_like(inputs)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, (seq, start) in enumerate(zip(seqs, starts)):
        n = len(seq) - 1
        inputs[row, :n] = torch.tensor(seq[:-1], device=device)
        targets[row, :n] = torch.tensor(seq[1:], device=device)
        mask[row, start:n] = True

    amp = device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(inputs)
    token_lp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    totals = (token_lp * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    means = totals / counts
    return totals.cpu().tolist(), means.cpu().tolist()


@torch.no_grad()
def perplexity_from_ids(
    model,
    ids: Sequence[int],
    max_length: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Sliding-window perplexity; count each token once while retaining context."""
    if len(ids) < 2:
        raise ValueError("Perplexity corpus has fewer than two tokens")
    total_nll = 0.0
    total_tokens = 0
    windows = []
    previous_end = 0
    for begin in range(0, len(ids) - 1, stride):
        end = min(begin + max_length, len(ids))
        begin = max(0, end - max_length)
        target_start = max(1, previous_end - begin)
        windows.append((ids[begin:end], target_start))
        previous_end = end
        if end == len(ids):
            break

    for offset in range(0, len(windows), batch_size):
        batch = windows[offset : offset + batch_size]
        width = max(len(seq) for seq, _ in batch) - 1
        inputs = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        targets = torch.zeros_like(inputs)
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        for row, (seq, target_start) in enumerate(batch):
            n = len(seq) - 1
            inputs[row, :n] = torch.tensor(seq[:-1], device=device)
            targets[row, :n] = torch.tensor(seq[1:], device=device)
            mask[row, target_start - 1 : n] = True
        amp = device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(inputs)
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        total_nll += float((losses * mask).sum().item())
        total_tokens += int(mask.sum().item())

    loss = total_nll / total_tokens
    return {"loss": loss, "ppl": math.exp(min(loss, 20.0)), "tokens": total_tokens}


@torch.no_grad()
def perplexity_from_blocks(model, blocks: torch.Tensor, batch_size: int, device: torch.device) -> Dict[str, float]:
    """Perplexity on the exact non-overlapping blocks used during training validation."""
    total_nll = 0.0
    total_tokens = 0
    for offset in range(0, int(blocks.size(0)), batch_size):
        block = blocks[offset : offset + batch_size].long().to(device)
        inputs, targets = block[:, :-1], block[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inputs)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss.item())
        total_tokens += int(targets.numel())
        if offset and offset % (batch_size * 1000) == 0:
            print(f"[owt] blocks={offset:,}/{int(blocks.size(0)):,}", flush=True)
    avg = total_nll / total_tokens
    return {"loss": avg, "ppl": math.exp(min(avg, 20.0)), "tokens": total_tokens}


def text_to_ids(
    tokenizer: GPT2TokenizerFast,
    texts: Iterable[str],
    max_tokens: int = 0,
) -> List[int]:
    """Tokenize documents with one EOS per document and an optional token cap."""
    ids: List[int] = []
    eos = tokenizer.eos_token_id
    for text in texts:
        part = encode(tokenizer, text)
        if not part:
            continue
        if max_tokens:
            remaining = max_tokens - len(ids)
            if remaining <= 0:
                break
            if len(part) + 1 > remaining:
                ids.extend(part[:remaining])
                break
        ids.extend(part)
        ids.append(eos)
    return ids


def load_tinystories_texts(data_dir: Path, limit: int) -> List[str]:
    """Load the official TinyStories validation split from local text or Parquet."""
    text_path = data_dir / "tinystories_valid.txt"
    if text_path.is_file():
        raw = text_path.read_text(encoding="utf-8", errors="replace")
        texts = [item.strip() for item in raw.split("<|endoftext|>") if item.strip()]
        return texts[:limit] if limit else texts

    rows = local_or_hf_rows(
        data_dir,
        "tinystories_validation.parquet",
        lambda: load_dataset("roneneldan/TinyStories", split="validation"),
    )
    rows = dataset_slice(rows, limit)
    return [row["text"] for row in rows] if isinstance(rows, list) else list(rows["text"])


def dataset_slice(ds, limit: int):
    if not limit:
        return ds
    if isinstance(ds, list):
        return ds[:limit]
    return ds.select(range(min(limit, len(ds))))


def parquet_rows(path: Path):
    return pq.read_table(path).to_pylist()


def load_blimp_rows(data_dir: Path):
    """Load a version-pinned local BLiMP copy prepared for this benchmark."""
    parquet = data_dir / "blimp_all.parquet"
    if parquet.is_file():
        return parquet_rows(parquet)

    jsonl_dir = data_dir / "blimp"
    files = sorted(path for path in jsonl_dir.glob("*.jsonl") if not path.name.startswith("._"))
    if not files:
        raise RuntimeError(
            f"BLiMP data not found: expected {parquet} or JSONL files under {jsonl_dir}"
        )
    rows = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("UID", path.stem)
            rows.append(row)
    return rows


def local_or_hf_rows(data_dir: Path, filename: str, loader):
    path = data_dir / filename
    if path.is_file():
        return parquet_rows(path)
    if load_dataset is None:
        raise RuntimeError(f"Missing local benchmark file and datasets package: {path}")
    return loader()


def evaluate_lambada(model, tokenizer, cfg, device, batch_size, limit, data_dir: Path) -> Dict[str, float]:
    ds = local_or_hf_rows(
        data_dir,
        "lambada_test.parquet",
        lambda: load_dataset("EleutherAI/lambada_openai", "default", split="test"),
    )
    ds = dataset_slice(ds, limit)
    correct = 0
    total = 0
    for start in range(0, len(ds), batch_size):
        batch = ds[start : start + batch_size]
        texts = [row["text"] for row in batch] if isinstance(batch, list) else batch["text"]
        for text in texts:
            match = re.search(r"\s+\S+\s*$", text)
            if not match:
                continue
            context, continuation = text[: match.start()], text[match.start() :].rstrip()
            ctx, target = encode_pair(tokenizer, context, continuation)
            max_ctx = max(1, cfg.max_seq_len - len(target))
            generated: List[int] = []
            prefix = ctx[-max_ctx:]
            for _ in range(len(target)):
                x = torch.tensor([(prefix + generated)[-cfg.max_seq_len :]], device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(x)
                generated.append(int(logits[0, -1].argmax().item()))
            correct += int(generated == target)
            total += 1
    return {"acc": correct / total, "correct": correct, "examples": total}


def evaluate_choices(model, tokenizer, cfg, device, batch_size, limit, task: str, data_dir: Path) -> Dict[str, float]:
    if task == "hellaswag":
        ds = local_or_hf_rows(
            data_dir,
            "hellaswag_validation.parquet",
            lambda: load_dataset("Rowan/hellaswag", split="validation"),
        )
    elif task == "piqa":
        jsonl = data_dir / "piqa_valid.jsonl"
        labels = data_dir / "piqa_valid_labels.lst"
        if jsonl.is_file() and labels.is_file():
            rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            ys = [int(line) for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != len(ys):
                raise RuntimeError("PIQA examples and labels have different lengths")
            ds = [dict(row, label=label) for row, label in zip(rows, ys)]
        else:
            if load_dataset is None:
                raise RuntimeError("Local PIQA files are missing")
            ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    elif task == "arc_easy":
        ds = local_or_hf_rows(
            data_dir,
            "arc_easy_test.parquet",
            lambda: load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"),
        )
    else:
        raise ValueError(task)
    ds = dataset_slice(ds, limit)

    correct = 0
    total = 0
    for index in range(len(ds)):
        row = ds[index]
        if task == "hellaswag":
            context = row.get("ctx") or (row["ctx_a"] + " " + row["ctx_b"].capitalize())
            choices = row["endings"]
            label = int(row["label"])
        elif task == "piqa":
            context = row["goal"]
            choices = [row["sol1"], row["sol2"]]
            label = int(row["label"])
        else:
            context = "Question: " + row["question"] + "\nAnswer:"
            choices = row["choices"]["text"]
            labels = row["choices"]["label"]
            answer = str(row["answerKey"])
            label = labels.index(answer)

        pairs = [encode_pair(tokenizer, context, " " + choice.lstrip()) for choice in choices]
        _, normalized = continuation_scores(model, pairs, cfg.max_seq_len, device)
        prediction = max(range(len(normalized)), key=normalized.__getitem__)
        correct += int(prediction == label)
        total += 1
        if (index + 1) % 1000 == 0:
            print(f"[{task}] {index + 1}/{len(ds)} acc={correct/total:.4f}", flush=True)
    return {"acc_norm": correct / total, "correct": correct, "examples": total}


def evaluate_blimp(model, tokenizer, cfg, device, batch_size, limit, data_dir: Path) -> Dict[str, float]:
    """BLiMP forced-choice accuracy using full-sentence autoregressive likelihood."""
    rows = dataset_slice(load_blimp_rows(data_dir), limit)
    correct = 0
    total = 0
    group_correct: Dict[str, int] = {}
    group_total: Dict[str, int] = {}

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        pairs = []
        for row in batch:
            pairs.append(encode_pair(tokenizer, "", row["sentence_good"]))
            pairs.append(encode_pair(tokenizer, "", row["sentence_bad"]))
        totals, _ = continuation_scores(model, pairs, cfg.max_seq_len, device)

        for index, row in enumerate(batch):
            uid = str(row.get("UID", "unknown"))
            hit = int(totals[2 * index] > totals[2 * index + 1])
            correct += hit
            total += 1
            group_correct[uid] = group_correct.get(uid, 0) + hit
            group_total[uid] = group_total.get(uid, 0) + 1

        if total and total % 10000 == 0:
            print(f"[blimp] {total}/{len(rows)} acc={correct/total:.4f}", flush=True)

    per_uid = {
        uid: group_correct[uid] / group_total[uid]
        for uid in sorted(group_total)
    }
    return {
        "acc": correct / total,
        "correct": correct,
        "examples": total,
        "protocol": "full-sentence total log-likelihood",
        "per_uid": per_uid,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    benchmark_data_dir = Path(args.benchmark_data_dir)
    results = json.loads(output.read_text()) if args.resume and output.is_file() else {}

    model, cfg, checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
    current_metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "size": cfg.size,
        "model_type": cfg.model_type,
        "embed_dim": cfg.embed_dim,
        "max_seq_len": cfg.max_seq_len,
    }
    previous_metadata = results.get("metadata", {})
    checkpoint_changed = bool(previous_metadata) and (
        previous_metadata.get("checkpoint") != current_metadata["checkpoint"]
        or int(previous_metadata.get("checkpoint_step", -1)) != current_metadata["checkpoint_step"]
    )
    if checkpoint_changed:
        print("[resume] checkpoint changed; discarding stale task metrics", flush=True)
        results = {}
    results.setdefault("metadata", {}).update(current_metadata)
    results.setdefault("tasks", {})
    atomic_json_dump(results, output)

    for task in args.tasks:
        if args.resume and task in results["tasks"]:
            print(f"[skip] {task}", flush=True)
            continue
        print(f"[start] {task}", flush=True)
        started = time.time()
        if task == "owt":
            cache = Path(args.owt_valid).with_name(
                f"{Path(args.owt_valid).name}.gpt2.seq{cfg.max_seq_len}.full.blocks.pt"
            )
            if cache.is_file():
                payload = torch.load(cache, map_location="cpu", mmap=True)
                value = perplexity_from_blocks(model, payload["blocks"], args.batch_size, device)
                value["protocol"] = "training-validation blocks"
            else:
                text = Path(args.owt_valid).read_text(encoding="utf-8", errors="replace")
                ids = text_to_ids(tokenizer, [text])
                value = perplexity_from_ids(model, ids, cfg.max_seq_len, args.ppl_stride, args.batch_size, device)
                value["protocol"] = f"sliding window, stride={args.ppl_stride}"
        elif task == "wikitext2":
            rows = local_or_hf_rows(
                benchmark_data_dir,
                "wikitext2_test.parquet",
                lambda: load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test"),
            )
            texts = [row["text"] for row in rows] if isinstance(rows, list) else rows["text"]
            ids = text_to_ids(tokenizer, texts)
            value = perplexity_from_ids(model, ids, cfg.max_seq_len, args.ppl_stride, args.batch_size, device)
        elif task == "tinystories":
            texts = load_tinystories_texts(benchmark_data_dir, args.limit)
            ids = text_to_ids(tokenizer, texts, max_tokens=args.tinystories_max_tokens)
            value = perplexity_from_ids(model, ids, cfg.max_seq_len, args.ppl_stride, args.batch_size, device)
            value["documents_available"] = len(texts)
            value["token_cap"] = args.tinystories_max_tokens
            value["protocol"] = (
                f"one EOS per story, GPT-2 BPE sliding window, stride={args.ppl_stride}"
            )
        elif task == "ptb":
            path = benchmark_data_dir / "ptb_test.txt"
            if not path.is_file():
                raise RuntimeError(f"PTB test data not found: {path}")
            texts = path.read_text(encoding="utf-8").splitlines()
            ids = text_to_ids(tokenizer, texts)
            value = perplexity_from_ids(model, ids, cfg.max_seq_len, args.ppl_stride, args.batch_size, device)
            value["protocol"] = f"GPT-2 BPE sliding window, stride={args.ppl_stride}"
        elif task == "blimp":
            value = evaluate_blimp(
                model, tokenizer, cfg, device, args.batch_size, args.limit, benchmark_data_dir
            )
        elif task == "lambada":
            value = evaluate_lambada(model, tokenizer, cfg, device, args.batch_size, args.limit, benchmark_data_dir)
        else:
            value = evaluate_choices(model, tokenizer, cfg, device, args.batch_size, args.limit, task, benchmark_data_dir)
        value["seconds"] = time.time() - started
        results["tasks"][task] = value
        accs = []
        for name in ("lambada", "hellaswag", "piqa", "arc_easy"):
            if name in results["tasks"]:
                accs.append(results["tasks"][name].get("acc", results["tasks"][name].get("acc_norm")))
        if len(accs) == 4:
            results["zero_shot_average"] = sum(accs) / 4.0
        atomic_json_dump(results, output)
        print(f"[done] {task}: {value}", flush=True)


if __name__ == "__main__":
    main()
