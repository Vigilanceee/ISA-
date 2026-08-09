#!/usr/bin/env python3
"""GPT2 CIM training with multi-component matching loss.

Three-component loss (Phase 1 – matching):
  L = λ1·MSE(FFN_phys, FFN_dig)
    + λ2·MSE(BlockOut_phys, BlockOut_dig)
    + λ3·KL(Logits_dig || Logits_phys)

Phase 2 – LM training: standard cross-entropy next-token prediction.

Supports:
  - 3 model sizes: tiny, mid, small
  - 3 model types: standard (digital baseline), hybrid, physical
  - DDP (torchrun) with gradient all-reduce
  - Checkpoint save/resume
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2TokenizerFast

from isa.language.config import ModelConfig, TrainConfig, InitConfig, ModelSize, ModelType
from isa.language.models import GPT2Model, create_model
from isa.operators.cim import CIMLinear, VoltageMapping


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class BlockDataset(Dataset):
    def __init__(self, blocks: torch.Tensor):
        self.blocks = blocks

    def __len__(self) -> int:
        return int(self.blocks.size(0))

    def __getitem__(self, idx: int):
        block = self.blocks[idx]
        return block[:-1].long(), block[1:].long()


def load_text(path: str, max_chars: int) -> str:
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(max_chars) if max_chars > 0 else handle.read()


def tokenize_to_blocks(tokenizer: GPT2TokenizerFast, text: str, seq_len: int,
                       chunk_chars: int = 20000, chunk_batch_size: int = 64) -> torch.Tensor:
    token_ids: List[int] = []
    eos = tokenizer.eos_token_id
    start = 0
    while start < len(text):
        chunks = []
        for _ in range(max(1, chunk_batch_size)):
            if start >= len(text):
                break
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start, end),
                    text.rfind(" ", start, end),
                )
                if boundary > start + chunk_chars // 2:
                    end = boundary
            chunks.append(text[start:end])
            start = max(end, start + 1)

        encoded = tokenizer(
            chunks,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        for ids in encoded:
            if ids:
                token_ids.extend(ids)
                token_ids.append(eos)
    block_len = seq_len + 1
    usable = (len(token_ids) // block_len) * block_len
    if usable < block_len:
        raise RuntimeError(f"Not enough tokens for seq_len={seq_len}.")
    # GPT-2 token IDs fit in int32. BlockDataset converts each sampled block to
    # int64, so this halves cache and host-memory size without changing inputs.
    return torch.tensor(token_ids[:usable], dtype=torch.int32).view(-1, block_len)


def _token_cache_path(text_path: str, seq_len: int, max_chars: int) -> Path:
    source = Path(text_path)
    extent = f"max{max_chars}" if max_chars > 0 else "full"
    return source.with_name(f"{source.name}.gpt2.seq{seq_len}.{extent}.blocks.pt")


def _load_token_cache(path: Path):
    return torch.load(path, map_location="cpu", mmap=True)


def load_or_create_token_blocks(
    tokenizer: GPT2TokenizerFast,
    text_path: str,
    seq_len: int,
    max_chars: int,
) -> torch.Tensor:
    """Build once on rank 0, then mmap the same token blocks on every rank."""
    source = Path(text_path)
    stat = source.stat()
    metadata = {
        "format": 1,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "max_chars": int(max_chars),
        "seq_len": int(seq_len),
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token_id": int(tokenizer.eos_token_id),
    }
    cache_path = _token_cache_path(text_path, seq_len, max_chars)
    payload = None
    rebuilt = False

    if is_main():
        if cache_path.is_file():
            try:
                candidate = _load_token_cache(cache_path)
                if candidate.get("metadata") == metadata:
                    payload = candidate
                    print(f"[data] token cache hit: {cache_path} blocks={payload['blocks'].size(0):,}")
            except Exception as exc:
                print(f"[data] ignoring unreadable token cache {cache_path}: {exc}")

        if payload is None:
            print(f"[data] building token cache once on rank 0: {cache_path}")
            text = load_text(text_path, max_chars)
            blocks = tokenize_to_blocks(tokenizer, text, seq_len)
            payload = {"metadata": metadata, "blocks": blocks}
            temp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
            torch.save(payload, temp_path)
            os.replace(temp_path, cache_path)
            print(f"[data] wrote token cache: {cache_path} blocks={blocks.size(0):,}")
            rebuilt = True
            del text, blocks

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not is_main() or rebuilt:
        if rebuilt:
            del payload
            gc.collect()
        payload = _load_token_cache(cache_path)
    if payload.get("metadata") != metadata:
        raise RuntimeError(f"Token cache metadata mismatch after synchronization: {cache_path}")
    return payload["blocks"]


# ═══════════════════════════════════════════════════════════════════════════
# DDP helpers
# ═══════════════════════════════════════════════════════════════════════════

def is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


# ═══════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_ffn_mse(intermed_phys: List[Dict], intermed_dig: List[Dict]) -> torch.Tensor:
    """Per-layer FFN output MSE."""
    loss = torch.tensor(0.0, device=intermed_phys[0]["ffn_output"].device)
    for ip, idg in zip(intermed_phys, intermed_dig):
        loss = loss + F.mse_loss(ip["ffn_output"].float(), idg["ffn_output"].float())
    return loss / len(intermed_phys)


def compute_block_mse(intermed_phys: List[Dict], intermed_dig: List[Dict]) -> torch.Tensor:
    """Per-layer block output MSE."""
    loss = torch.tensor(0.0, device=intermed_phys[0]["block_output"].device)
    for ip, idg in zip(intermed_phys, intermed_dig):
        loss = loss + F.mse_loss(ip["block_output"].float(), idg["block_output"].float())
    return loss / len(intermed_phys)


class _ChunkedLogitKL(torch.autograd.Function):
    """Exact token-mean distillation KL with bounded temporary memory."""

    @staticmethod
    def forward(ctx, logits_phys: torch.Tensor, logits_dig: torch.Tensor,
                temperature: float, chunk_tokens: int):
        if logits_phys.shape != logits_dig.shape:
            raise ValueError(
                f"Student/teacher logits must have the same shape, got "
                f"{tuple(logits_phys.shape)} and {tuple(logits_dig.shape)}"
            )
        if temperature <= 0:
            raise ValueError("KL temperature must be positive")

        vocab_size = logits_phys.size(-1)
        phys_flat = logits_phys.reshape(-1, vocab_size)
        dig_flat = logits_dig.reshape(-1, vocab_size)
        token_count = phys_flat.size(0)
        chunk_tokens = max(1, int(chunk_tokens))
        total = torch.zeros((), device=logits_phys.device, dtype=torch.float32)
        for start in range(0, token_count, chunk_tokens):
            end = min(start + chunk_tokens, token_count)
            log_student = F.log_softmax(phys_flat[start:end].float() / temperature, dim=-1)
            teacher = F.softmax(dig_flat[start:end].float() / temperature, dim=-1)
            total.add_(F.kl_div(log_student, teacher, reduction="sum"))

        ctx.save_for_backward(logits_phys, logits_dig)
        ctx.temperature = float(temperature)
        ctx.chunk_tokens = chunk_tokens
        ctx.token_count = token_count
        return total * (temperature * temperature) / max(token_count, 1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits_phys, logits_dig = ctx.saved_tensors
        vocab_size = logits_phys.size(-1)
        phys_flat = logits_phys.reshape(-1, vocab_size)
        dig_flat = logits_dig.reshape(-1, vocab_size)
        grad_phys = torch.empty_like(logits_phys)
        grad_flat = grad_phys.view(-1, vocab_size)
        scale = (
            grad_output.float()
            * ctx.temperature
            / max(ctx.token_count, 1)
        )

        for start in range(0, ctx.token_count, ctx.chunk_tokens):
            end = min(start + ctx.chunk_tokens, ctx.token_count)
            student = F.softmax(
                phys_flat[start:end].float() / ctx.temperature, dim=-1
            )
            teacher = F.softmax(
                dig_flat[start:end].float() / ctx.temperature, dim=-1
            )
            grad_flat[start:end].copy_(((student - teacher) * scale).to(logits_phys.dtype))
        return grad_phys, None, None, None


def compute_logit_kl(logits_phys: torch.Tensor, logits_dig: torch.Tensor,
                     T: float = 1.0, chunk_tokens: int = 4096) -> torch.Tensor:
    """Token-mean KL divergence with the frozen digital model as teacher."""
    return _ChunkedLogitKL.apply(logits_phys, logits_dig, T, chunk_tokens)


# ═══════════════════════════════════════════════════════════════════════════
# ViT-style initialization
# ═══════════════════════════════════════════════════════════════════════════

def load_shared_weights(model: GPT2Model, baseline_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Copy backbone weights from digital baseline."""
    shared = {}
    model_dict = model.state_dict()
    for key in model_dict:
        if key in baseline_state:
            if model_dict[key].shape == baseline_state[key].shape:
                model_dict[key].copy_(baseline_state[key])
                shared[key] = 1
    model.load_state_dict(model_dict)
    return {"shared_params": len(shared)}


def _collect_module_inputs(model: GPT2Model, modules: List[Tuple[str, nn.Module]],
                           loader: DataLoader, device: torch.device,
                           max_batches: int, max_rows: int, amp_dtype) -> Dict[str, torch.Tensor]:
    """Collect bounded, flattened inputs with forward pre-hooks."""
    collected: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            remaining = max_rows - sum(t.size(0) for t in collected[name])
            if remaining > 0:
                collected[name].append(x[:remaining].float().cpu())
        return hook

    for name, module in modules:
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                model(inputs)
    model.train(was_training)

    for handle in handles:
        handle.remove()
    return {name: torch.cat(parts, dim=0) for name, parts in collected.items() if parts}


@torch.no_grad()
def calibrate_voltage_mappings(model: GPT2Model, loader: DataLoader, device: torch.device,
                               init_cfg: InitConfig, amp_dtype) -> Dict:
    """ViT-style quantile calibration of every external voltage mapping."""
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, VoltageMapping)]
    records = {}
    q_high = min(max(float(init_cfg.voltage_map_quantile), 0.5), 0.9999)
    q_low = max(0.0, 1.0 - q_high)

    for _ in range(max(1, int(init_cfg.voltage_map_iters))):
        inputs = _collect_module_inputs(
            model, modules, loader, device, init_cfg.init_max_batches,
            init_cfg.init_max_rows, amp_dtype,
        )
        for name, module in modules:
            if name not in inputs:
                continue
            data = inputs[name].flatten().float()
            lo = float(torch.quantile(data, q_low).item())
            hi = float(torch.quantile(data, q_high).item())
            spread = max(hi - lo, 1e-6)
            out_low = max(0.0, min(init_cfg.voltage_map_low, module.voltage_max))
            out_high = max(out_low + 1e-6, min(init_cfg.voltage_map_high, module.voltage_max))
            scale = (out_high - out_low) / spread
            shift = out_low - lo * scale
            module.set_affine(scale, shift)
            records[name] = (scale, shift, lo, hi)

    return {
        "voltage_map_layers": len(records),
        "voltage_map_scale_mean": (
            sum(value[0] for value in records.values()) / max(len(records), 1)
        ),
    }


@torch.no_grad()
def centered_reverse_tia_init(model: GPT2Model, loader: DataLoader, device: torch.device,
                              init_cfg: InitConfig, amp_dtype) -> Dict:
    """Match the ViT initialization: center Vth from data, then calibrate R_TIA."""
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, CIMLinear)]
    for _, module in modules:
        module.reset_vth(center=None, eps_k=init_cfg.ekv_eps_k, symmetric=True)
        module.set_r_tia(float(module.cfg["R_TIA"]))

    inputs = _collect_module_inputs(
        model, modules, loader, device, init_cfg.init_max_batches,
        init_cfg.init_max_rows, amp_dtype,
    )
    for name, module in modules:
        if name in inputs:
            module.reset_vth(
                center=float(inputs[name].mean().item()),
                eps_k=init_cfg.ekv_eps_k,
                symmetric=True,
            )

    inputs = _collect_module_inputs(
        model, modules, loader, device, init_cfg.init_max_batches,
        init_cfg.init_max_rows, amp_dtype,
    )
    rtias = {}
    clip_ratios = {}
    for name, module in modules:
        if name not in inputs:
            continue
        x = inputs[name].to(device, non_blocking=True)
        idiff = module.estimate_idiff(x, max_rows=init_cfg.init_max_rows).detach().float()
        q = torch.quantile(idiff.abs().flatten(), init_cfg.ekv_quantile).clamp_min(1e-12)
        clip_limit = max(abs(float(module.cfg["V_signed_min"])),
                         abs(float(module.cfg["V_signed_max"])))
        target_amplitude = clip_limit * init_cfg.ekv_rho
        module.set_r_tia(float(target_amplitude / q.item()))
        r_tia = float(module.r_tia.detach().item())
        rtias[name] = r_tia
        clip_ratios[name] = float(((idiff * r_tia).abs() > clip_limit).float().mean().item())

    return {
        "ekv_layers": len(rtias),
        "ekv_r_tia_mean": sum(rtias.values()) / max(len(rtias), 1),
        "ekv_clip_mean": sum(clip_ratios.values()) / max(len(clip_ratios), 1),
    }


def collect_baseline_pairs(digital_model: GPT2Model, loader: DataLoader,
                           device: torch.device, max_batches: int, max_rows: int,
                           amp_dtype) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Collect bounded FFN and block outputs from each digital block."""
    digital_model.eval()
    pairs: Dict[str, Tuple[List, List]] = {}

    with torch.no_grad():
        rows = 0
        for batch_idx, (inputs, _) in enumerate(loader):
            if batch_idx >= max_batches or rows >= max_rows:
                break
            inputs = inputs.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits, intermed = digital_model(inputs, need_intermediate=True)
            for block_idx, info in enumerate(intermed):
                key = f"block_{block_idx}"
                if key not in pairs:
                    pairs[key] = ([], [])
                pairs[key][0].append(info["ffn_output"].detach().cpu())
                pairs[key][1].append(info["block_output"].detach().cpu())
            rows += inputs.size(0)

    result = {}
    for key, (ffn_list, block_list) in pairs.items():
        result[key] = (
            torch.cat(ffn_list, dim=0)[:max_rows],
            torch.cat(block_list, dim=0)[:max_rows],
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    totals = torch.tensor([total_loss, total_tokens], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float((totals[0] / totals[1].clamp_min(1.0)).item())


def save_checkpoint(model: nn.Module, optimizer, step: int, loss: float, path: str,
                    model_cfg: ModelConfig, train_cfg: TrainConfig,
                    best_val_loss: float = float("inf"), is_best: bool = False):
    state = {
        "step": step,
        "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
        "best_val_loss": best_val_loss,
        "model_config": model_cfg,
        "train_config": train_cfg,
    }
    torch.save(state, path)
    if is_best:
        best_path = str(Path(path).parent / "best.pt")
        torch.save(state, best_path)


def load_checkpoint(path: str, model: nn.Module, optimizer, device: torch.device):
    state = torch.load(path, map_location=device)
    if hasattr(model, "module"):
        model.module.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    return state


def set_trainable_scope(model: nn.Module, match_only: bool) -> int:
    """Match only physical FFNs; LM training restores the complete model."""
    trainable = 0
    for name, param in model.named_parameters():
        is_ffn = name.startswith("ffn.") or ".ffn." in name
        enabled = not match_only or is_ffn
        param.requires_grad_(enabled)
        if enabled:
            trainable += param.numel()
    return trainable


def train(args: argparse.Namespace):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── Config ──────────────────────────────────────────────────────
    model_cfg = ModelConfig.from_preset(args.size, args.model_type, vocab_size=50257, max_seq_len=args.seq_len)
    if args.ffn_dim > 0: model_cfg.ffn_dim = args.ffn_dim
    train_cfg = TrainConfig(
        eval_freq=args.eval_freq, save_freq=args.save_freq, print_freq=args.print_freq,
        tokenizer_name=args.tokenizer_name, train_text=args.train_text, val_text=args.val_text,
        baseline_checkpoint=args.baseline_checkpoint, output_dir=args.output_dir,
        seq_len=args.seq_len, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        max_steps=args.max_steps, lambda_ffn=args.lambda_ffn,
        lambda_block=args.lambda_block, lambda_logit=args.lambda_logit,
        kl_chunk_tokens=args.kl_chunk_tokens,
        match_steps=args.match_steps, lm_steps=args.lm_steps,
        grad_clip=args.grad_clip, amp=args.amp, seed=args.seed,
    )
    init_cfg = InitConfig()

    set_seed(args.seed + rank)

    # ── Tokenizer & Data ────────────────────────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained(train_cfg.tokenizer_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_blocks = load_or_create_token_blocks(
        tokenizer, train_cfg.train_text, train_cfg.seq_len, args.max_train_chars
    )
    val_loader = None
    if train_cfg.val_text:
        val_blocks = load_or_create_token_blocks(
            tokenizer, train_cfg.val_text, train_cfg.seq_len, 0
        )
        val_dataset = BlockDataset(val_blocks)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                                sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True)

    train_sampler = DistributedSampler(train_blocks) if world_size > 1 else None
    train_loader = DataLoader(BlockDataset(train_blocks), batch_size=train_cfg.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)

    # ── Model ────────────────────────────────────────────────────────
    amp_dtype = torch.bfloat16 if train_cfg.amp and train_cfg.amp_dtype == "bf16" else None
    model = create_model(model_cfg).to(device)

    # Digital baseline for matching loss (Phase 1)
    digital_model = None
    if (model_cfg.model_type in ("hybrid", "physical") and train_cfg.match_steps > 0
            and not train_cfg.baseline_checkpoint):
        raise ValueError("Hybrid/physical matching requires --baseline_checkpoint")

    if model_cfg.model_type in ("hybrid", "physical") and train_cfg.baseline_checkpoint:
        digital_cfg = ModelConfig.from_preset(args.size, "standard", vocab_size=50257, max_seq_len=args.seq_len)
        digital_model = create_model(digital_cfg).to(device)
        # Keep the checkpoint (including its optimizer state) off GPU; only
        # model tensors are copied into the student and frozen teacher.
        digital_state = torch.load(train_cfg.baseline_checkpoint, map_location="cpu")
        digital_model.load_state_dict(digital_state["model_state"])
        digital_model.eval()
        for p in digital_model.parameters():
            p.requires_grad = False

        if not args.resume:
            load_shared_weights(model, digital_state["model_state"])

            # Only rank 0 initializes; DDP broadcasts its parameters and buffers.
            if is_main():
                print("[init] Calibrating VoltageMappings...")
                voltage_metrics = calibrate_voltage_mappings(
                    model, train_loader, device, init_cfg, amp_dtype,
                )
                print(f"[init] VoltageMappings: {voltage_metrics}")
                print("[init] Initializing centered Vth and reverse-calibrated R_TIA...")
                ekv_metrics = centered_reverse_tia_init(
                    model, train_loader, device, init_cfg, amp_dtype,
                )
                print(f"[init] EKV: {ekv_metrics}")
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        del digital_state

    # ── Resume model and determine phase before building DDP/optimizer ──
    start_step = 0
    best_val_loss = float("inf")
    resume_state = None
    if args.resume:
        resume_state = load_checkpoint(args.resume, model, None, device)
        start_step = int(resume_state.get("step", 0))
        best_val_loss = float(resume_state.get(
            "best_val_loss", resume_state.get("loss", float("inf")),
        ))

    has_match_phase = train_cfg.match_steps > 0 and digital_model is not None
    total_steps = train_cfg.match_steps + train_cfg.lm_steps if has_match_phase else train_cfg.lm_steps
    phase = "match" if has_match_phase and start_step < train_cfg.match_steps else "lm"
    match_step = min(start_step, train_cfg.match_steps) if has_match_phase else 0
    lm_step = max(0, start_step - train_cfg.match_steps) if has_match_phase else start_step

    # Construct DDP with only FFN parameters during matching. This avoids
    # allocating and all-reducing gradients for frozen attention/embedding
    # parameters. DDP is rebuilt once when full-model LM training begins.
    set_trainable_scope(model, match_only=(phase == "match"))
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Optimizer & Scheduler ───────────────────────────────────────
    def build_optimizer(match_only: bool):
        base_model = model.module if hasattr(model, "module") else model
        weight_decay = 0.0 if match_only else train_cfg.weight_decay
        groups = base_model.configure_optimizer_groups(
            weight_decay, train_cfg.lr, ffn_only=match_only,
        )
        opt = torch.optim.AdamW(groups, lr=train_cfg.lr, betas=train_cfg.adam_betas)
        params = [param for group in opt.param_groups for param in group["params"]]
        return opt, [group["lr"] for group in opt.param_groups], params

    optimizer, base_lrs, optimizer_params = build_optimizer(match_only=(phase == "match"))

    if resume_state is not None:
        try:
            optimizer.load_state_dict(resume_state["optimizer_state"])
        except ValueError as exc:
            if is_main():
                print(f"[resume] optimizer state is incompatible with phase={phase}; starting it fresh: {exc}")
        if is_main():
            print(f"[resume] step={start_step} best_val_loss={best_val_loss:.4f} phase={phase}")

    def lm_lr_scale(lm_step: int) -> float:
        if lm_step < train_cfg.warmup_steps:
            return (lm_step + 1) / max(train_cfg.warmup_steps, 1)
        progress = (lm_step - train_cfg.warmup_steps) / max(train_cfg.lm_steps - train_cfg.warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def match_lr_scale(match_step: int) -> float:
        if train_cfg.match_steps <= 0:
            return 1.0
        warmup = max(1, min(train_cfg.warmup_steps, train_cfg.match_steps))
        return min(1.0, (match_step + 1) / warmup)

    def set_lr(scale: float) -> float:
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * scale
        return optimizer.param_groups[0]["lr"]

    # ── Output dir ──────────────────────────────────────────────────
    out_dir = Path(train_cfg.output_dir) / f"{args.size}_{args.model_type}"
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ───────────────────────────────────────────────
    model.train()
    global_step = start_step
    total_loss = 0.0
    log_loss = 0.0
    start_time = time.time()
    last_print = global_step
    current_lr = set_lr(match_lr_scale(match_step) if phase == "match" else lm_lr_scale(lm_step))

    if is_main():
        optimized_numel = sum(param.numel() for param in optimizer_params)
        print(f"[train] size={args.size} type={args.model_type} "
              f"steps={total_steps} phase={phase} optimized_params={optimized_numel:,} "
              f"device={device} ddp={world_size}")

    for epoch in range(math.ceil(total_steps / len(train_loader))):
        if global_step >= total_steps:
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for inputs, targets in train_loader:
            if global_step >= total_steps:
                break

            # Phase transition. LM has its own local LR schedule starting at lm_step=0.
            if has_match_phase and global_step >= train_cfg.match_steps and phase == "match":
                phase = "lm"
                lm_step = 0
                log_loss = 0.0
                last_print = global_step
                digital_model = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                # ViT-style initialization ends here. LM training gets a fresh
                # optimizer over the complete model and its own LR schedule.
                if world_size > 1:
                    dist.barrier()
                    model = model.module
                set_trainable_scope(model, match_only=False)
                if world_size > 1:
                    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
                optimizer, base_lrs, optimizer_params = build_optimizer(match_only=False)
                if is_main():
                    optimized_numel = sum(param.numel() for param in optimizer_params)
                    print(f"[phase] switching to LM training at step {global_step}; "
                          f"optimized_params={optimized_numel:,}")

            current_lr = set_lr(match_lr_scale(match_step) if phase == "match" else lm_lr_scale(lm_step))

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                if phase == "match" and digital_model is not None:
                    # ── Phase 1: Multi-component matching ──
                    logits_phys, intermed_phys = model(inputs, need_intermediate=True)
                    with torch.no_grad():
                        logits_dig, intermed_dig = digital_model(inputs, need_intermediate=True)

                    loss_ffn = compute_ffn_mse(intermed_phys, intermed_dig)
                    loss_block = compute_block_mse(intermed_phys, intermed_dig)
                    loss_logit = compute_logit_kl(
                        logits_phys, logits_dig, chunk_tokens=train_cfg.kl_chunk_tokens
                    )

                    loss = (train_cfg.lambda_ffn * loss_ffn +
                            train_cfg.lambda_block * loss_block +
                            train_cfg.lambda_logit * loss_logit)

                    if global_step % args.print_freq == 0 and is_main():
                        print(f"[match {global_step}/{train_cfg.match_steps}] "
                              f"ffn={loss_ffn.item():.4f} block={loss_block.item():.4f} "
                              f"logit={loss_logit.item():.4f} total={loss.item():.4f}")
                else:
                    # ── Phase 2: LM training ──
                    logits = model(inputs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            # Match-only optimization intentionally excludes attention,
            # embeddings, layer norms, and lm_head from optimizer updates.
            model.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(optimizer_params, train_cfg.grad_clip)
            optimizer.step()
            for module in (model.module if hasattr(model, "module") else model).modules():
                if isinstance(module, CIMLinear):
                    module.clamp_vth()

            total_loss += loss.item()
            log_loss += loss.item()
            if phase == "match":
                match_step += 1
            else:
                lm_step += 1
            global_step += 1

            if phase == "match":
                del logits_phys, logits_dig, intermed_phys, intermed_dig
                del loss_ffn, loss_block, loss_logit
            else:
                del logits
            del loss

            # Logging
            if global_step % train_cfg.print_freq == 0 and is_main() and global_step > last_print:
                avg_loss = log_loss / (global_step - last_print)
                elapsed = time.time() - start_time
                if phase == "lm":
                    ppl = math.exp(min(avg_loss, 20))
                    print(f"[{global_step}/{total_steps}] loss={avg_loss:.4f} ppl={ppl:.1f} "
                          f"lr={current_lr:.2e} phase={phase} lm_step={lm_step} time={elapsed:.0f}s")
                else:
                    print(f"[{global_step}/{total_steps}] match_loss={avg_loss:.4f} "
                          f"lr={current_lr:.2e} phase={phase} match_step={match_step} time={elapsed:.0f}s")
                log_loss = 0.0
                last_print = global_step

            # Validation
            if val_loader is not None and global_step % train_cfg.eval_freq == 0:
                val_loss = validate(model.module if hasattr(model, "module") else model,
                                    val_loader, device, amp_dtype)
                val_ppl = math.exp(min(val_loss, 20))
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                if is_main():
                    print(f"[eval {global_step}] val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}")
                    save_checkpoint(model, optimizer, global_step, val_loss,
                                    str(out_dir / f"checkpoint_{global_step}.pt"),
                                    model_cfg, train_cfg, best_val_loss, is_best=is_best)
                model.train()

            # Periodic save
            if global_step % train_cfg.save_freq == 0 and is_main():
                save_checkpoint(model, optimizer, global_step,
                                total_loss / max(global_step - start_step, 1),
                                str(out_dir / f"checkpoint_{global_step}.pt"),
                                model_cfg, train_cfg, best_val_loss)

    # ── Final validation and save ───────────────────────────────────
    final_val_loss = None
    if val_loader is not None:
        final_val_loss = validate(model.module if hasattr(model, "module") else model,
                                  val_loader, device, amp_dtype)

    if is_main():
        final_loss = total_loss / max(global_step - start_step, 1)
        save_checkpoint(model, optimizer, global_step, final_loss,
                        str(out_dir / "final.pt"), model_cfg, train_cfg, best_val_loss)
        if final_val_loss is not None:
            print(f"[final] val_loss={final_val_loss:.4f} "
                  f"val_ppl={math.exp(min(final_val_loss, 20)):.1f}")

    cleanup_ddp()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="GPT2 CIM match-step v2 training")
    # Model
    p.add_argument("--size", choices=["tiny", "mid", "small"], default="tiny")
    p.add_argument("--model_type", choices=["standard", "hybrid", "physical"], default="standard")
    p.add_argument("--ffn_dim", type=int, default=0, help="Override FFN dim (0=auto)")
    # Data
    p.add_argument("--tokenizer_name", default="gpt2")
    p.add_argument("--train_text", required=True)
    p.add_argument("--val_text", default="")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--max_train_chars", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")
    # Training
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--baseline_checkpoint", default="")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=15000)
    p.add_argument("--match_steps", type=int, default=200)
    p.add_argument("--lm_steps", type=int, default=15000)
    p.add_argument("--lambda_ffn", type=float, default=1.0)
    p.add_argument("--lambda_block", type=float, default=1.0)
    p.add_argument("--lambda_logit", type=float, default=1.0)
    p.add_argument(
        "--kl_chunk_tokens",
        type=int,
        default=4096,
        help="Tokens per recomputed KL chunk; bounds match-stage temporary memory",
    )
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    # System
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_freq", type=int, default=1000)
    p.add_argument("--save_freq", type=int, default=2500)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--resume", default="")
    return p.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
