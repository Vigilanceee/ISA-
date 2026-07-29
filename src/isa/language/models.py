"""GPT2 models with Standard/Hybrid/Physical FFN variants.

Reuses the ViT-optimized CIM layer (Triton LUT + EKV kernel fusion).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from isa.language.config import ModelConfig, ModelType, PHYSICAL_CONFIG
from isa.operators.cim import CIMLinear, PhysicalFFN as CIMPhysicalFFN, VoltageMapping, clamp_signed


_FLASH_SDPA_CONFIGURED = False


def _configure_flash_sdpa() -> None:
    global _FLASH_SDPA_CONFIGURED
    if _FLASH_SDPA_CONFIGURED:
        return
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    _FLASH_SDPA_CONFIGURED = True


# ═══════════════════════════════════════════════════════════════════════════
# FFN Variants — reuse ViT CIM for physical, build hybrid on top
# ═══════════════════════════════════════════════════════════════════════════

class StandardFFN(nn.Module):
    """Digital FFN: Linear → GELU → Linear."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class HybridFFN(nn.Module):
    """Hybrid: VoltageMapping → CIMLinear (up-proj) → Digital Linear (down-proj)."""
    def __init__(self, embed_dim: int, ffn_dim: int, voltage_max: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.input_mapping = VoltageMapping(voltage_max=voltage_max)
        self.cim1 = CIMLinear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.adc_affine_scale = nn.Parameter(torch.ones(embed_dim))
        self.adc_affine_shift = nn.Parameter(torch.zeros(embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_mapping(x)
        x = self.cim1(x)
        x = self.fc2(x)
        return self.dropout(x * self.adc_affine_scale + self.adc_affine_shift)


class GPT2PhysicalFFN(nn.Module):
    """Physical FFN wrapping ViT's optimized CIMPhysicalFFN for GPT2 interface."""
    def __init__(self, embed_dim: int, ffn_dim: int, voltage_max: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.cim_ffn = CIMPhysicalFFN(embed_dim, ffn_dim, PHYSICAL_CONFIG, voltage_max)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.cim_ffn(x))


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Block & GPT2 Model
# ═══════════════════════════════════════════════════════════════════════════

class GPT2Block(nn.Module):
    """GPT2 transformer block with intermediate output hooks."""

    def __init__(self, embed_dim: int, ffn_dim: int, num_heads: int,
                 model_type: ModelType, voltage_max: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)

        if model_type == "standard":
            self.ffn = StandardFFN(embed_dim, ffn_dim, dropout)
        elif model_type == "hybrid":
            self.ffn = HybridFFN(embed_dim, ffn_dim, voltage_max, dropout)
        elif model_type == "physical":
            self.ffn = GPT2PhysicalFFN(embed_dim, ffn_dim, voltage_max, dropout)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _causal_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Use the existing MHA weights with fused scaled-dot-product attention."""
        normalized = self.ln1(x)
        qkv = F.linear(normalized, self.attn.in_proj_weight, self.attn.in_proj_bias)
        batch, tokens, _ = qkv.shape
        query, key, value = qkv.view(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        dropout_p = float(self.attn.dropout) if self.training else 0.0

        if query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
            # These models all use head_dim=64, which is supported by the
            # PyTorch Flash SDPA backend shipped in the H100 training image.
            _configure_flash_sdpa()
        output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=True
        )

        output = output.transpose(1, 2).contiguous().view(batch, tokens, -1)
        return F.linear(output, self.attn.out_proj.weight, self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor, need_intermediate: bool = False):
        attn_out = self._causal_attention(x)
        x = x + attn_out
        ffn_out = self.ffn(self.ln2(x))
        block_out = x + ffn_out
        if need_intermediate:
            return block_out, {"ffn_output": ffn_out, "block_output": block_out}
        return block_out


class GPT2Model(nn.Module):
    """GPT2 model supporting standard/hybrid/physical FFN and multiple sizes."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            GPT2Block(cfg.embed_dim, cfg.ffn_dim, cfg.num_heads,
                      cfg.model_type, cfg.voltage_max, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer("pos_ids", torch.arange(cfg.max_seq_len).unsqueeze(0), persistent=False)

    def forward(self, input_ids: torch.Tensor, need_intermediate: bool = False):
        B, T = input_ids.shape
        pos = self.pos_ids[:, :T].expand(B, -1)
        x = self.drop(self.token_embedding(input_ids) + self.pos_embedding(pos))

        intermediates = []
        for block in self.blocks:
            result = block(x, need_intermediate=need_intermediate)
            if need_intermediate:
                x, block_info = result
                intermediates.append(block_info)
            else:
                x = result

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if need_intermediate:
            return logits, intermediates
        return logits

    def configure_optimizer_groups(self, weight_decay: float, lr: float,
                                   ffn_only: bool = False):
        """Build full-model or FFN-only AdamW groups."""
        decay_params = []
        nodecay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ffn_only and ".ffn." not in name:
                continue
            if (name.startswith(("lm_head", "token_embedding"))
                    or "vth_" in name or "r_tia" in name):
                nodecay_params.append(param)
            elif param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
            {"params": nodecay_params, "weight_decay": 0.0, "lr": lr},
        ]


def create_model(cfg: ModelConfig) -> GPT2Model:
    return GPT2Model(cfg)
