"""Physical ViT, Hybrid ViT, and Standard ViT baselines for CIFAR-100."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from isa.vision.config import ModelConfig, PHYSICAL_CONFIG
from isa.operators.cim import PhysicalFFN, HybridFFN


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep).div(keep)
    return x * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """Standard FFN: Linear -> GELU -> Linear."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=False,
        )
        x = x.transpose(1, 2).contiguous().reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PhysicalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        physical_config: Optional[dict] = None,
        voltage_max: float = 4.0,
        use_physical_ffn: bool = True,
        use_hybrid_ffn: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        if use_hybrid_ffn:
            self.mlp = HybridFFN(
                dim,
                hidden,
                physical_config=physical_config,
                voltage_max=voltage_max,
            )
        elif use_physical_ffn:
            self.mlp = PhysicalFFN(
                dim,
                hidden,
                physical_config=physical_config,
                voltage_max=voltage_max,
            )
        else:
            self.mlp = Mlp(dim, hidden, drop=drop)
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ViTBase(nn.Module):
    """Shared ViT backbone."""

    def __init__(self, cfg: ModelConfig, use_physical_ffn: bool = True, use_hybrid_ffn: bool = False):
        super().__init__()
        self.cfg = cfg
        img_size = cfg.img_size
        patch_size = cfg.patch_size
        embed_dim = cfg.embed_dim
        depth = cfg.depth

        self.patch_embed = nn.Conv2d(
            cfg.in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(cfg.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                PhysicalTransformerBlock(
                    dim=embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    drop=cfg.drop_rate,
                    attn_drop=cfg.attn_drop_rate,
                    drop_path_rate=dpr[i],
                    physical_config=PHYSICAL_CONFIG,
                    voltage_max=cfg.voltage_max,
                    use_physical_ffn=use_physical_ffn,
                    use_hybrid_ffn=use_hybrid_ffn,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)  # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)


class PhysicalViT(ViTBase):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        cfg = cfg or ModelConfig()
        super().__init__(cfg, use_physical_ffn=True, use_hybrid_ffn=False)


class HybridViT(ViTBase):
    """ViT with CIM first layer + Linear second layer in each FFN block."""

    def __init__(self, cfg: Optional[ModelConfig] = None):
        cfg = cfg or ModelConfig()
        super().__init__(cfg, use_physical_ffn=False, use_hybrid_ffn=True)


class StandardViT(ViTBase):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        cfg = cfg or ModelConfig()
        super().__init__(cfg, use_physical_ffn=False, use_hybrid_ffn=False)


def build_model(name: str, cfg: Optional[ModelConfig] = None) -> nn.Module:
    cfg = cfg or ModelConfig()
    if name == "physical_vit":
        return PhysicalViT(cfg)
    if name == "hybrid_vit":
        return HybridViT(cfg)
    if name == "standard_vit":
        return StandardViT(cfg)
    raise ValueError(f"Unknown model: {name}")
