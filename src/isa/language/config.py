"""Unified configuration for GPT2 CIM match-step experiments.

Three model sizes (tiny, mid, small) × three model types (standard, hybrid, physical)
= 9 experiments. Training uses a three-component matching loss followed by LM training.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

from isa.device_models.config import make_flash_transistor_ekv_config

# ── Physical device constants (shared with ViT) ──────────────────────────
PHYSICAL_CONFIG = make_flash_transistor_ekv_config(default_lut_size=4096)

# ── Model size presets ────────────────────────────────────────────────────
SIZE_PRESETS: Dict[str, Dict] = {
    "tiny": {
        "embed_dim": 192,
        "ffn_dim": 768,      # 192 × 4
        "num_heads": 3,
        "depth": 12,
        "dropout": 0.0,
    },
    "mid": {
        "embed_dim": 384,
        "ffn_dim": 1536,     # 384 × 4
        "num_heads": 6,
        "depth": 12,
        "dropout": 0.0,
    },
    "small": {
        "embed_dim": 768,
        "ffn_dim": 3072,     # 768 × 4
        "num_heads": 12,
        "depth": 12,
        "dropout": 0.0,
    },
}

ModelType = Literal["standard", "hybrid", "physical"]
ModelSize = Literal["tiny", "mid", "small"]


@dataclass
class ModelConfig:
    """GPT2 model architecture hyperparameters."""
    size: ModelSize = "tiny"
    model_type: ModelType = "standard"
    vocab_size: int = 50257
    max_seq_len: int = 128
    embed_dim: int = 192
    ffn_dim: int = 768
    num_heads: int = 3
    depth: int = 12
    dropout: float = 0.0
    # Physical FFN
    voltage_max: float = 4.0
    affine_scope: str = "final"       # "final" | "both" | "none"
    output_chunk_size: int = 16

    @classmethod
    def from_preset(cls, size: ModelSize, model_type: ModelType, **kwargs):
        preset = SIZE_PRESETS[size]
        return cls(size=size, model_type=model_type, **preset, **kwargs)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Paths
    tokenizer_name: str = "gpt2"
    train_text: str = ""
    val_text: str = ""
    baseline_checkpoint: str = ""    # required for hybrid/physical
    output_dir: str = "./outputs"
    # Data
    seq_len: int = 128
    batch_size: int = 8
    val_batch_size: int = 16
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.0       # ViT-style: no wd on CIM params
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    # Scheduler
    warmup_steps: int = 500
    max_steps: int = 15000
    # Loss weights (Phase 1: matching)
    lambda_ffn: float = 1.0         # λ1: FFN output MSE
    lambda_block: float = 1.0       # λ2: Block output MSE
    lambda_logit: float = 1.0       # λ3: Logit KL divergence
    kl_chunk_tokens: int = 4096      # memory-bounded exact KL implementation
    match_steps: int = 200           # Phase 1 matching steps
    # Phase 2: LM training
    lm_steps: int = 15000
    # Gradient
    grad_clip: float = 1.0
    # Precision
    amp: bool = True
    amp_dtype: str = "bf16"
    # Logging
    print_freq: int = 20
    eval_freq: int = 1000
    save_freq: int = 2500
    # Device
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    # Misc
    max_train_chars: int = 0
    tokenize_chunk_chars: int = 20000
    local_files_only: bool = True


@dataclass
class InitConfig:
    """ViT-style initialization for physical/hybrid models."""
    init_max_batches: int = 2
    init_max_rows: int = 4096
    voltage_map_quantile: float = 0.995
    voltage_map_low: float = 0.2
    voltage_map_high: float = 3.8
    voltage_map_iters: int = 2
    ekv_eps_k: float = 0.5
    ekv_rho: float = 0.8
    ekv_quantile: float = 0.99
    match_lr: float = 1e-3
    match_batch_size: int = 256
