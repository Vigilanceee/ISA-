"""Vision model configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict

from isa.device_models.config import make_flash_transistor_ekv_config


PHYSICAL_CONFIG = make_flash_transistor_ekv_config(default_lut_size=8192)


# Model scale presets: embed_dim, num_heads, mlp_ratio
# All scales use depth=12, patch_size=4
# hidden_dim = embed_dim * mlp_ratio
PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny":  {"embed_dim": 192, "num_heads": 3, "mlp_ratio": 4.0},  # hidden=768,  ~5.8M
    "mid":   {"embed_dim": 256, "num_heads": 4, "mlp_ratio": 4.0},  # hidden=1024, ~10M
    "small": {"embed_dim": 384, "num_heads": 6, "mlp_ratio": 4.0},  # hidden=1536, ~22M
    "base":  {"embed_dim": 768, "num_heads": 12, "mlp_ratio": 4.0},  # hidden=3072, ~86M
}


@dataclass
class ModelConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 100
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.05
    voltage_max: float = 4.0


def apply_preset(cfg: ModelConfig, scale: str) -> ModelConfig:
    """Apply a named model-scale preset to a ModelConfig in-place."""
    if scale not in PRESETS:
        raise ValueError(f"Unknown model scale: {scale}. Choices: {list(PRESETS.keys())}")
    p = PRESETS[scale]
    cfg.embed_dim = p["embed_dim"]
    cfg.num_heads = p["num_heads"]
    cfg.mlp_ratio = p["mlp_ratio"]
    return cfg


def get_model_config(scale: str = "tiny", **overrides) -> ModelConfig:
    """Create a ModelConfig from a scale preset with optional overrides."""
    cfg = ModelConfig()
    apply_preset(cfg, scale)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


@dataclass
class TrainConfig:
    data_dir: str = "./data"
    output_dir: str = "./outputs/physical_vit_cifar100"
    model: str = "physical_vit"  # physical_vit | standard_vit | hybrid_vit
    epochs: int = 200
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    print_freq: int = 50
    eval_freq: int = 1
    resume: str = ""
    device: str = "cuda"


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
