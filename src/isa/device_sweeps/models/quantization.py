"""
Uniform quantization helpers.
"""

import torch

_QUANT_ENABLED = True
_INPUT_BITS = 8
_OUTPUT_BITS = 8


def set_quantization_config(enabled: bool, input_bits: int = 8, output_bits: int = 8) -> None:
    """Set global quantization switches used by MLP/VGG8 forwards."""
    global _QUANT_ENABLED, _INPUT_BITS, _OUTPUT_BITS
    _QUANT_ENABLED = bool(enabled)
    _INPUT_BITS = int(input_bits)
    _OUTPUT_BITS = int(output_bits)


def quantize_input(x: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    if not _QUANT_ENABLED:
        return x
    return quantize_uniform(x, v_min, v_max, bits=_INPUT_BITS)


def quantize_output(x: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    if not _QUANT_ENABLED:
        return x
    return quantize_uniform(x, v_min, v_max, bits=_OUTPUT_BITS)


def quantize_uniform(x: torch.Tensor, v_min: float, v_max: float, bits: int = 8) -> torch.Tensor:
    """
    Clamp to [v_min, v_max] then quantize-dequantize with uniform levels.

    Uses a straight-through estimator so simulated ADC quantization does not
    block gradients during training.
    """
    if bits <= 1:
        return torch.clamp(x, v_min, v_max)
    levels = (1 << bits) - 1
    x_clamped = torch.clamp(x, v_min, v_max)
    scale = (v_max - v_min) / levels
    q = torch.round((x_clamped - v_min) / scale)
    x_quant = q * scale + v_min
    return x_clamped + (x_quant - x_clamped).detach()


def quantize_uniform_states(x: torch.Tensor, v_min: float, v_max: float, states: int) -> torch.Tensor:
    """
    Clamp to [v_min, v_max] then quantize-dequantize to a fixed number of states.

    This is the weight-QAT counterpart of quantize_uniform(). It uses the same
    straight-through estimator: backward sees the clamped real-valued weight,
    while forward sees the nearest discrete device state.
    """
    if states <= 1:
        return torch.clamp(x, v_min, v_max)
    x_clamped = torch.clamp(x, v_min, v_max)
    scale = (v_max - v_min) / (states - 1)
    q = torch.round((x_clamped - v_min) / scale).clamp(0, states - 1)
    x_quant = q * scale + v_min
    return x_clamped + (x_quant - x_clamped).detach()
