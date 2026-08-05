"""
NVM_Conv2d / NVM_Linear  —— 基于差分对 NVM 器件的自定义层

Conv2d 通过 F.unfold (im2col) 将卷积转化为矩阵乘法,
然后由对应器件的 autograd.Function 完成物理电流计算。

器件统一参数化:
  theta_pos / theta_neg 为独立可训练参数，
  I_net = f(vin, theta_pos) - f(vin, theta_neg)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from isa.approximations.lowrank_planar import (
    lowrank_planar_conv2d,
    lowrank_planar_linear,
)
from isa.approximations.node_planar import (
    factorized_conv2d,
    factorized_linear,
    node_planar_conv2d,
    node_planar_linear,
    state_nodes,
)
from isa.device_sweeps.backend import select_primitive_backend
from isa.device_sweeps.models.quantization import quantize_uniform_states

from .direct_conv_triton import DirectTransistorConv2d
from .fefet_triton import FeFETFunction
from .flash_triton import FlashFunction
from .lut_triton import FeFETLUTFunction, FlashLUTFunction, PCMLUTFunction, PCMPlanarFunction
from .pcm_triton import PCMFunction
from .reram_triton import ReRAMFunction
from .stt_triton import STTFunction

_FUNC_MAP = {
    'reram': ReRAMFunction,
    'pcm':   PCMFunction,
    'stt':   STTFunction,
    'fefet': FeFETFunction,
    'flash': FlashFunction,
}

_LUT_FUNC_MAP = {
    'pcm':   PCMLUTFunction,
    'fefet': FeFETLUTFunction,
    'flash': FlashLUTFunction,
}

_PLANAR_FUNC_MAP = {
    'pcm': PCMPlanarFunction,
}

def _maybe_quantize_weight(theta: torch.Tensor, device_params: dict):
    if not bool((device_params or {}).get("weight_quant_enabled", False)):
        return theta
    states = int(device_params.get("weight_quant_states", 8))
    lo = float(device_params.get("weight_quant_min", device_params.get("w_min", 0.0)))
    hi = float(device_params.get("weight_quant_max", device_params.get("w_max", 1.0)))
    if lo >= hi:
        raise ValueError(f"Invalid weight quantization range: [{lo}, {hi}]")
    return quantize_uniform_states(theta, lo, hi, states)


def _select_nvm_func(
    device_type: str,
    device_params: dict,
    backend: str | None = None,
):
    primitive_backend = select_primitive_backend(device_params, backend)
    if primitive_backend == "reference":
        return _FUNC_MAP[device_type]
    if primitive_backend == "lut":
        return _LUT_FUNC_MAP.get(device_type, _FUNC_MAP[device_type])
    if primitive_backend == "planar":
        return _PLANAR_FUNC_MAP.get(device_type, _FUNC_MAP[device_type])
    raise AssertionError(f"Unhandled primitive backend: {primitive_backend}")

def _default_init_center(device_type: str, device_params: dict,
                         w_min: float, w_max: float) -> float:
    if "init_center" in device_params:
        return float(device_params["init_center"])
    if device_type == "reram":
        # ReRAM conductance decays exponentially with gap; mid-range starts too
        # close to HRS and makes the two branches nearly cancel.
        return w_min + 0.2 * (w_max - w_min)
    if device_type == "stt":
        return 0.5 * (w_min + w_max)
    if device_type in {"fefet", "flash"}:
        return 0.5 * (w_min + w_max)  # mid-range Vth for transistor devices
    return 0.5 * (w_min + w_max)


def _init_weights(out_features: int, in_features: int,
                  device_type: str, device_params: dict,
                  w_min: float, w_max: float, w_init_max: float):
    """
    Initialise theta_pos / theta_neg independently.

    Both start near init_center with small independent jitter.
    For transistor devices, theta directly represents Vth (not offset from vcom).
    """
    center = _default_init_center(device_type, device_params, w_min, w_max)
    init_std = device_params.get("init_std")
    if init_std is not None:
        std = max(float(init_std), 1e-8)
        theta_pos = torch.empty(out_features, in_features).normal_(center, std)
        theta_neg = torch.empty(out_features, in_features).normal_(center, std)
    else:
        half_width = max(1e-6, float(w_init_max))
        theta_pos = torch.empty(out_features, in_features).uniform_(
            center - half_width, center + half_width
        )
        theta_neg = torch.empty(out_features, in_features).uniform_(
            center - half_width, center + half_width
        )
    theta_pos.clamp_(w_min, w_max)
    theta_neg.clamp_(w_min, w_max)
    return theta_pos, theta_neg


class NVM_Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device_type: str, device_params: dict,
                 w_init_max: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_params = device_params or {}
        self.device_type = device_type
        self.backend = str(
            self.device_params.get("linear_backend", self.device_params.get("conv_backend", "reference"))
        )
        self.nvm_func = _select_nvm_func(
            device_type, self.device_params, self.backend
        )
        self.w_min = float(self.device_params.get('w_min', 1e-6))
        self.w_max = float(self.device_params.get('w_max', 5.0))

        theta_pos, theta_neg = _init_weights(out_features, in_features,
                                             device_type, self.device_params,
                                             self.w_min, self.w_max, w_init_max)
        self.theta_pos = nn.Parameter(theta_pos)
        self.theta_neg = nn.Parameter(theta_neg)
        nodes = (
            state_nodes(self.device_params)
            if self.backend in {"node_planar", "lowrank_planar"}
            else None
        )
        self.register_buffer("planar_state_nodes", nodes, persistent=False)

    def forward(self, v_in: torch.Tensor) -> torch.Tensor:
        theta_pos = _maybe_quantize_weight(self.theta_pos, self.device_params)
        theta_neg = _maybe_quantize_weight(self.theta_neg, self.device_params)
        if self.backend == "factorized":
            return factorized_linear(
                v_in, theta_pos, theta_neg,
                self.device_params, self.device_type,
            )
        if self.backend == "node_planar":
            return node_planar_linear(
                v_in, theta_pos, theta_neg,
                self.planar_state_nodes,
                self.device_params, self.device_type,
            )
        if self.backend == "lowrank_planar":
            return lowrank_planar_linear(
                v_in, theta_pos, theta_neg,
                self.planar_state_nodes,
                self.device_params, self.device_type,
            )
        out = self.nvm_func.apply(v_in, theta_pos, theta_neg,
                                  self.device_params)
        return out


class NVM_Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 device_type: str = 'reram', device_params: dict = None,
                 w_init_max: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.device_params = device_params or {}
        self.device_type = device_type
        self.backend = str(self.device_params.get("conv_backend", "reference"))
        self.nvm_func = _select_nvm_func(
            device_type, self.device_params, self.backend
        )
        self.w_min = float(self.device_params.get('w_min', 1e-6))
        self.w_max = float(self.device_params.get('w_max', 5.0))

        K = in_channels * self.kernel_size[0] * self.kernel_size[1]
        theta_pos, theta_neg = _init_weights(out_channels, K, device_type, self.device_params,
                                             self.w_min, self.w_max, w_init_max)
        self.theta_pos = nn.Parameter(theta_pos)
        self.theta_neg = nn.Parameter(theta_neg)
        nodes = (
            state_nodes(self.device_params)
            if self.backend in {"node_planar", "lowrank_planar"}
            else None
        )
        self.register_buffer("planar_state_nodes", nodes, persistent=False)

    def forward(self, v_in: torch.Tensor) -> torch.Tensor:
        B, C, H, W = v_in.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        theta_pos = _maybe_quantize_weight(self.theta_pos, self.device_params)
        theta_neg = _maybe_quantize_weight(self.theta_neg, self.device_params)

        if self.backend == "factorized":
            return factorized_conv2d(
                v_in, theta_pos, theta_neg,
                self.device_params, self.device_type,
                self.kernel_size, self.stride, self.padding,
            )

        if self.backend == "node_planar":
            return node_planar_conv2d(
                v_in, theta_pos, theta_neg,
                self.planar_state_nodes,
                self.device_params, self.device_type,
                self.kernel_size, self.stride, self.padding,
            )

        if self.backend == "lowrank_planar":
            return lowrank_planar_conv2d(
                v_in, theta_pos, theta_neg,
                self.planar_state_nodes,
                self.device_params, self.device_type,
                self.kernel_size, self.stride, self.padding,
            )

        if (
            bool((self.device_params or {}).get("direct_conv_enabled", False))
            and self.device_type in {"flash", "fefet"}
            and self.backend != "reference"
        ):
            return DirectTransistorConv2d.apply(
                v_in, theta_pos, theta_neg, self.device_params,
                self.device_type, self.kernel_size, self.stride, self.padding,
                self.nvm_func,
            )

        L = H_out * W_out
        v_col = F.unfold(v_in, self.kernel_size,
                         stride=self.stride, padding=self.padding)
        v_flat = v_col.permute(0, 2, 1).contiguous().view(B * L, -1)

        out_flat = self.nvm_func.apply(v_flat, theta_pos, theta_neg,
                                       self.device_params)

        return (out_flat.view(B, L, self.out_channels)
                        .permute(0, 2, 1)
                        .contiguous()
                        .view(B, self.out_channels, H_out, W_out))
