"""
VGG-8 网络  —— 使用 NVM 物理器件层 + TIA 构建

结构 (8 weight layers):
  Block-1: Conv128 → TIA+BN → Conv128 → TIA+BN → MaxPool
  Block-2: Conv256 → TIA+BN → Conv256 → TIA+BN → MaxPool
  Block-3: Conv512 → TIA+BN → Conv512 → TIA+BN → MaxPool
  FC: Linear1024 → TIA+BN1d → Linear10

网络中不使用任何标准激活函数 (ReLU 等),
所有非线性来源于器件 I-V 曲线与 TIA 电压钳位。

说明:
  v_min / v_max 仅定义外部输入编码范围。
  FeFET / Flash 的中间 TIA 输出钳位到 [-4, 4]V。
  ReRAM / PCM / STT 的中间 TIA 保持传入的器件工作区间。
  VGG8 的卷积/FC 层会在 bitline 上聚合更大的 fan-in 电流。
  因此对每一层的 TIA 电阻按前一层 fan-in 做 1/sqrt(K) 缩放，
  以减轻深层卷积/FC 层的饱和；MLP 保持原有全局 tia_r 不变。
"""

import math

import torch.nn as nn

from isa.kernels.device_sweep.custom_layers import NVM_Conv2d, NVM_Linear
from .tia_layer import TIA_Layer
from .quantization import quantize_input, quantize_output


class VGG8(nn.Module):
    def __init__(self, device_type: str, device_params: dict,
                 tia_r: float = 1.0, w_init_max: float = 0.1,
                 v_min: float = -1.0, v_max: float = 1.0):
        super().__init__()
        self.in_v_min = v_min
        self.in_v_max = v_max
        tia_v_min, tia_v_max = (
            (-4.0, 4.0)
            if device_type in {"fefet", "flash"}
            else (v_min, v_max)
        )
        def _scaled_tia_r(fan_in: int) -> float:
            return tia_r / math.sqrt(float(fan_in))

        def _conv(ci, co):
            return NVM_Conv2d(ci, co, kernel_size=3, padding=1,
                              device_type=device_type,
                              device_params=device_params,
                              w_init_max=w_init_max)

        def _tia2d(ch, fan_in):
            return TIA_Layer(
                _scaled_tia_r(fan_in),
                nn.BatchNorm2d(ch),
                v_min=tia_v_min,
                v_max=tia_v_max,
            )

        def _tia1d(ch, fan_in):
            return TIA_Layer(
                _scaled_tia_r(fan_in),
                nn.BatchNorm1d(ch),
                v_min=tia_v_min,
                v_max=tia_v_max,
            )

        self.features = nn.Sequential(
            _conv(3, 128),    _tia2d(128, 3 * 3 * 3),
            _conv(128, 128),  _tia2d(128, 128 * 3 * 3),
            nn.MaxPool2d(2, 2),

            _conv(128, 256),  _tia2d(256, 128 * 3 * 3),
            _conv(256, 256),  _tia2d(256, 256 * 3 * 3),
            nn.MaxPool2d(2, 2),

            _conv(256, 512),  _tia2d(512, 256 * 3 * 3),
            _conv(512, 512),  _tia2d(512, 512 * 3 * 3),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            NVM_Linear(512 * 4 * 4, 1024,
                       device_type=device_type,
                       device_params=device_params,
                       w_init_max=w_init_max),
            _tia1d(1024, 512 * 4 * 4),

            NVM_Linear(1024, 10,
                       device_type=device_type,
                       device_params=device_params,
                       w_init_max=w_init_max),
        )

    def forward(self, x):
        x = quantize_input(x, self.in_v_min, self.in_v_max)
        x = self.features(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return quantize_output(logits, -1.0, 1.0)
