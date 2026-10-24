"""
MLP 网络  —— 使用 NVM 物理器件层 + TIA 构建

结构 (2 weight layers):
  Flatten → Linear(784, 128) → TIA(×R, clamp[-4,4]) → Linear(128, 10)

所有非线性来源于器件 I-V 曲线与 TIA 电压钳位（无 BatchNorm）。
"""

import torch.nn as nn

from isa.kernels.device_sweep.custom_layers import NVM_Linear
from .peripherals import FixedGateCalibration, FixedOutputGain, gate_calibration
from .tia_layer import TIA_Layer
from .quantization import quantize_input, quantize_output


class MLP(nn.Module):
    def __init__(self, device_type: str, device_params: dict,
                 tia_r: float = 1e5, w_init_max: float = 0.05,
                 v_min: float = -1.0, v_max: float = 1.0):
        super().__init__()
        self.in_v_min = v_min
        self.in_v_max = v_max
        op_cal_enabled = (
            device_type == "fefet"
            and bool((device_params or {}).get("op_calibration_enabled", False))
        )
        calibration_slots = bool((device_params or {}).get("calibration_slots", True))
        if op_cal_enabled and not calibration_slots:
            raise ValueError("FeFET gate calibration requires calibration_slots=true")
        output_gain = float((device_params or {}).get("output_gain", 1.0))

        def _tia():
            return TIA_Layer(tia_r, nn.Identity(),
                             v_min=float((device_params or {}).get("tia_v_min", -4.0)),
                             v_max=float((device_params or {}).get("tia_v_max", 4.0)))

        layers = [
            NVM_Linear(784, 128, device_type, device_params, w_init_max),
            _tia(),
        ]
        if calibration_slots:
            layers.append(gate_calibration(device_type, device_params or {}))
        layers.append(NVM_Linear(128, 10, device_type, device_params, w_init_max))
        self.classifier = nn.Sequential(*layers)
        self.output_gain = FixedOutputGain(output_gain)

    def forward(self, x):
        x = quantize_input(x, self.in_v_min, self.in_v_max)
        x = x.view(x.size(0), -1)
        logits = self.output_gain(self.classifier(x))
        return quantize_output(logits, -1.0, 1.0)
