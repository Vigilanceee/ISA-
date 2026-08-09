"""
跨阻放大器 (Transimpedance Amplifier, TIA) 层

单位约定:
  I_in  : μA  (器件物理输出)
  R_TIA : Ω   (真实跨阻值, 典型 10 kΩ ~ 10 MΩ)
  V_out : V   (电压)

信号流:
  I_in(μA) → I(A) = I_in × 1e-6
           → V = I(A) × R_TIA(Ω)
           → clamp(v_min, v_max)
           → BatchNorm
           → V_out

v_min / v_max 可配置:
  忆阻器 (ReRAM/PCM/STT):   使用对应器件工作区间
  晶体管 (FeFET/Flash):      中间层使用 [-4, 4]V
"""

import torch
import torch.nn as nn

_UA_TO_A = 1e-6


class TIA_Layer(nn.Module):
    """将电流(μA)转换为电压(V)，施加饱和钳位与 BatchNorm。"""

    def __init__(self, R_tia: float, bn: nn.Module,
                 v_min: float = -1.0, v_max: float = 1.0):
        super().__init__()
        self.R_tia = R_tia          # Ω
        self.bn = bn
        self.v_min = v_min
        self.v_max = v_max

    def forward(self, I_in: torch.Tensor) -> torch.Tensor:
        V = I_in * (_UA_TO_A * self.R_tia)
        V = torch.clamp(V, self.v_min, self.v_max)
        return self.bn(V)
