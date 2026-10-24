"""Fixed peripheral calibration used by the FeFET device experiments."""

import torch.nn as nn


class FixedGateCalibration(nn.Module):
    """Map a TIA output to the gate operating interval before the next array."""

    def __init__(self, center: float, scale: float, v_min: float, v_max: float):
        super().__init__()
        self.center = float(center)
        self.scale = float(scale)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

    def forward(self, x):
        return (self.center + self.scale * x).clamp(self.v_min, self.v_max)


class FixedOutputGain(nn.Module):
    """Apply the fixed readout gain after the final current aggregation."""

    def __init__(self, gain: float):
        super().__init__()
        self.gain = float(gain)

    def forward(self, x):
        return x * self.gain


def gate_calibration(device_type: str, params: dict) -> nn.Module:
    if device_type != "fefet" or not bool(params.get("op_calibration_enabled", False)):
        return nn.Identity()
    return FixedGateCalibration(
        params.get("op_calibration_center", 2.5),
        params.get("op_calibration_scale", 1.0),
        params.get("op_calibration_min", 0.0),
        params.get("op_calibration_max", 4.0),
    )
