"""Fixed measured-codebook forward with an EKV straight-through gradient.

Only Physical ViT fc1 layers are replaced.  Parameter names remain identical
to :class:`CIMLinear`, so original and fine-tuned checkpoints stay compatible.
The measured LUT and assignment tensors are non-persistent buffers: a resumed
run rebuilds them from the fixed codebook and the checkpoint's current Vth.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from isa.operators.cim import CIMLinear


class DeploymentAwareCIMLinear(CIMLinear):
    """CIMLinear whose forward can be continuous, measured, or measured+STE."""

    MODES: ClassVar[set[str]] = {"continuous", "measured", "ste"}

    def __init__(self, in_features: int, out_features: int, physical_config=None) -> None:
        super().__init__(in_features, out_features, physical_config)
        self.forward_mode = "continuous"
        self.vin_lut_bins = 0
        self.lookup_min = float(self.cfg.get("V_min", 0.0))
        self.lookup_max = float(self.cfg.get("V_max", 4.0))
        self.register_buffer("measured_vg", torch.empty(0), persistent=False)
        self.register_buffer("measured_currents", torch.empty(0), persistent=False)
        self.register_buffer("pos_idx", torch.empty(0, dtype=torch.int16), persistent=False)
        self.register_buffer("neg_idx", torch.empty(0, dtype=torch.int16), persistent=False)
        self.register_buffer("active_segments", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("diff_a", torch.empty(0), persistent=False)
        self.register_buffer("diff_b", torch.empty(0), persistent=False)
        self.register_buffer("bin_centers", torch.empty(0), persistent=False)
        self.register_buffer("diff_current", torch.empty(0), persistent=False)

    @classmethod
    def from_cimlinear(cls, original: CIMLinear) -> DeploymentAwareCIMLinear:
        replacement = cls(original.in_features, original.out_features, original.cfg)
        replacement.load_state_dict(original.state_dict(), strict=True)
        replacement.to(device=original.vth_pos.device, dtype=original.vth_pos.dtype)
        return replacement

    def set_forward_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported deployment-aware forward mode: {mode}")
        self.forward_mode = mode

    @torch.no_grad()
    def configure_codebook(
        self,
        vg: torch.Tensor,
        currents_a: torch.Tensor,
        vin_lut_bins: int = 0,
    ) -> None:
        device = self.vth_pos.device
        vg = vg.detach().to(device=device, dtype=torch.float32).contiguous()
        currents = currents_a.detach().to(device=device, dtype=torch.float32).contiguous()
        if vg.ndim != 1 or currents.ndim != 2 or currents.shape[1] != vg.numel():
            raise ValueError("Measured codebook tensors have inconsistent shapes")
        if vg.numel() < 2 or not bool(torch.all(vg[1:] > vg[:-1])):
            raise ValueError("Measured Vg grid must be strictly increasing")
        self.measured_vg = vg
        self.measured_currents = currents
        self.lookup_min = max(float(vg[0].item()), float(self.cfg.get("V_min", 0.0)))
        self.lookup_max = min(float(vg[-1].item()), float(self.cfg.get("V_max", 4.0)))
        self.vin_lut_bins = int(vin_lut_bins)
        if self.vin_lut_bins < 0:
            raise ValueError("vin_lut_bins must be non-negative")

    @torch.no_grad()
    def set_assignment(self, pos_idx: torch.Tensor, neg_idx: torch.Tensor) -> None:
        if self.measured_vg.numel() == 0:
            raise RuntimeError("configure_codebook must be called before set_assignment")
        expected = (self.out_features, self.in_features)
        if tuple(pos_idx.shape) != expected or tuple(neg_idx.shape) != expected:
            raise ValueError(f"Assignment shape must be {expected}")
        device = self.vth_pos.device
        self.pos_idx = pos_idx.to(device=device, dtype=torch.int16).contiguous()
        self.neg_idx = neg_idx.to(device=device, dtype=torch.int16).contiguous()
        pos = self.pos_idx.long()
        neg = self.neg_idx.long()
        if int(pos.min()) < 0 or int(neg.min()) < 0:
            raise ValueError("Assignment contains negative state indices")
        state_count = int(self.measured_currents.shape[0])
        if int(pos.max()) >= state_count or int(neg.max()) >= state_count:
            raise ValueError("Assignment references a state outside the codebook")

        if self.vin_lut_bins > 0:
            centers = torch.linspace(
                self.lookup_min,
                self.lookup_max,
                self.vin_lut_bins,
                device=device,
                dtype=torch.float32,
            )
            diff = torch.empty(
                self.vin_lut_bins,
                self.out_features,
                self.in_features,
                device=device,
                dtype=torch.float32,
            )
            for bin_id, center in enumerate(centers):
                measured = self.currents_at(center)
                diff[bin_id] = measured[pos] - measured[neg]
            self.bin_centers = centers
            self.diff_current = diff
            self.active_segments = torch.empty(0, dtype=torch.long, device=device)
            self.diff_a = torch.empty(0, device=device)
            self.diff_b = torch.empty(0, device=device)
            return

        vg = self.measured_vg
        currents = self.measured_currents
        dv = vg[1:] - vg[:-1]
        slopes = (currents[:, 1:] - currents[:, :-1]) / dv.unsqueeze(0)
        intercepts = currents[:, :-1] - slopes * vg[:-1].unsqueeze(0)
        active = torch.nonzero(
            (vg[:-1] <= self.lookup_max) & (vg[1:] >= self.lookup_min),
            as_tuple=False,
        ).flatten()
        diff_a = torch.empty(
            active.numel(),
            self.out_features,
            self.in_features,
            device=device,
            dtype=torch.float32,
        )
        diff_b = torch.empty_like(diff_a)
        for local_id, segment_id in enumerate(active.tolist()):
            diff_a[local_id] = slopes[:, segment_id][pos] - slopes[:, segment_id][neg]
            diff_b[local_id] = (
                intercepts[:, segment_id][pos] - intercepts[:, segment_id][neg]
            )
        self.active_segments = active
        self.diff_a = diff_a
        self.diff_b = diff_b
        self.bin_centers = torch.empty(0, device=device)
        self.diff_current = torch.empty(0, device=device)

    def currents_at(self, voltage: torch.Tensor | float) -> torch.Tensor:
        if self.measured_vg.numel() == 0:
            raise RuntimeError("Measured codebook has not been configured")
        value = torch.as_tensor(
            voltage, device=self.measured_vg.device, dtype=torch.float32
        ).clamp(self.measured_vg[0], self.measured_vg[-1])
        upper = torch.searchsorted(self.measured_vg, value, right=True).clamp(
            1, self.measured_vg.numel() - 1
        )
        lower = upper - 1
        alpha = (value - self.measured_vg[lower]) / (
            self.measured_vg[upper] - self.measured_vg[lower]
        )
        return self.measured_currents[:, lower] + alpha * (
            self.measured_currents[:, upper] - self.measured_currents[:, lower]
        )

    def measured_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_idx.numel() == 0:
            raise RuntimeError("Measured assignment is not initialized")
        original_dtype = x.dtype
        shape = x.shape
        # The measured path stays float32 even when the EKV surrogate uses AMP.
        with torch.amp.autocast("cuda", enabled=False):
            flat = x.float().reshape(-1, self.in_features).clamp(
                self.lookup_min, self.lookup_max
            )
            output = torch.zeros(
                flat.shape[0],
                self.out_features,
                dtype=torch.float32,
                device=flat.device,
            )
            if self.vin_lut_bins > 0:
                if self.vin_lut_bins == 1:
                    bin_id = torch.zeros_like(flat, dtype=torch.long)
                else:
                    scale = (self.vin_lut_bins - 1) / (
                        self.lookup_max - self.lookup_min
                    )
                    bin_id = torch.round((flat - self.lookup_min) * scale).long().clamp(
                        0, self.vin_lut_bins - 1
                    )
                for current_bin in range(self.vin_lut_bins):
                    mask = (bin_id == current_bin).to(flat.dtype)
                    output.add_(F.linear(mask, self.diff_current[current_bin]))
            else:
                segment_id = torch.searchsorted(
                    self.measured_vg, flat, right=True
                ).sub_(1).clamp_(0, self.measured_vg.numel() - 2)
                for local_id in range(self.active_segments.numel()):
                    mask = (segment_id == self.active_segments[local_id]).to(flat.dtype)
                    output.add_(F.linear(flat * mask, self.diff_a[local_id]))
                    output.add_(F.linear(mask, self.diff_b[local_id]))
            r_tia = self.r_tia.detach().float()
            output.mul_(r_tia).clamp_(
                float(self.cfg.get("V_signed_min", -4.0)),
                float(self.cfg.get("V_signed_max", 4.0)),
            )
        return output.reshape(*shape[:-1], self.out_features).to(original_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.forward_mode == "continuous":
            return super().forward(x)
        if self.forward_mode == "measured":
            # State ids and measured slopes/intercepts are fixed buffers, so this
            # path cannot update the deployed fc1 weights.  Keeping the exact
            # piecewise-linear derivative with respect to Vin is nevertheless
            # required when LayerNorm/attention before fc1 are trainable during
            # measured-forward digital compensation.
            return self.measured_forward(x)
        # STE keeps the measured value path detached and uses only the
        # continuous EKV surrogate during backward.
        with torch.no_grad():
            measured = self.measured_forward(x)
        continuous = super().forward(x)
        # Forward equals measured; backward equals the continuous EKV surrogate.
        return continuous + (measured - continuous).detach()


def replace_physical_fc1(model: nn.Module) -> list[DeploymentAwareCIMLinear]:
    layers: list[DeploymentAwareCIMLinear] = []
    for block_id, block in enumerate(model.blocks):
        original = block.mlp.fc1
        if not isinstance(original, CIMLinear):
            raise TypeError(f"blocks.{block_id}.mlp.fc1 is not CIMLinear")
        replacement = DeploymentAwareCIMLinear.from_cimlinear(original)
        block.mlp.fc1 = replacement
        layers.append(replacement)
    return layers


def iter_deployment_fc1(model: nn.Module) -> Iterable[tuple[int, DeploymentAwareCIMLinear]]:
    for block_id, block in enumerate(model.blocks):
        layer = block.mlp.fc1
        if not isinstance(layer, DeploymentAwareCIMLinear):
            raise TypeError(f"blocks.{block_id}.mlp.fc1 is not deployment-aware")
        yield block_id, layer


def set_fc1_forward_mode(model: nn.Module, mode: str) -> None:
    for _, layer in iter_deployment_fc1(model):
        layer.set_forward_mode(mode)
