"""Opt-in Triton backend patch for DeploymentAwareCIMLinear.

This is kept outside the original research project until the standalone kernel
passes CCI correctness and speed gates.  ``install_fused_backend`` patches the
class before model construction, so existing checkpoints and parameter names
remain unchanged.
"""

from __future__ import annotations

import os

import torch

from isa.measured_deployment.kernels.measured_lut_triton import fused_measured_forward


def install_fused_backend() -> None:
    from isa.measured_deployment.operator import DeploymentAwareCIMLinear

    cls = DeploymentAwareCIMLinear
    if getattr(cls, "_fg50_fused_backend_installed", False):
        return
    original_set_assignment = cls.set_assignment
    original_measured_forward = cls.measured_forward
    build_segmented = os.environ.get("FG50_BUILD_SEGMENTED", "1") != "0"
    allow_tf32 = os.environ.get("FG50_ALLOW_TF32", "1") != "0"
    pair_token_threshold = int(
        os.environ.get("FG50_PAIR_FUSED_TOKEN_THRESHOLD", "64")
    )
    member_token_threshold = int(
        os.environ.get("FG50_MEMBER_FUSED_TOKEN_THRESHOLD", "16")
    )
    # All deployment-aware fc1 layers use the same immutable codebook.
    # Cache its CUDA copy and coefficients once per source tensor/device instead
    # of duplicating the 32,240-curve table across every transformer block.
    codebook_cache: dict[tuple, dict[str, object]] = {}

    def set_nonpersistent_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            setattr(self, name, value)
        else:
            self.register_buffer(name, value, persistent=False)

    @torch.no_grad()
    def configure_codebook(
        self,
        vg: torch.Tensor,
        currents_a: torch.Tensor,
        vin_lut_bins: int = 0,
    ) -> None:
        if int(vin_lut_bins) != 0:
            raise ValueError(
                "FG50 fused backend implements exact measured segments; "
                "vin_lut_bins must be zero"
            )
        device = self.vth_pos.device
        cache_key = (
            device.type,
            device.index,
            id(vg),
            id(currents_a),
            tuple(vg.shape),
            tuple(currents_a.shape),
        )
        cached = codebook_cache.get(cache_key)
        if cached is None:
            measured_vg = (
                vg.detach().to(device=device, dtype=torch.float32).contiguous()
            )
            measured_currents = (
                currents_a.detach()
                .to(device=device, dtype=torch.float32)
                .contiguous()
            )
            if (
                measured_vg.ndim != 1
                or measured_currents.ndim != 2
                or measured_currents.shape[1] != measured_vg.numel()
            ):
                raise ValueError("Measured codebook tensors have inconsistent shapes")
            if measured_vg.numel() < 2 or not bool(
                torch.all(measured_vg[1:] > measured_vg[:-1])
            ):
                raise ValueError("Measured Vg grid must be strictly increasing")
            grid_step = measured_vg[1:] - measured_vg[:-1]
            if not bool(
                torch.allclose(
                    grid_step,
                    grid_step[0],
                    atol=1e-6,
                    rtol=1e-5,
                )
            ):
                raise ValueError(
                    "FG50 fused backend requires a uniform measured voltage grid"
                )
            curve_slopes = (
                (measured_currents[:, 1:] - measured_currents[:, :-1])
                / grid_step.unsqueeze(0)
            ).contiguous()
            curve_intercepts = (
                measured_currents[:, :-1]
                - curve_slopes * measured_vg[:-1].unsqueeze(0)
            ).contiguous()
            curve_count = int(measured_currents.shape[0])
            pair_slopes = None
            pair_intercepts = None
            if curve_count <= 64:
                pair_slopes = (
                    curve_slopes[:, None, :] - curve_slopes[None, :, :]
                ).reshape(curve_count * curve_count, -1).contiguous()
                pair_intercepts = (
                    curve_intercepts[:, None, :]
                    - curve_intercepts[None, :, :]
                ).reshape(curve_count * curve_count, -1).contiguous()
            cached = {
                # Hold the source tensors too, so Python object ids cannot be
                # recycled while this cache entry remains live.
                "source_vg": vg,
                "source_currents": currents_a,
                "vg": measured_vg,
                "currents": measured_currents,
                "curve_slopes": curve_slopes,
                "curve_intercepts": curve_intercepts,
                "pair_slopes": pair_slopes,
                "pair_intercepts": pair_intercepts,
                "voltage_min": float(measured_vg[0].item()),
                "inverse_voltage_step": 1.0 / float(grid_step[0].item()),
            }
            codebook_cache[cache_key] = cached

        self.measured_vg = cached["vg"]
        self.measured_currents = cached["currents"]
        self.lookup_min = max(
            float(cached["voltage_min"]),
            float(self.cfg.get("V_min", 0.0)),
        )
        voltage_max = float(cached["voltage_min"]) + (
            (self.measured_vg.numel() - 1)
            / float(cached["inverse_voltage_step"])
        )
        self.lookup_max = min(
            voltage_max,
            float(self.cfg.get("V_max", 4.0)),
        )
        self.vin_lut_bins = 0
        set_nonpersistent_buffer(
            self, "_fg50_curve_slopes", cached["curve_slopes"]
        )
        set_nonpersistent_buffer(
            self, "_fg50_curve_intercepts", cached["curve_intercepts"]
        )
        # Cache the Python scalars once. Re-reading CUDA tensors with .item()
        # inside every layer forward would force device synchronization.
        self._fg50_voltage_min = float(cached["voltage_min"])
        self._fg50_inverse_voltage_step = float(
            cached["inverse_voltage_step"]
        )
        self._fg50_pair_mode = False
        set_nonpersistent_buffer(
            self,
            "_fg50_pair_ids",
            torch.empty(0, dtype=torch.int16, device=self.measured_vg.device),
        )
        set_nonpersistent_buffer(
            self,
            "_fg50_pair_slopes",
            torch.empty(0, dtype=torch.float32, device=self.measured_vg.device),
        )
        set_nonpersistent_buffer(
            self,
            "_fg50_pair_intercepts",
            torch.empty(0, dtype=torch.float32, device=self.measured_vg.device),
        )
        self._fg50_cached_pair_slopes = cached["pair_slopes"]
        self._fg50_cached_pair_intercepts = cached["pair_intercepts"]

    @torch.no_grad()
    def set_assignment(
        self, pos_idx: torch.Tensor, neg_idx: torch.Tensor
    ) -> None:
        if self.measured_vg.numel() == 0:
            raise RuntimeError("configure_codebook must be called before set_assignment")
        # Build the exact expanded segmented weights as the high-throughput
        # backend for realistic token counts. This costs memory, but its TF32
        # GEMMs are substantially faster than irregular gather/reduction once
        # the flattened token count exceeds the measured crossover.
        if build_segmented:
            original_set_assignment(self, pos_idx, neg_idx)
            pos = self.pos_idx
            neg = self.neg_idx
        else:
            expected = (self.out_features, self.in_features)
            if tuple(pos_idx.shape) != expected or tuple(neg_idx.shape) != expected:
                raise ValueError(f"Assignment shape must be {expected}")
            device = self.vth_pos.device
            pos = pos_idx.to(device=device, dtype=torch.int16).contiguous()
            neg = neg_idx.to(device=device, dtype=torch.int16).contiguous()
            if int(pos.min()) < 0 or int(neg.min()) < 0:
                raise ValueError("Assignment contains negative state indices")
            self.pos_idx = pos
            self.neg_idx = neg
        device = self.vth_pos.device
        curve_count = int(self.measured_currents.shape[0])
        if int(pos.max()) >= curve_count or int(neg.max()) >= curve_count:
            raise ValueError("Assignment references a curve outside the codebook")
        self.pos_idx = pos
        self.neg_idx = neg

        # The 24-state deterministic path has only 576 differential pairs.
        # For the 32,240-curve Monte Carlo path, materializing all possible
        # pairs would be impossible; the kernel gathers pos/neg coefficients.
        if curve_count <= 64:
            self._fg50_pair_slopes = self._fg50_cached_pair_slopes
            self._fg50_pair_intercepts = self._fg50_cached_pair_intercepts
            self._fg50_pair_ids = (
                pos.to(torch.int32) * curve_count + neg.to(torch.int32)
            ).to(torch.int16).contiguous()
            self._fg50_pair_mode = True
        else:
            self._fg50_pair_mode = False
            self._fg50_pair_ids = torch.empty(
                0, dtype=torch.int16, device=device
            )
            self._fg50_pair_slopes = torch.empty(
                0, dtype=torch.float32, device=device
            )
            self._fg50_pair_intercepts = torch.empty_like(
                self._fg50_pair_slopes
            )

        if not build_segmented:
            self.active_segments = torch.empty(
                0, dtype=torch.long, device=device
            )
            self.diff_a = torch.empty(0, device=device)
            self.diff_b = torch.empty(0, device=device)
            self.bin_centers = torch.empty(0, device=device)
            self.diff_current = torch.empty(0, device=device)

    def measured_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_idx.numel() == 0:
            raise RuntimeError("Measured assignment is not initialized")
        if torch.is_grad_enabled() and x.requires_grad:
            if not build_segmented:
                raise RuntimeError(
                    "FG50 compact-only measured-LUT has no direct measured "
                    "backward; enable FG50_BUILD_SEGMENTED=1"
                )
            return original_measured_forward(self, x)
        flat_tokens = x.numel() // self.in_features
        threshold = (
            pair_token_threshold
            if self._fg50_pair_mode
            else member_token_threshold
        )
        if build_segmented and flat_tokens > threshold:
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            try:
                return original_measured_forward(self, x)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        original_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            if self._fg50_pair_mode:
                output = fused_measured_forward(
                    x.float(),
                    self._fg50_pair_ids,
                    self._fg50_pair_slopes,
                    self._fg50_pair_intercepts,
                    self.measured_vg,
                    pair_mode=True,
                    lookup_min=self.lookup_min,
                    lookup_max=self.lookup_max,
                    current_scale=1.0,
                    r_tia=self.r_tia,
                    signed_min=float(self.cfg.get("V_signed_min", -4.0)),
                    signed_max=float(self.cfg.get("V_signed_max", 4.0)),
                    voltage_min=self._fg50_voltage_min,
                    inverse_voltage_step=self._fg50_inverse_voltage_step,
                )
            else:
                output = fused_measured_forward(
                    x.float(),
                    self.pos_idx,
                    self._fg50_curve_slopes,
                    self._fg50_curve_intercepts,
                    self.measured_vg,
                    assignment_b=self.neg_idx,
                    pair_mode=False,
                    lookup_min=self.lookup_min,
                    lookup_max=self.lookup_max,
                    current_scale=1.0,
                    r_tia=self.r_tia,
                    signed_min=float(self.cfg.get("V_signed_min", -4.0)),
                    signed_max=float(self.cfg.get("V_signed_max", 4.0)),
                    voltage_min=self._fg50_voltage_min,
                    inverse_voltage_step=self._fg50_inverse_voltage_step,
                )
        return output.to(original_dtype)

    cls.configure_codebook = configure_codebook
    cls.set_assignment = set_assignment
    cls.measured_forward = measured_forward
    cls._fg50_fused_backend_installed = True
