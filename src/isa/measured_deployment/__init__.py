"""Measured-codebook verify/write deployment for physical ViT FFNs.

This package is intentionally separate from :mod:`isa.device_sweeps`: device
sweeps approximate analytic device equations, while measured deployment snaps
a trained Physical ViT to a finite empirical I-V codebook and optionally
performs fixed-assignment compensation training.
"""

from isa.measured_deployment.fg50_loader import load_center_library
from isa.measured_deployment.operator import DeploymentAwareCIMLinear

__all__ = ["DeploymentAwareCIMLinear", "load_center_library"]
