"""Pure backend-selection helpers shared by CLI and NVM layers."""

from __future__ import annotations

BACKENDS = (
    "exact",
    "reference",
    "lut",
    "factorized",
    "node_planar",
)


def select_primitive_backend(
    device_params: dict | None,
    explicit_backend: str | None,
) -> str:
    """Select the primitive used by an NVM layer.

    Factorized/planar implementations are dispatched separately by the layer,
    so their fallback primitive is the reference formula. Explicit selection
    always takes precedence over legacy global flags.
    """

    if explicit_backend:
        if explicit_backend not in BACKENDS:
            raise ValueError(f"Unsupported NVM backend: {explicit_backend}")
        return "lut" if explicit_backend == "lut" else "reference"
    params = device_params or {}
    if bool(params.get("planar_enabled", False)):
        return "planar"
    if bool(params.get("lut_enabled", False)):
        return "lut"
    return "reference"


def normalize_backend_parameters(
    params: dict,
    *,
    conv_backend: str = "",
    linear_backend: str = "",
) -> dict:
    """Return device parameters with coherent, explicit layer backends."""

    resolved = dict(params)
    if conv_backend:
        resolved["conv_backend"] = conv_backend
    if linear_backend:
        resolved["linear_backend"] = linear_backend

    conv = str(resolved.get("conv_backend", "reference"))
    linear = str(resolved.get("linear_backend", conv))
    unknown = {conv, linear}.difference(BACKENDS)
    if unknown:
        raise ValueError(f"Unsupported NVM backend(s): {sorted(unknown)}")

    resolved["lut_enabled"] = "lut" in {conv, linear}
    resolved["planar_enabled"] = False
    return resolved
