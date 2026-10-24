"""Pure backend-selection helpers shared by CLI and NVM layers."""

from __future__ import annotations

BACKENDS = (
    "exact",
    "reference",
    "lut",
    "planar",
    "polynomial",
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
        if explicit_backend in {"lut", "planar", "polynomial"}:
            return explicit_backend
        return "reference"
    params = device_params or {}
    if bool(params.get("poly_enabled", False)):
        return "polynomial"
    if bool(params.get("planar_enabled", False)):
        return "planar"
    if bool(params.get("lut_enabled", False)):
        return "lut"
    return "reference"


def configured_layer_backend(params: dict, layer: str) -> str:
    """Resolve a layer backend, including configurations saved with flags only."""

    if layer not in {"conv", "linear"}:
        raise ValueError(f"Unknown layer kind: {layer}")
    explicit = params.get(f"{layer}_backend")
    if explicit is None and layer == "linear":
        explicit = params.get("conv_backend")
    if explicit:
        select_primitive_backend(params, str(explicit))
        return str(explicit)
    return select_primitive_backend(params, None)


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
    resolved["planar_enabled"] = "planar" in {conv, linear}
    resolved["poly_enabled"] = "polynomial" in {conv, linear}
    return resolved
