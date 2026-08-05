from isa.device_sweeps.backend import (
    normalize_backend_parameters,
    select_primitive_backend,
)


def test_explicit_reference_ignores_stale_lut_flag():
    selected = select_primitive_backend(
        {"lut_enabled": True}, explicit_backend="reference"
    )
    assert selected == "reference"


def test_explicit_lut_is_still_selectable():
    selected = select_primitive_backend(
        {"lut_enabled": False}, explicit_backend="lut"
    )
    assert selected == "lut"


def test_reference_override_normalizes_legacy_acceleration_flags():
    resolved = normalize_backend_parameters(
        {
            "conv_backend": "lowrank_planar",
            "linear_backend": "lowrank_planar",
            "lut_enabled": True,
            "direct_conv_enabled": True,
        },
        conv_backend="reference",
        linear_backend="reference",
    )
    assert resolved["conv_backend"] == "reference"
    assert resolved["linear_backend"] == "reference"
    assert resolved["lut_enabled"] is False
    assert resolved["direct_conv_enabled"] is False
