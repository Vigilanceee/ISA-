from isa.device_sweeps.backend import (
    normalize_backend_parameters,
    select_primitive_backend,
)
from isa.kernels.device_sweep.custom_layers import _use_flash_ffn_exact_kernel


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


def test_exact_override_normalizes_legacy_approximation_flags():
    resolved = normalize_backend_parameters(
        {
            "conv_backend": "lowrank_planar",
            "linear_backend": "lowrank_planar",
            "lut_enabled": True,
            "direct_conv_enabled": True,
        },
        conv_backend="exact",
        linear_backend="exact",
    )
    assert resolved["conv_backend"] == "exact"
    assert resolved["linear_backend"] == "exact"
    assert resolved["lut_enabled"] is False
    # Direct convolution changes addressing/reduction only; it evaluates the
    # same physical equation and therefore remains valid for the exact backend.
    assert resolved["direct_conv_enabled"] is True


def test_exact_backend_selects_the_formula_primitive():
    selected = select_primitive_backend({}, explicit_backend="exact")
    assert selected == "reference"


def test_flash_exact_kernel_reuse_is_shape_aware():
    params = {
        "exact_matrix_backend": "auto",
        "exact_ffn_reuse_max_conv_fanin": 64,
    }
    assert _use_flash_ffn_exact_kernel(
        "flash", "exact", params, is_convolution=False, fan_in=8192
    )
    assert _use_flash_ffn_exact_kernel(
        "flash", "exact", params, is_convolution=True, fan_in=27
    )
    assert not _use_flash_ffn_exact_kernel(
        "flash", "exact", params, is_convolution=True, fan_in=1152
    )
    assert not _use_flash_ffn_exact_kernel(
        "flash",
        "exact",
        {"exact_matrix_backend": "split"},
        is_convolution=False,
        fan_in=8192,
    )
