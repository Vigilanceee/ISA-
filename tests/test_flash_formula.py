import pytest


torch = pytest.importorskip("torch")


def _reference_current(vgs, vth, cfg):
    inv_2nut = 1.0 / (2.0 * cfg["n"] * cfg["U_T"])
    u1 = (vgs - vth) * inv_2nut
    u2 = (vgs - vth - cfg["V_D"]) * inv_2nut
    basic = torch.nn.functional.softplus(u1).square()
    basic = basic - torch.nn.functional.softplus(u2).square()
    return cfg["I_S"] * basic / (1.0 + vgs / cfg["V_sat"])


def test_flash_ekv_uses_direct_drain_voltage_term():
    from isa.device_models.flash_transistor import ekv_current

    cfg = {
        "I_S": 0.75875,
        "n": 4.1360,
        "U_T": 0.026,
        "V_D": 0.1,
        "V_sat": 8.2624,
    }
    vgs = torch.tensor([0.4, 1.2, 2.7], dtype=torch.float64)
    vth = torch.tensor([0.1, 0.8, 2.0], dtype=torch.float64)

    actual = ekv_current(vgs, vth, cfg)
    expected = _reference_current(vgs, vth, cfg)
    assert torch.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

def test_flash_analytic_gradients_match_autograd_with_direct_vd():
    from isa.device_models.flash_transistor import (
        ekv_current,
        ekv_current_grad_v,
        ekv_current_grad_vth,
    )

    cfg = {
        "I_S": 0.75875,
        "n": 4.1360,
        "U_T": 0.026,
        "V_D": 0.1,
        "V_sat": 8.2624,
    }
    vgs = torch.tensor([0.5, 1.4, 3.0], dtype=torch.float64, requires_grad=True)
    vth = torch.tensor([0.2, 0.9, 2.2], dtype=torch.float64, requires_grad=True)
    current = ekv_current(vgs, vth, cfg)
    grad_v, grad_vth = torch.autograd.grad(current.sum(), (vgs, vth))

    assert torch.allclose(ekv_current_grad_v(vgs, vth, cfg), grad_v, rtol=1.0e-10, atol=1.0e-10)
    assert torch.allclose(ekv_current_grad_vth(vgs, vth, cfg), grad_vth, rtol=1.0e-10, atol=1.0e-10)


def test_flash_delta_lut_uses_the_public_formula():
    from isa.approximations.delta_lut import _cpu_lut_cached

    cfg = {
        "I_S": 0.75875,
        "n": 4.1360,
        "U_T": 0.026,
        "V_D": 0.1,
        "V_sat": 8.2624,
    }
    delta_min, delta_max, size = -1.0, 3.0, 257
    key = (
        cfg["n"],
        cfg["V_sat"],
        cfg["I_S"],
        cfg["U_T"],
        cfg["V_D"],
        delta_min,
        delta_max,
        size,
    )
    table, _ = _cpu_lut_cached(key)
    delta = torch.linspace(delta_min, delta_max, size, dtype=torch.float32)
    inv_2nut = 1.0 / (2.0 * cfg["n"] * cfg["U_T"])
    expected = cfg["I_S"] * (
        torch.nn.functional.softplus(delta * inv_2nut).square()
        - torch.nn.functional.softplus((delta - cfg["V_D"]) * inv_2nut).square()
    )
    assert torch.allclose(table.float(), expected, rtol=8.0e-3, atol=3.0e-3)
