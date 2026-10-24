import copy

import pytest
import torch

from isa.device_sweeps.checkpoints import (
    build_checkpoint_model,
    checkpoint_state,
    infer_calibration_slots,
)
from isa.device_sweeps.models.mlp import MLP
from isa.device_sweeps.models.peripherals import FixedGateCalibration
from isa.device_sweeps.models.vgg8 import VGG8
from isa.kernels.device_sweep.lut_triton import _make_fefet_tables
from isa.kernels.device_sweep.polynomial import FlashPolynomialFunction, polynomial_coefficients
from isa.kernels.device_sweep.custom_layers import NVM_Linear, _init_weights


PARAMS = {"w_min": 0.2, "w_max": 1.7, "I0": 200.0, "V0": 0.35, "g0": 0.15}


def test_selected_study_initialization_width_is_explicit():
    params = {"init_center": 2.5, "init_half_width_multiplier": 0.1}
    positive, negative = _init_weights(4, 3, "flash", params, 0.0, 5.0, 0.5)
    for state in (positive, negative):
        assert torch.all((state >= 2.45) & (state <= 2.55))
    assert not torch.equal(positive, negative)


@pytest.mark.parametrize("slots", [False, True])
def test_mlp_checkpoint_layout_strictly_roundtrips(slots):
    params = {**PARAMS, "calibration_slots": slots}
    original = MLP("reram", params)
    state = original.state_dict()
    assert infer_calibration_slots("MLP", state) is slots
    config = {
        "architecture": "MLP", "device": "ReRAM",
        "device_parameters": PARAMS, "model_parameters": {},
    }
    loaded = build_checkpoint_model(config, state)
    assert list(loaded.state_dict()) == list(state)
    assert all(torch.equal(state[key], loaded.state_dict()[key]) for key in state)


@pytest.mark.parametrize("slots", [False, True])
def test_vgg8_checkpoint_layout_and_calibration(slots):
    params = {
        "w_min": 0.0, "w_max": 4.0, "calibration_slots": slots,
        "op_calibration_enabled": slots,
        "op_calibration_center": 2.5, "op_calibration_scale": 0.02,
        "output_gain": 0.01,
    }
    model = VGG8("fefet", params)
    state = model.state_dict()
    assert infer_calibration_slots("VGG8", state) is slots
    assert sum(isinstance(layer, FixedGateCalibration) for layer in model.modules()) == (7 if slots else 0)
    assert model.output_gain.gain == 0.01
    config = {
        "architecture": "VGG8", "device": "FeFET",
        "device_parameters": params, "model_parameters": {},
    }
    loaded = build_checkpoint_model(config, state)
    assert list(loaded.state_dict()) == list(state)


def test_checkpoint_loader_rejects_unknown_or_nonfinite_state():
    with pytest.raises(ValueError, match="non-tensor"):
        checkpoint_state({"weights": "not a tensor"})
    with pytest.raises(ValueError, match="non-finite"):
        checkpoint_state({"weight": torch.tensor(float("nan"))})
    with pytest.raises(ValueError, match="layout"):
        infer_calibration_slots("MLP", {"classifier.7.theta_pos": torch.zeros(1)})


def test_vgg8_uses_the_checkpoint_tia_interval():
    model = VGG8("flash", {"tia_v_min": 0.0, "tia_v_max": 4.0, "w_min": 0.0, "w_max": 5.0})
    assert model.features[1].v_min == 0.0
    assert model.features[1].v_max == 4.0
    assert model.classifier[1].v_min == 0.0


def test_fefet_surrogate_changes_only_the_backward_table():
    exact = {
        "I_S": 6.37517e-6, "n": 1.03653, "U_T": 0.02585, "V_D": 0.1,
        "A_lk": 0.832635, "B_lk": 1.289047, "V_sat": 5.0,
        "lut_size": 512, "surrogate_backward_enabled": False,
    }
    surrogate = copy.deepcopy(exact)
    surrogate.update(surrogate_backward_enabled=True, surrogate_backward_alpha=1.0, surrogate_A_lk=0.1)
    y_exact, dy_exact, _ = _make_fefet_tables(exact, torch.device("cpu"), torch.float32)
    y_surrogate, dy_surrogate, _ = _make_fefet_tables(surrogate, torch.device("cpu"), torch.float32)
    assert torch.equal(y_exact, y_surrogate)
    assert not torch.allclose(dy_exact, dy_surrogate)
    assert torch.isfinite(dy_surrogate).all()


def test_flash_polynomial_array_matches_independent_scalar_horner_evaluation():
    params = {"n": 4.136, "U_T": 0.026, "V_D": 0.1, "V_sat": 8.2624,
              "I_S": 0.75875, "poly_degree": 5, "poly_fit_points": 4096,
              "poly_x_min": -5.0, "poly_x_max": 4.0, "poly_bf16": False}
    v = torch.tensor([[0.2, 0.7, 2.1], [1.0, 3.0, 3.8]], requires_grad=True)
    wp = torch.tensor([[0.3, 1.2, 2.0], [0.8, 2.0, 3.1]], requires_grad=True)
    wn = torch.tensor([[0.7, 1.6, 2.2], [1.0, 1.1, 2.7]], requires_grad=True)
    packed = polynomial_coefficients(params, v.device)

    def scalar_response(w):
        z = (v[:, None, :] - w[None, :, :] - packed[0]) / packed[1]
        y = torch.zeros_like(z)
        for coefficient in packed[2:].flip(0):
            y = y * z + coefficient
        return params["I_S"] * y / (1.0 + v[:, None, :] / params["V_sat"])

    expected = (scalar_response(wp) - scalar_response(wn)).sum(-1)
    actual = FlashPolynomialFunction.apply(v, wp, wn, params)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    actual_grads = torch.autograd.grad(actual.sum(), (v, wp, wn), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (v, wp, wn))
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-4)


def test_pcm_tanh_state_parameterization_preserves_forward_and_chain_rule():
    params = {"I0_pcm": 1.0, "V_T_pcm": 1.0, "w_min": 0.0, "w_max": 1.0,
              "planar_enabled": True, "planar_order": 7, "state_mapping": "tanh"}
    layer = NVM_Linear(3, 2, "pcm", params)
    with torch.no_grad():
        layer.theta_pos.fill_(0.2)
        layer.theta_neg.fill_(0.3)
    v = torch.tensor([[0.1, 0.2, -0.3]], requires_grad=True)
    actual = layer(v)
    p, n = layer.theta_pos.tanh(), layer.theta_neg.tanh()
    expected = (p[None] * torch.sinh(p[None] * v[:, None]) -
                n[None] * torch.sinh(n[None] * v[:, None])).sum(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    actual_grads = torch.autograd.grad(actual.sum(), (v, layer.theta_pos, layer.theta_neg), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (v, layer.theta_pos, layer.theta_neg))
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol=1e-6, atol=1e-8)
