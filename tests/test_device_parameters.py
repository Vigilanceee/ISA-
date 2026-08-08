from pathlib import Path

import pytest
import yaml

from isa.device_models.config import FLASH_TRANSISTOR_PARAMETERS


ROOT = Path(__file__).resolve().parents[1]


def test_public_flash_parameter_file_matches_runtime_defaults():
    payload = yaml.safe_load(
        (ROOT / "configs/devices/flash_transistor_ekv.yaml").read_text(
            encoding="utf-8"
        )
    )
    parameters = payload["parameters"]
    assert parameters["I_S"] == pytest.approx(FLASH_TRANSISTOR_PARAMETERS["I_S"])
    assert parameters["n"] == pytest.approx(FLASH_TRANSISTOR_PARAMETERS["n"])
    assert parameters["U_T"] == pytest.approx(FLASH_TRANSISTOR_PARAMETERS["U_T"])
    assert parameters["V_D"] == pytest.approx(FLASH_TRANSISTOR_PARAMETERS["V_D"])
    assert parameters["V_sat"] == pytest.approx(FLASH_TRANSISTOR_PARAMETERS["V_sat"])
    assert payload["ranges"]["Vth"] == [
        FLASH_TRANSISTOR_PARAMETERS["V_TH_MIN"],
        FLASH_TRANSISTOR_PARAMETERS["V_TH_MAX"],
    ]


@pytest.mark.parametrize(
    "relative",
    (
        "configs/device_sweeps/device_params.yaml",
        "configs/device_sweeps/device_params_fefet_latest_fit.yaml",
    ),
)
def test_device_sweeps_default_to_exact_physical_operators(relative):
    payload = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert payload["reram"]["conv_backend"] == "factorized"
    assert payload["stt"]["conv_backend"] == "factorized"
    for device in ("pcm", "fefet", "flash"):
        assert payload[device]["conv_backend"] == "exact"
        assert payload[device]["linear_backend"] == "exact"
        assert payload[device]["direct_conv_enabled"] is False
        assert payload[device]["lut_enabled"] is False
        assert not any(key.startswith("lowrank_") for key in payload[device])
    assert payload["pcm"]["raw_kernel_backend"] == "split"
    assert payload["pcm"]["raw_forward_backend"] == "split_k"


def test_fefet_uses_supplied_clean_fit_formula_parameters():
    payload = yaml.safe_load(
        (ROOT / "configs/device_sweeps/device_params.yaml").read_text(
            encoding="utf-8"
        )
    )["fefet"]
    assert payload["I_S"] == pytest.approx(6.3026)
    assert payload["n"] == pytest.approx(1.0280)
    assert payload["A_lk"] == pytest.approx(0.8214)
    assert payload["B_lk"] == pytest.approx(1.3171)
    assert payload["V_sat"] == pytest.approx(5.0)
    assert payload["fitted_vth_states"] == pytest.approx(
        [1.8711, 2.1054, 2.3445, 2.5970, 2.8471, 3.1103, 3.3727, 3.7399]
    )
    assert payload["w_min"] == pytest.approx(1.8711)
    assert payload["w_max"] == pytest.approx(3.7399)


def test_flash_uses_exact_split_reduction_backend():
    payload = yaml.safe_load(
        (ROOT / "configs/device_sweeps/device_params.yaml").read_text(
            encoding="utf-8"
        )
    )["flash"]
    assert payload["exact_matrix_backend"] == "split"
    assert payload["raw_kernel_backend"] == "split"
    assert payload["raw_forward_backend"] == "split_k"
