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
