from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_main_training_matrix_has_27_experiments():
    paths = (
        "configs/vision/cifar100.yaml",
        "configs/vision/imagenet200.yaml",
        "configs/language/openwebtext.yaml",
    )
    assert sum(len(load(path)["experiments"]) for path in paths) == 27


def test_every_main_matrix_has_all_variants_and_sizes():
    expected = {
        f"{variant}_{size}"
        for variant in ("digital", "hybrid", "physical")
        for size in ("s", "m", "l")
    }
    for path in (
        "configs/vision/cifar100.yaml",
        "configs/vision/imagenet200.yaml",
        "configs/language/openwebtext.yaml",
    ):
        assert set(load(path)["experiments"]) == expected


def test_device_matrix_has_ten_experiments():
    matrix = load("configs/device_sweeps/all_devices.yaml")
    assert len(matrix["experiments"]) == 10
