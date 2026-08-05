from pathlib import Path

import pytest
import yaml

from isa.device_sweeps import fefet_health_runner as runner
from isa.device_sweeps.fefet_health_runner import (
    independent_seed_environment,
    terminal_record,
    validate_execution_options,
    worker_gpu_ids,
)
from isa.device_sweeps.health import evaluate_training_health

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative",
    (
        "configs/device_sweeps/device_params.yaml",
        "configs/device_sweeps/device_params_fefet_latest_fit.yaml",
    ),
)
def test_fefet_configs_select_raw_reference_backend(relative):
    payload = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    fefet = payload["fefet"]
    assert fefet["conv_backend"] == "reference"
    assert fefet["linear_backend"] == "reference"
    assert fefet["raw_kernel_backend"] == "split"
    assert fefet["raw_forward_backend"] == "split_k"
    assert fefet["lut_enabled"] is False
    assert fefet["planar_enabled"] is False
    assert fefet["direct_conv_enabled"] is False
    assert not any(key.startswith("lowrank_") for key in fefet)
    assert "planar_nodes" not in fefet


def decision(epoch, history, **metrics):
    defaults = {
        "train_loss": 1.0,
        "train_acc": 0.2,
        "val_loss": 1.0,
        "val_acc": history[-1],
    }
    defaults.update(metrics)
    return evaluate_training_health(
        epoch=epoch,
        val_history=history,
        threshold_pruning=True,
        **defaults,
    )


def test_non_finite_metric_stops_immediately():
    result = decision(1, [0.1], train_loss=float("nan"))
    assert result.should_stop
    assert result.reason == "non_finite_metric"


def test_epoch8_rule_uses_best_accuracy():
    failing = decision(8, [0.10, 0.12, 0.14, 0.13])
    healthy = decision(8, [0.10, 0.15, 0.14, 0.13])
    assert failing.should_stop
    assert failing.reason.startswith("epoch8_best_below")
    assert not healthy.should_stop


def test_epoch20_rule_requires_low_best_and_low_recent_gain():
    stalled = [0.20] * 15 + [0.25, 0.255, 0.257, 0.258, 0.259]
    improving = [0.20] * 15 + [0.25, 0.26, 0.27, 0.28, 0.29]
    high_best = [0.36] + [0.30] * 19
    assert decision(20, stalled).should_stop
    assert not decision(20, improving).should_stop
    assert not decision(20, high_best).should_stop


def test_threshold_rules_can_be_disabled_without_disabling_nan_check():
    result = evaluate_training_health(
        epoch=20,
        train_loss=1.0,
        train_acc=0.1,
        val_loss=1.0,
        val_acc=0.1,
        val_history=[0.1] * 20,
        threshold_pruning=False,
    )
    assert not result.should_stop


def test_five_seed_manifest_is_fixed_and_resumable():
    manifest = yaml.safe_load(
        (
            ROOT / "experiments/fefet_raw_vgg8/manifest.yaml"
        ).read_text(encoding="utf-8")
    )
    assert manifest["device"] == "fefet"
    assert manifest["model"] == "vgg8"
    assert len(manifest["seeds"]) == 5
    assert len(set(manifest["seeds"])) == 5
    assert manifest["fixed_hyperparameters"] == {
        "provenance": (
            "20260729 FeFET clean-fit T1; backend-only reference/raw split"
        ),
        "lr": pytest.approx(0.014701762450),
        "init_center": pytest.approx(2.214381698),
        "init_half_width": pytest.approx(1.020930704),
        "tia_r": pytest.approx(133.2815),
    }
    assert manifest["health"] == {
        "non_finite": "immediate",
        "epoch8_min_best": 0.15,
        "epoch20_min_best": 0.35,
        "plateau_window": 5,
        "plateau_min_gain": 0.02,
    }


def test_terminal_record_allows_completed_or_pruned_seed_to_be_skipped(tmp_path):
    marker = tmp_path / "terminal.json"
    marker.write_text(
        '{"seed": 1, "state": "PRUNED", "stop_reason": "health"}\n',
        encoding="utf-8",
    )
    assert terminal_record(tmp_path)["state"] == "PRUNED"


def test_seed_parallelism_and_ddp_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_execution_options(parallel_seeds=2, nproc_per_node=2)
    validate_execution_options(parallel_seeds=2, nproc_per_node=1)
    validate_execution_options(parallel_seeds=1, nproc_per_node=2)


def test_parallel_workers_receive_distinct_visible_gpu_ids():
    assert worker_gpu_ids(
        2, {"CUDA_VISIBLE_DEVICES": "GPU-alpha,GPU-beta"}
    ) == ["GPU-alpha", "GPU-beta"]
    with pytest.raises(ValueError, match="requires 2 visible GPUs"):
        worker_gpu_ids(2, {"CUDA_VISIBLE_DEVICES": "0"})


def test_independent_seed_environment_removes_acp_ddp_variables():
    environment = independent_seed_environment(
        "1",
        {
            "PATH": "/usr/bin",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "platform-master",
            "MASTER_PORT": "23456",
        },
    )
    assert environment == {
        "PATH": "/usr/bin",
        "CUDA_VISIBLE_DEVICES": "1",
    }


def test_seed_launcher_sets_gpu_and_persists_log(tmp_path, monkeypatch):
    manifest = yaml.safe_load(
        (
            ROOT / "experiments/fefet_raw_vgg8/manifest.yaml"
        ).read_text(encoding="utf-8")
    )
    captured = {}

    def fake_subprocess_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        kwargs["stdout"].write("mock training output\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        runner,
        "write_terminal_record",
        lambda seed_dir, seed: {
            "seed": seed,
            "state": "PRUNED",
            "best_val_acc": 0.1,
            "stop_reason": "health",
        },
    )
    record = runner.run_seed(
        manifest,
        repo_root=ROOT,
        data_dir=tmp_path / "data",
        output_root=tmp_path / "runs",
        seed=123,
        nproc_per_node=1,
        gpu_id="GPU-beta",
        dry_run=False,
    )
    log_path = tmp_path / "runs/seed_123/launcher.log"
    assert record["state"] == "PRUNED"
    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-beta"
    assert "mock training output" in log_path.read_text(encoding="utf-8")
