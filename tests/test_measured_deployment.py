import csv
import json
from pathlib import Path

import numpy as np
import torch

from isa.measured_deployment.fg50_runtime import state_partition
from isa.measured_deployment.operator import DeploymentAwareCIMLinear
from isa.measured_deployment.ragged import sample_member_ids

ROOT = Path(__file__).resolve().parents[1]


def test_state_partition_preserves_published_counts_and_stable_order():
    score = np.asarray([0.4, 0.1, 0.3, 0.2, 0.2], dtype=float)
    counts = np.asarray([2, 3], dtype=np.int32)
    state_ids, ordered_ids, offsets = state_partition(score, counts)
    assert np.bincount(state_ids, minlength=2).tolist() == [2, 3]
    assert ordered_ids.tolist() == [1, 3, 4, 2, 0]
    assert offsets.tolist() == [0, 2, 5]


def test_ragged_member_sampler_is_seeded_and_respects_state_ranges():
    state_ids = torch.tensor([[0, 1, 1, 0]], dtype=torch.int16)
    members = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
    first = torch.Generator(device="cpu").manual_seed(123)
    second = torch.Generator(device="cpu").manual_seed(123)
    draw_a = sample_member_ids(state_ids, members, offsets, first)
    draw_b = sample_member_ids(state_ids, members, offsets, second)
    assert torch.equal(draw_a, draw_b)
    assert set(draw_a[state_ids == 0].tolist()) <= {10, 11}
    assert set(draw_a[state_ids == 1].tolist()) <= {20, 21, 22}


def test_measured_operator_matches_piecewise_linear_reference_on_cpu():
    layer = DeploymentAwareCIMLinear(2, 1)
    layer.set_r_tia(1.0)
    voltage = torch.tensor([0.0, 1.0, 2.0])
    currents = torch.tensor([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    layer.configure_codebook(voltage, currents)
    layer.set_assignment(
        torch.tensor([[1, 1]], dtype=torch.int16),
        torch.tensor([[0, 0]], dtype=torch.int16),
    )
    layer.set_forward_mode("measured")
    output = layer(torch.tensor([[[0.25, 0.50]]]))
    assert torch.allclose(output, torch.tensor([[[0.75]]]), atol=1e-6)


def test_committed_monte_carlo_rows_are_complete_and_match_summary():
    sample_path = ROOT / "experiments/fg50_24state/results/mc_samples_200.csv"
    rows = list(csv.DictReader(sample_path.open(newline="", encoding="utf-8")))
    seeds = [int(row["seed"]) for row in rows]
    accuracy = np.asarray([float(row["accuracy"]) for row in rows])
    summary = json.loads(
        (ROOT / "experiments/fg50_24state/results/mc_summary.json").read_text()
    )
    assert seeds == list(range(10000, 10200))
    assert len(set(seeds)) == 200
    assert np.isclose(accuracy.mean(), summary["accuracy_mean"], atol=1e-12)
    assert np.isclose(accuracy.std(ddof=1), summary["accuracy_std"], atol=1e-12)


def test_selected_device_table_uses_legacy_fefet_and_old_reram_result():
    rows = list(
        csv.DictReader(
            (ROOT / "results/device_sweep_selected.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    by_key = {(row["architecture"], row["device"]): row for row in rows}
    assert float(by_key[("VGG8", "ReRAM")]["best_val_percent"]) == 87.50
    assert float(by_key[("VGG8", "FeFET")]["best_val_percent"]) == 83.79
    assert by_key[("VGG8", "FeFET")]["fefet_fit_group"].startswith(
        "legacy_poor_fit"
    )
