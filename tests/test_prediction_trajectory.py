from __future__ import annotations

import numpy as np

from isa.prediction_trajectory.analysis import (
    cosine_similarity,
    interpolate_first_crossing,
    joint_pca,
    normalized_distance,
    select_endpoint_index,
)
from isa.prediction_trajectory.protocol import stratified_probe_indices


def test_stratified_probe_indices_are_balanced_and_deterministic() -> None:
    labels = np.repeat(np.arange(10), 500)
    first = stratified_probe_indices(labels, size=1000, seed=20260802)
    second = stratified_probe_indices(labels, size=1000, seed=20260802)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 1000
    counts = np.bincount(labels[first], minlength=10)
    assert counts.tolist() == [100] * 10


def test_joint_pca_recovers_low_rank_variance() -> None:
    generator = np.random.default_rng(7)
    latent = generator.normal(size=(24, 2))
    basis = generator.normal(size=(2, 40))
    matrix = latent @ basis
    coordinates, explained = joint_pca(matrix)
    assert coordinates.shape[0] == 24
    assert explained[:2].sum() > 1.0 - 1e-12
    assert np.isclose(explained.sum(), 1.0)


def test_first_crossing_interpolates_prediction_and_epoch() -> None:
    epochs = np.asarray([0, 5, 10])
    accuracies = np.asarray([0.10, 0.30, 0.70])
    vectors = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 6.0]])
    vector, epoch = interpolate_first_crossing(epochs, accuracies, vectors, 0.50)
    assert np.allclose(vector, [2.0, 4.0])
    assert np.isclose(epoch, 7.5)


def test_prediction_metrics_have_expected_scale() -> None:
    first = np.asarray([1.0, 0.0, 0.0, 0.0])
    second = np.asarray([1.0, 1.0, 0.0, 0.0])
    assert np.isclose(cosine_similarity(first, second), 1.0 / np.sqrt(2.0))
    assert np.isclose(normalized_distance(first, second), 0.5)


def test_endpoint_selection_distinguishes_best_and_final() -> None:
    accuracies = np.asarray([0.10, 0.72, 0.81, 0.79])
    assert select_endpoint_index(accuracies, "best") == 2
    assert select_endpoint_index(accuracies, "final") == 3
