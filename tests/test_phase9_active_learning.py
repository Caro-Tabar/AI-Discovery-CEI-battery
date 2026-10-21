"""Integrity tests for retrospective active learning."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from cei_scout.ml.active_learning import (
    make_partition,
    run_experiment,
    select_uncertain,
)
from cei_scout.ml.dataset import load_ml_dataset
from cei_scout.ml.extended_models import predict_outer_fold

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    base = ROOT / "configs/phase9_ml_v1.yaml"
    extended = ROOT / "configs/phase9_extended_models_v1.yaml"
    active = ROOT / "configs/phase9_active_learning_v1.yaml"

    dataset = load_ml_dataset(base)
    settings = yaml.safe_load(extended.read_text(encoding="utf-8"))
    config = yaml.safe_load(active.read_text(encoding="utf-8"))

    return dataset, settings, config


def test_initial_and_test_partitions_are_disjoint(inputs):
    dataset, _, config = inputs
    experiment = config["experiment"]

    for seed in experiment["seeds"]:
        initial, pool, test = make_partition(
            dataset,
            seed,
            experiment["test_size"],
            experiment["initial_labeled_size"],
        )

        assert len(initial) == 12
        assert len(pool) == 32
        assert len(test) == 12
        assert not set(initial) & set(pool)
        assert not set(initial) & set(test)
        assert not set(pool) & set(test)
        assert set(initial) | set(pool) | set(test) == set(range(56))


def test_uncertainty_selection_rejects_invalid_values(inputs):
    dataset, _, _ = inputs
    initial, pool, _ = make_partition(dataset, 1729, 12, 12)

    sigma = {
        "ionization_energy": np.ones(len(pool)),
        "electron_affinity": np.ones(len(pool)),
    }

    chosen, _ = select_uncertain(dataset, initial, pool, sigma, 2)
    assert len(chosen) == 2
    assert set(chosen).issubset(pool)

    sigma["electron_affinity"][0] = -1.0

    with pytest.raises(ValueError, match="Negative pool uncertainty"):
        select_uncertain(dataset, initial, pool, sigma, 2)


def test_real_gpr_runs_on_small_initial_set(inputs):
    dataset, extended, _ = inputs
    initial, pool, test = make_partition(dataset, 1729, 12, 12)

    predicted, sigma, _ = predict_outer_fold(
        dataset,
        extended,
        "ionization_energy",
        "gaussian_process",
        initial,
        (*test, *pool),
    )

    assert predicted.shape == (44,)
    assert sigma is not None
    assert sigma.shape == (44,)
    assert np.isfinite(predicted).all()
    assert np.isfinite(sigma).all()
    assert np.all(sigma >= 0)


def test_simulation_integrity_and_no_test_label_leakage(inputs, monkeypatch):
    dataset, extended, original = inputs
    config = deepcopy(original)
    config["experiment"]["seeds"] = [1729, 1730]
    config["experiment"]["acquisition_rounds"] = 1

    def lightweight_gpr(data, settings, target, model, train_indices, test_indices):
        assert model == "gaussian_process"

        y_train = data.targets[target][list(train_indices)]
        estimates = np.full(len(test_indices), np.mean(y_train))

        sigma = np.asarray(
            [0.1 + 0.01 * abs(data.X[i, 0]) for i in test_indices],
            dtype=float,
        )

        return estimates, sigma, ()

    monkeypatch.setattr(
        "cei_scout.ml.active_learning.predict_outer_fold",
        lightweight_gpr,
    )

    first = run_experiment(dataset, extended, config)
    second = run_experiment(dataset, extended, config)

    assert first == second

    curves, acquisitions, manifest, summary = first
    assert len(curves) == 16
    assert len(acquisitions) == 8
    assert len(manifest) == 112
    assert len(summary) == 8

    for seed in config["experiment"]["seeds"]:
        test_ids = {
            row["candidate_id"]
            for row in manifest
            if row["seed"] == seed and row["initial_role"] == "held_out_test"
        }
        selected = {row["candidate_id"] for row in acquisitions if row["seed"] == seed}
        assert not test_ids & selected

    # Changing held-out labels must not change acquisition decisions.
    # Test one seed independently to avoid cross-seed contamination.
    one_seed = deepcopy(config)
    one_seed["experiment"]["seeds"] = [1729]

    expected_acquisitions = run_experiment(dataset, extended, one_seed)[1]

    _, _, test_indices = make_partition(dataset, 1729, 12, 12)

    altered_targets = {
        target: values.copy() for target, values in dataset.targets.items()
    }

    for values in altered_targets.values():
        values[list(test_indices)] += 1000.0

    changed_dataset = replace(dataset, targets=altered_targets)

    changed_acquisitions = run_experiment(changed_dataset, extended, one_seed)[1]

    assert changed_acquisitions == expected_acquisitions
