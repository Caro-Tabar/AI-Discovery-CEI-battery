"""Tests for Phase 9 regularized and nonlinear surrogate models."""

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from cei_scout.ml.baselines import evaluate_baselines
from cei_scout.ml.dataset import load_ml_dataset
from cei_scout.ml.surrogates import (
    evaluate_surrogates,
    make_inner_splits,
    predict_outer_fold,
    write_surrogate_results,
)
from cei_scout.ml.validation import make_cv_folds

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9_ml_v1.yaml"


@pytest.fixture(scope="module")
def prepared():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    dataset = load_ml_dataset(CONFIG)
    results = evaluate_surrogates(dataset, config)
    return dataset, config, results


def test_output_counts_and_metrics(prepared):
    dataset, config, (predictions, folds, summary) = prepared

    assert len(dataset.candidate_ids) == 56
    assert config["models"]["candidates"] == ["ridge", "extra_trees"]

    assert len(predictions) == 1344
    assert len(folds) == 124
    assert len(summary) == 8

    for row in summary:
        assert row["n_unique_candidates"] == 56
        assert row["n_predictions"] == (280 if row["validation"] == "primary" else 56)
        assert np.isfinite(row["mae_ev"])
        assert np.isfinite(row["rmse_ev"])
        assert row["rmse_ev"] >= row["mae_ev"] >= 0

    for row in folds:
        if row["model"] == "ridge":
            assert row["selected_alpha"] in config["models"]["ridge"]["alpha_grid"]
            assert np.isfinite(row["inner_cv_mae_ev"])
            assert row["n_inner_folds"] == (4 if row["validation"] == "primary" else 5)
        else:
            assert row["selected_alpha"] == ""
            assert row["n_inner_folds"] == 0


def test_inner_family_splits_are_disjoint(prepared):
    dataset, config, _ = prepared
    outer = make_cv_folds(dataset, config, "family")[0]

    families = tuple(dataset.families[i] for i in outer.train_indices)

    inner = make_inner_splits(families, config, "family")

    assert len(set(families)) == 5
    assert len(inner) == 5

    for train, test in inner:
        training_groups = {families[i] for i in train}
        testing_groups = {families[i] for i in test}

        assert len(testing_groups) == 1
        assert not training_groups & testing_groups
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(range(len(families)))


def test_held_out_targets_cannot_change_predictions(prepared):
    dataset, config, _ = prepared
    fold = make_cv_folds(dataset, config, "family")[0]
    target = "ionization_energy"

    altered = dataset.targets[target].copy()
    altered[list(fold.test_indices)] += 1000.0

    changed_dataset = replace(
        dataset,
        targets={**dataset.targets, target: altered},
    )

    for model in ("ridge", "extra_trees"):
        original, details = predict_outer_fold(
            dataset,
            config,
            target,
            model,
            "family",
            fold.train_indices,
            fold.test_indices,
        )
        repeated, changed_details = predict_outer_fold(
            changed_dataset,
            config,
            target,
            model,
            "family",
            fold.train_indices,
            fold.test_indices,
        )

        np.testing.assert_array_equal(original, repeated)
        assert details == changed_details


def test_outer_predictions_match_baseline_splits(prepared):
    dataset, config, (surrogate_predictions, _, _) = prepared
    baseline_predictions, _, _ = evaluate_baselines(dataset, config)

    def keys(rows):
        return Counter(
            (
                row["validation"],
                row["target"],
                row["fold"],
                row["candidate_id"],
            )
            for row in rows
        )

    assert keys(surrogate_predictions) == keys(baseline_predictions)


def test_reproducible_fold_and_versioned_outputs(prepared, tmp_path):
    dataset, config, (predictions, folds, summary) = prepared

    fold = make_cv_folds(dataset, config, "primary")[0]

    for model in ("ridge", "extra_trees"):
        first = predict_outer_fold(
            dataset,
            config,
            "electron_affinity",
            model,
            "primary",
            fold.train_indices,
            fold.test_indices,
        )
        second = predict_outer_fold(
            dataset,
            config,
            "electron_affinity",
            model,
            "primary",
            fold.train_indices,
            fold.test_indices,
        )

        np.testing.assert_array_equal(first[0], second[0])
        assert first[1] == second[1]

    write_surrogate_results(
        tmp_path,
        CONFIG,
        config,
        predictions,
        folds,
        summary,
    )

    expected = {
        "surrogate_predictions_v1.csv": 1344,
        "surrogate_fold_metrics_v1.csv": 124,
        "surrogate_summary_v1.csv": 8,
    }

    for filename, expected_rows in expected.items():
        with (tmp_path / filename).open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == expected_rows

    metadata = json.loads(
        (tmp_path / "surrogate_metadata_v1.json").read_text(encoding="utf-8")
    )

    assert metadata["output_version"] == 1
    assert metadata["prediction_policy"] == "outer_fold_held_out_only"
    assert set(metadata["input_sha256"]) == {
        "configuration",
        "labeled",
        "prospective",
    }
