"""Tests for Phase 9 baseline regression and reporting."""

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.linear_model import LinearRegression

from cei_scout.ml.baselines import (
    evaluate_baselines,
    predict_fold,
    write_baseline_results,
)
from cei_scout.ml.dataset import load_ml_dataset
from cei_scout.ml.validation import make_cv_folds

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9_ml_v1.yaml"


@pytest.fixture(scope="module")
def prepared():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    dataset = load_ml_dataset(CONFIG)
    results = evaluate_baselines(dataset, config)
    return dataset, config, results


def test_expected_output_dimensions(prepared):
    dataset, _, (predictions, folds, summary) = prepared

    assert len(dataset.candidate_ids) == 56
    assert len(predictions) == 1344
    assert len(folds) == 124
    assert len(summary) == 8

    counts = Counter(
        (
            row["validation"],
            row["target"],
            row["model"],
            row["candidate_id"],
        )
        for row in predictions
    )

    for (validation, _, _, _), count in counts.items():
        assert count == (5 if validation == "primary" else 1)

    for row in summary:
        assert row["n_unique_candidates"] == 56
        assert row["n_predictions"] == (280 if row["validation"] == "primary" else 56)
        assert np.isfinite(row["mae_ev"])
        assert np.isfinite(row["rmse_ev"])
        assert row["rmse_ev"] >= row["mae_ev"] >= 0


def test_mean_baseline_uses_training_mean(prepared):
    dataset, config, _ = prepared
    fold = make_cv_folds(dataset, config, "primary")[0]

    predictions = predict_fold(
        dataset,
        config,
        "ionization_energy",
        "mean",
        fold.train_indices,
        fold.test_indices,
    )

    expected = np.mean(dataset.targets["ionization_energy"][list(fold.train_indices)])

    np.testing.assert_allclose(predictions, expected)


def test_linear_baseline_uses_correct_orbital(prepared):
    dataset, config, _ = prepared
    fold = make_cv_folds(dataset, config, "primary")[0]

    target = "electron_affinity"
    feature = config["features"]["simple_baselines"][target][0]
    column = dataset.feature_names.index(feature)

    X = dataset.X[:, [column]]
    y = dataset.targets[target]

    reference = LinearRegression()
    reference.fit(X[list(fold.train_indices)], y[list(fold.train_indices)])

    expected = reference.predict(X[list(fold.test_indices)])
    actual = predict_fold(
        dataset,
        config,
        target,
        "linear_regression",
        fold.train_indices,
        fold.test_indices,
    )

    np.testing.assert_allclose(actual, expected)


def test_held_out_target_cannot_influence_predictions(prepared):
    dataset, config, _ = prepared
    fold = make_cv_folds(dataset, config, "primary")[0]

    for model in ("mean", "linear_regression"):
        original = predict_fold(
            dataset,
            config,
            "ionization_energy",
            model,
            fold.train_indices,
            fold.test_indices,
        )

        altered_target = dataset.targets["ionization_energy"].copy()
        altered_target[list(fold.test_indices)] += 1000.0

        altered_dataset = replace(
            dataset,
            targets={
                **dataset.targets,
                "ionization_energy": altered_target,
            },
        )

        repeated = predict_fold(
            altered_dataset,
            config,
            "ionization_energy",
            model,
            fold.train_indices,
            fold.test_indices,
        )

        np.testing.assert_array_equal(original, repeated)


def test_reproducibility_and_versioned_outputs(prepared, tmp_path):
    dataset, config, original = prepared
    repeated = evaluate_baselines(dataset, config)

    assert original == repeated

    predictions, folds, summary = original

    write_baseline_results(
        tmp_path,
        CONFIG,
        config,
        predictions,
        folds,
        summary,
    )

    expected_files = {
        "baseline_predictions_v1.csv",
        "baseline_fold_metrics_v1.csv",
        "baseline_summary_v1.csv",
        "baseline_metadata_v1.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files

    for filename, expected_count in (
        ("baseline_predictions_v1.csv", 1344),
        ("baseline_fold_metrics_v1.csv", 124),
        ("baseline_summary_v1.csv", 8),
    ):
        with (tmp_path / filename).open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == expected_count

    metadata = json.loads(
        (tmp_path / "baseline_metadata_v1.json").read_text(encoding="utf-8")
    )

    assert metadata["output_version"] == 1
    assert metadata["prediction_policy"] == "held_out_only"
    assert set(metadata["input_sha256"]) == {
        "configuration",
        "labeled",
        "prospective",
    }
