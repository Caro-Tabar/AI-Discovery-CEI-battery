"""Tests for the Gaussian-process and MLP experiments."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from cei_scout.ml.dataset import load_ml_dataset
from cei_scout.ml.extended_models import (
    evaluate_models,
    predict_outer_fold,
    write_results,
)
from cei_scout.ml.validation import make_cv_folds

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/phase9_ml_v1.yaml"
EXT_CONFIG = ROOT / "configs/phase9_extended_models_v1.yaml"


@pytest.fixture(scope="module")
def prepared():
    dataset = load_ml_dataset(BASE_CONFIG)
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    settings = yaml.safe_load(EXT_CONFIG.read_text(encoding="utf-8"))
    return dataset, base, settings


def test_models_return_valid_predictions(prepared):
    dataset, base, settings = prepared
    fold = make_cv_folds(dataset, base, "primary")[0]

    for model_name in ("gaussian_process", "mlp"):
        predicted, std, _ = predict_outer_fold(
            dataset,
            settings,
            "ionization_energy",
            model_name,
            fold.train_indices,
            fold.test_indices,
        )

        assert predicted.shape == (len(fold.test_indices),)
        assert np.isfinite(predicted).all()

        if model_name == "gaussian_process":
            assert std is not None
            assert std.shape == predicted.shape
            assert np.isfinite(std).all()
            assert np.all(std >= 0)
        else:
            assert std is None


def test_held_out_dft_targets_do_not_leak(prepared):
    dataset, base, settings = prepared
    fold = make_cv_folds(dataset, base, "family")[0]
    target = "electron_affinity"

    changed = dataset.targets[target].copy()
    changed[list(fold.test_indices)] += 1000.0

    altered_dataset = replace(
        dataset,
        targets={**dataset.targets, target: changed},
    )

    for model_name in ("gaussian_process", "mlp"):
        original, std_original, _ = predict_outer_fold(
            dataset,
            settings,
            target,
            model_name,
            fold.train_indices,
            fold.test_indices,
        )

        repeated, std_repeated, _ = predict_outer_fold(
            altered_dataset,
            settings,
            target,
            model_name,
            fold.train_indices,
            fold.test_indices,
        )

        np.testing.assert_array_equal(original, repeated)

        if std_original is not None:
            np.testing.assert_array_equal(std_original, std_repeated)


def test_invalid_model_is_rejected(prepared):
    dataset, base, settings = prepared
    fold = make_cv_folds(dataset, base, "primary")[0]

    with pytest.raises(ValueError, match="Unknown model"):
        predict_outer_fold(
            dataset,
            settings,
            "ionization_energy",
            "unsupported_model",
            fold.train_indices,
            fold.test_indices,
        )


def test_evaluation_counts_and_outputs(prepared, monkeypatch, tmp_path):
    dataset, base, settings = prepared

    def cheap_predict(data, params, target, model, train_indices, test_indices):
        training_mean = float(np.mean(data.targets[target][list(train_indices)]))
        predictions = np.full(len(test_indices), training_mean)
        std = np.full(len(test_indices), 0.2) if model == "gaussian_process" else None
        return predictions, std, ()

    monkeypatch.setattr(
        "cei_scout.ml.extended_models.predict_outer_fold",
        cheap_predict,
    )

    predictions, folds, summary = evaluate_models(dataset, base, settings)

    assert len(predictions) == 1344
    assert len(folds) == 124
    assert len(summary) == 8

    for row in summary:
        assert row["n_predictions"] == (280 if row["validation"] == "primary" else 56)
        assert np.isfinite(row["mae_ev"])
        assert np.isfinite(row["rmse_ev"])

    write_results(
        tmp_path,
        EXT_CONFIG,
        BASE_CONFIG,
        settings,
        base,
        predictions,
        folds,
        summary,
    )

    for filename, count in (
        ("extended_predictions_v1.csv", 1344),
        ("extended_fold_metrics_v1.csv", 124),
        ("extended_summary_v1.csv", 8),
    ):
        with (tmp_path / filename).open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == count

    metadata = json.loads(
        (tmp_path / "extended_metadata_v1.json").read_text(encoding="utf-8")
    )

    assert metadata["version"] == 1
    assert set(metadata["input_sha256"]) == {
        "extended_config",
        "base_config",
        "labeled",
        "prospective",
    }
