"""Phase 9 dataset and cross-validation integrity tests."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from cei_scout.ml.dataset import load_ml_dataset
from cei_scout.ml.validation import make_cv_folds

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9_ml_v1.yaml"


@pytest.fixture(scope="module")
def inputs():
    data = load_ml_dataset(CONFIG)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return data, config


def test_dataset_population_and_numeric_data(inputs):
    data, cfg = inputs

    assert data.X.shape == (56, 8)
    assert len(data.candidate_ids) == len(set(data.candidate_ids)) == 56
    assert len(set(data.families)) == 6
    assert data.candidate_ids == tuple(sorted(data.candidate_ids))
    assert data.feature_names == tuple(cfg["features"]["primary"])
    assert set(data.targets) == {
        "ionization_energy",
        "electron_affinity",
    }

    assert np.isfinite(data.X).all()
    for values in data.targets.values():
        assert values.shape == (56,)
        assert np.isfinite(values).all()


def test_qc_retained_without_becoming_features(inputs):
    data, _ = inputs

    assert "CEI-0054" in data.candidate_ids
    assert not any(name.startswith("qc_") for name in data.feature_names)


def test_primary_cv_is_reproducible(inputs):
    data, cfg = inputs

    first = make_cv_folds(data, cfg, "primary")
    assert first == make_cv_folds(data, cfg, "primary")
    assert len(first) == 25

    test_counts = Counter()

    for fold in first:
        train = set(fold.train_indices)
        test = set(fold.test_indices)

        assert train.isdisjoint(test)
        assert train | test == set(range(56))
        assert fold.held_out_family is None

        test_counts.update(test)

    assert set(test_counts.values()) == {5}


def test_family_cv_has_no_family_overlap(inputs):
    data, cfg = inputs
    folds = make_cv_folds(data, cfg, "family")

    assert len(folds) == 6
    assert {f.held_out_family for f in folds} == set(data.families)

    test_indices = []

    for fold in folds:
        train_families = {data.families[i] for i in fold.train_indices}
        test_families = {data.families[i] for i in fold.test_indices}

        assert fold.held_out_family in test_families
        assert len(test_families) == 1
        assert train_families.isdisjoint(test_families)
        assert set(fold.train_indices).isdisjoint(fold.test_indices)

        test_indices.extend(fold.test_indices)

    assert sorted(test_indices) == list(range(56))


def test_target_cannot_become_predictor(tmp_path, inputs):
    _, original = inputs
    cfg = deepcopy(original)

    cfg["datasets"]["prospective"] = str(ROOT / "data/processed/phase7_features_v1.csv")
    cfg["datasets"]["labeled"] = str(ROOT / "data/processed/phase8_features_v1.csv")

    cfg["features"]["primary"].append("vertical_ionization_energy_ev")

    new_config = tmp_path / "configs/phase9_ml_v1.yaml"
    new_config.parent.mkdir()
    new_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="Ineligible ML predictor"):
        load_ml_dataset(new_config)
