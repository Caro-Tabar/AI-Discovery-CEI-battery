"""Load Phase 9 data while preserving candidate identity and DFT exclusions."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class MLDataset:
    """Ordered molecular predictors, targets, and validation groups."""

    candidate_ids: tuple[str, ...]
    families: tuple[str, ...]
    feature_names: tuple[str, ...]
    X: np.ndarray
    targets: dict[str, np.ndarray]


def _read_rows(path: Path, id_column: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise ValueError(f"Missing {id_column!r} column in {path}")

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            candidate_id = row.get(id_column)
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(f"Blank candidate ID in {path}")

            if any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in row.items()
            ):
                raise ValueError(f"Malformed CSV row for {candidate_id}")

            if candidate_id in rows:
                raise ValueError(f"Duplicate candidate ID: {candidate_id}")

            rows[candidate_id] = {
                key: value
                for key, value in row.items()
                if isinstance(key, str) and isinstance(value, str)
            }

        return rows


def _number(row: dict[str, str], column: str, candidate_id: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing/non-numeric {column} for {candidate_id}") from exc

    if not math.isfinite(value):
        raise ValueError(f"Non-finite {column} for {candidate_id}")

    return value


def _check_feature_policy(
    features: tuple[str, ...],
    targets: tuple[str, ...],
) -> None:
    if not features or len(set(features)) != len(features):
        raise ValueError("Predictors must be a nonempty, duplicate-free whitelist")

    for name in features:
        if (
            name in targets
            or name.startswith(("qc_", "vertical_", "energy_", "runtime_"))
            or "total_energy" in name
            or "complex_energy" in name
            or name in {"xtb_selection_rank", "xtb_attempt_number"}
        ):
            raise ValueError(f"Ineligible ML predictor: {name}")


def load_ml_dataset(config_path: str | Path) -> MLDataset:
    """Load ML inputs and enforce frozen-pool integrity."""
    config_path = Path(config_path).resolve()

    with config_path.open(encoding="utf-8") as handle:
        cfg: dict[str, Any] = yaml.safe_load(handle)

    root = config_path.parent.parent

    def source(key: str) -> Path:
        path = Path(cfg["datasets"][key])
        return path if path.is_absolute() else root / path

    id_col = cfg["datasets"]["candidate_id_column"]
    family_col = cfg["datasets"]["family_column"]

    prospective = _read_rows(source("prospective"), id_col)
    labeled = _read_rows(source("labeled"), id_col)

    population = cfg["population"]
    excluded = set(prospective) - set(labeled)

    if (
        len(prospective) != population["frozen_pool_size"]
        or len(labeled) != population["dft_complete_size"]
        or not set(labeled).issubset(prospective)
        or excluded != set(population["dft_incomplete_ids"])
    ):
        raise ValueError("Candidate population or DFT-exclusion policy mismatch")

    features = tuple(cfg["features"]["primary"])
    target_map: dict[str, str] = cfg["targets"]

    if not target_map or len(set(target_map.values())) != len(target_map):
        raise ValueError("Targets must be unique and nonempty")

    _check_feature_policy(features, tuple(target_map.values()))

    ids = tuple(sorted(labeled))
    families: list[str] = []
    X: list[list[float]] = []
    y: dict[str, list[float]] = {name: [] for name in target_map}

    smiles_seen: set[str] = set()

    # Check the entire prospective population, including DFT exclusions.
    for candidate_id in sorted(prospective):
        row = prospective[candidate_id]
        smiles = row.get("canonical_smiles")

        if not smiles or smiles in smiles_seen:
            raise ValueError(f"Missing/duplicate canonical SMILES: {candidate_id}")

        smiles_seen.add(smiles)

        for feature in features:
            _number(row, feature, candidate_id)

    # Only DFT-complete candidates receive supervised targets.
    for candidate_id in ids:
        current = labeled[candidate_id]
        reference = prospective[candidate_id]

        if any(
            current.get(column) != reference.get(column)
            for column in ("canonical_smiles", family_col)
        ):
            raise ValueError(f"Identity/family drift: {candidate_id}")

        family = current.get(family_col)
        if not family:
            raise ValueError(f"Missing chemical family: {candidate_id}")

        families.append(family)
        X.append([])

        for feature in features:
            actual = _number(current, feature, candidate_id)
            expected = _number(reference, feature, candidate_id)

            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Descriptor drift: {candidate_id} / {feature}")

            X[-1].append(actual)

        for name, column in target_map.items():
            y[name].append(_number(current, column, candidate_id))

    return MLDataset(
        candidate_ids=ids,
        families=tuple(families),
        feature_names=features,
        X=np.asarray(X, dtype=np.float64),
        targets={
            name: np.asarray(values, dtype=np.float64) for name, values in y.items()
        },
    )
