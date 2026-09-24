"""Validation tests for CEI-Scout data contracts."""

from typing import Any

import pytest
from pydantic import ValidationError

from cei_scout.schemas import (
    AcquisitionRecord,
    ActiveLearningRoundRecord,
    CalculationRecord,
    CandidateRecord,
    MDObservableRecord,
    MDReplicaRecord,
    MDSystemRecord,
    ProvenanceRecord,
    SchemaModel,
)

PROVENANCE = {
    "provenance_id": "provenance-001",
    "source_name": "CEI-Scout",
    "source_url": ("https://github.com/Caro-Tabar/AI-Discovery-CEI-battery"),
    "license_id": "MIT",
    "retrieval_date": "2026-09-14",
    "source_version": "a8358e4",
    "sha256": "0" * 64,
    "transformation": "Original project record.",
}

CANDIDATE = {
    "provenance_id": "provenance-001",
    "candidate_id": "candidate-001",
    "name": "Example candidate",
    "candidate_family": "sulfite",
    "eligibility_status": "accepted",
}

CALCULATION = {
    "provenance_id": "provenance-001",
    "calculation_id": "calculation-001",
    "candidate_id": "candidate-001",
    "engine": "pyscf",
    "job_type": "neutral_single_point",
    "method": "wB97X-V",
    "basis_set": "def2-SVPD",
    "solvent_model": "IEF-PCM",
    "dielectric": 20.0,
    "charge": 0,
    "spin_multiplicity": 1,
    "input_geometry_path": "geometry.xyz",
    "status": "succeeded",
    "electronic_energy_hartree": -100.0,
}

ROUND = {
    "provenance_id": "provenance-001",
    "round_index": 0,
    "observed_candidate_ids": ("candidate-001",),
    "selected_candidate_ids": ("candidate-002",),
    "batch_size": 1,
    "random_seed": 17,
    "reference_point": (-1.0, -1.0),
    "model_version": "unfit",
}

ACQUISITION = {
    "provenance_id": "provenance-001",
    "round_index": 0,
    "candidate_id": "candidate-002",
    "oxidation_selectivity_mean_ev": 0.30,
    "oxidation_selectivity_std_ev": 0.10,
    "li_coordination_objective_mean_ev": -0.10,
    "li_coordination_objective_std_ev": 0.05,
    "acquisition_value": 0.20,
    "acquisition_rank": 1,
    "selected": True,
}

MD_SYSTEM = {
    "provenance_id": "provenance-001",
    "system_id": "baseline",
    "system_role": "baseline",
    "temperature_k": 298.15,
    "pressure_bar": 1.0,
    "force_field_family": "openff-sage",
}

MD_REPLICA = {
    "provenance_id": "provenance-001",
    "replica_id": "baseline-r1",
    "system_id": "baseline",
    "replica_index": 1,
    "random_seed": 17,
    "equilibration_time_ns": 1.0,
    "production_time_ns": 5.0,
}

MD_OBSERVABLE = {
    "provenance_id": "provenance-001",
    "observable_id": "observable-001",
    "system_id": "baseline",
    "replica_id": "baseline-r1",
    "observable_name": "li_self_diffusion",
    "value": 1.0e-6,
    "unit": "cm^2/s",
    "uncertainty": 1.0e-7,
    "uncertainty_method": "block analysis",
    "analysis_method": "mean squared displacement",
}

VALID_FIXTURES: list[tuple[type[SchemaModel], dict[str, Any]]] = [
    (ProvenanceRecord, PROVENANCE),
    (CandidateRecord, CANDIDATE),
    (CalculationRecord, CALCULATION),
    (ActiveLearningRoundRecord, ROUND),
    (AcquisitionRecord, ACQUISITION),
    (MDSystemRecord, MD_SYSTEM),
    (MDReplicaRecord, MD_REPLICA),
    (MDObservableRecord, MD_OBSERVABLE),
]

INVALID_FIXTURES: list[tuple[type[SchemaModel], dict[str, Any]]] = [
    (
        ProvenanceRecord,
        {
            **PROVENANCE,
            "source_url": None,
            "source_doi": None,
        },
    ),
    (
        CandidateRecord,
        {
            **CANDIDATE,
            "eligibility_status": "rejected",
            "rejection_reasons": (),
        },
    ),
    (
        CalculationRecord,
        {
            **CALCULATION,
            "electronic_energy_hartree": None,
        },
    ),
    (
        ActiveLearningRoundRecord,
        {
            **ROUND,
            "selected_candidate_ids": ("candidate-001",),
        },
    ),
    (
        AcquisitionRecord,
        {
            **ACQUISITION,
            "oxidation_selectivity_std_ev": -0.1,
        },
    ),
    (
        MDSystemRecord,
        {
            **MD_SYSTEM,
            "additive_candidate_id": "candidate-001",
            "additive_mass_fraction": 0.05,
        },
    ),
    (
        MDReplicaRecord,
        {
            **MD_REPLICA,
            "production_time_ns": -1.0,
        },
    ),
    (
        MDObservableRecord,
        {
            **MD_OBSERVABLE,
            "unit": "eV",
        },
    ),
]


@pytest.mark.parametrize(
    ("model", "data"),
    VALID_FIXTURES,
    ids=[
        "provenance",
        "candidate",
        "calculation",
        "active-learning-round",
        "acquisition",
        "md-system",
        "md-replica",
        "md-observable",
    ],
)
def test_valid_fixture(
    model: type[SchemaModel],
    data: dict[str, Any],
) -> None:
    """Each schema accepts one representative valid record."""
    record = model.model_validate(data)
    assert record.model_dump()


@pytest.mark.parametrize(
    ("model", "data"),
    INVALID_FIXTURES,
    ids=[
        "provenance-missing-locator",
        "candidate-missing-rejection-reason",
        "calculation-missing-energy",
        "active-learning-label-leakage",
        "acquisition-negative-uncertainty",
        "md-baseline-with-additive",
        "md-replica-negative-time",
        "md-observable-wrong-unit",
    ],
)
def test_invalid_fixture(
    model: type[SchemaModel],
    data: dict[str, Any],
) -> None:
    """Each schema rejects one targeted invalid record."""
    with pytest.raises(ValidationError):
        model.model_validate(data)
