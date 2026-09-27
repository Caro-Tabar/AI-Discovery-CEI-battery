"""Tests for force-field conformer prescreening."""

from __future__ import annotations

import math

import pytest

from cei_scout.preprocessing.conformers import (
    ConformerSettings,
    generate_conformers,
)
from cei_scout.preprocessing.forcefield import (
    ForceFieldSettings,
    prescreen_conformers,
)

BENCHMARK_SMILES = {
    "EC": "C1COC(=O)O1",
    "EMC": "CCOC(=O)OC",
    "DTD": "C1COS(=O)(=O)O1",
    "sulfolane": "C1CCS(=O)(=O)C1",
}


def _conformer_settings() -> ConformerSettings:
    return ConformerSettings(
        random_seed=20260925,
        num_conformers=10,
        prune_rms_threshold_angstrom=0.5,
        num_threads=1,
        enforce_chirality=True,
        use_small_ring_torsions=True,
        use_macrocycle_torsions=True,
        use_macrocycle_14config=True,
        only_heavy_atoms_for_pruning=True,
    )


def _force_field_settings() -> ForceFieldSettings:
    return ForceFieldSettings(
        primary_force_field="MMFF94s",
        fallback_force_field="UFF",
        fallback_if_mmff_unavailable=True,
        max_iterations=500,
        num_threads=1,
    )


@pytest.mark.parametrize(
    ("name", "smiles"),
    BENCHMARK_SMILES.items(),
)
def test_benchmark_molecules_complete_force_field_prescreen(
    name: str,
    smiles: str,
) -> None:
    molecule, conformer_ids = generate_conformers(
        smiles,
        _conformer_settings(),
    )

    results = prescreen_conformers(
        molecule,
        _force_field_settings(),
    )

    assert results, name
    assert len(results) == len(conformer_ids)
    assert any(result.converged for result in results)

    assert all(result.force_field in {"MMFF94s", "UFF"} for result in results)

    assert all(math.isfinite(result.energy_kcal_mol) for result in results)

    converged_results = tuple(result for result in results if result.converged)

    assert tuple(result.energy_kcal_mol for result in converged_results) == tuple(
        sorted(result.energy_kcal_mol for result in converged_results)
    )
