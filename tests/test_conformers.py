"""Tests for deterministic RDKit conformer generation."""

from __future__ import annotations

import pytest
from rdkit.Chem.rdchem import Mol

from cei_scout.preprocessing.conformers import (
    ConformerSettings,
    generate_conformers,
)

EC_SMILES = "O=C1OCCO1"


def _coordinates(
    molecule: Mol,
    conformer_id: int,
) -> tuple[float, ...]:
    """Return flattened coordinates for one conformer."""

    conformer = molecule.GetConformer(conformer_id)

    return tuple(
        coordinate
        for atom_index in range(molecule.GetNumAtoms())
        for coordinate in (
            conformer.GetAtomPosition(atom_index).x,
            conformer.GetAtomPosition(atom_index).y,
            conformer.GetAtomPosition(atom_index).z,
        )
    )


def _test_settings() -> ConformerSettings:
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


def test_generate_conformers_is_deterministic() -> None:
    first_molecule, first_ids = generate_conformers(
        EC_SMILES,
        _test_settings(),
    )

    second_molecule, second_ids = generate_conformers(
        EC_SMILES,
        _test_settings(),
    )

    assert first_ids == second_ids
    assert first_ids

    for first_id, second_id in zip(
        first_ids,
        second_ids,
        strict=True,
    ):
        assert _coordinates(
            first_molecule,
            first_id,
        ) == pytest.approx(
            _coordinates(second_molecule, second_id),
            abs=1.0e-8,
        )


def test_generate_conformers_adds_explicit_hydrogens() -> None:
    molecule, conformer_ids = generate_conformers(
        EC_SMILES,
        _test_settings(),
    )

    assert conformer_ids
    assert any(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms())


def test_generate_conformers_rejects_invalid_smiles() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        generate_conformers(
            "not-a-valid-smiles",
            _test_settings(),
        )
