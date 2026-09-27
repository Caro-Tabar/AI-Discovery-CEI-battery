"""Tests for the frozen Phase 7 feature dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cei_scout.preprocessing.conformers import (
    ConformerSettings,
    generate_conformers,
)
from cei_scout.preprocessing.li_complexes import (
    check_organic_connectivity,
    compute_li_association_energy,
)
from cei_scout.preprocessing.phase7_features import (
    build_phase7_feature_table,
    compute_rdkit_descriptor_row,
)


def test_rdkit_descriptor_element_counts() -> None:
    result = compute_rdkit_descriptor_row(
        "TEST",
        "CSC",
    )

    assert result["rdkit_heavy_atom_count"] == 3

    assert result["rdkit_c_count"] == 2

    assert result["rdkit_h_count"] == 6

    assert result["rdkit_s_count"] == 1


def test_li_association_energy_units() -> None:
    result = compute_li_association_energy(
        complex_energy_hartree=-11.0,
        neutral_energy_hartree=-8.0,
        lithium_energy_hartree=-2.9,
    )

    assert result.delta_energy_hartree == pytest.approx(-0.1)

    assert result.delta_energy_ev == pytest.approx(-2.7211386245988)

    assert result.delta_energy_kcal_mol == pytest.approx(-62.75094740631)


def _write_xyz(
    path: Path,
    molecule: object,
    conformer_id: int,
) -> None:
    rdkit_molecule = molecule

    conformer = rdkit_molecule.GetConformer(conformer_id)

    lines = [
        str(rdkit_molecule.GetNumAtoms()),
        "connectivity test",
    ]

    for atom_index, atom in enumerate(rdkit_molecule.GetAtoms()):
        position = conformer.GetAtomPosition(atom_index)

        lines.append(
            f"{atom.GetSymbol()} {position.x:.12f} {position.y:.12f} {position.z:.12f}"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def test_connectivity_check_preserves_reference(
    tmp_path: Path,
) -> None:
    molecule, conformer_ids = generate_conformers(
        "CO",
        ConformerSettings(
            random_seed=20260925,
            num_conformers=1,
            prune_rms_threshold_angstrom=0.5,
            num_threads=1,
            enforce_chirality=True,
            use_small_ring_torsions=True,
            use_macrocycle_torsions=True,
            use_macrocycle_14config=True,
            only_heavy_atoms_for_pruning=True,
        ),
    )

    xyz_path = tmp_path / "methanol.xyz"

    _write_xyz(
        xyz_path,
        molecule,
        conformer_ids[0],
    )

    result = check_organic_connectivity(
        molecule,
        xyz_path,
    )

    assert result.connectivity_preserved
    assert not result.fragmented
    assert result.removed_bond_count == 0
    assert result.added_bond_count == 0


def test_phase7_feature_table_is_complete() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "CEI-TEST",
                "preferred_name": "Test",
                "canonical_smiles": "CSC",
                "molecular_formula": "C2H6S",
                "molecular_weight_g_mol": 62.14,
                "candidate_family": ("thioether_or_disulfide"),
                "source_entry_id": "source-1",
                "source_url": "https://example.com",
                "registry_version": "v1",
                "selection_status": "selected",
            }
        ]
    )

    xtb = pd.DataFrame(
        [
            {
                "candidate_id": "CEI-TEST",
                "candidate_family": ("thioether_or_disulfide"),
                "source_entry_id": "source-1",
                "registry_version": "v1",
                "selection_rank": 1,
                "attempt_number": 1,
                "method": "GFN2-xTB",
                "gfn_level": 2,
                "solvation_model": "none",
                "total_energy_hartree": -10.0,
                "homo_ev": -10.0,
                "lumo_ev": -4.0,
                "homo_lumo_gap_ev": 6.0,
                "dipole_debye": 2.0,
                "polarizability_au": 50.0,
                "total_charge_e": 0.0,
            }
        ]
    )

    li_association = pd.DataFrame(
        [
            {
                "candidate_id": "CEI-TEST",
                "candidate_family": ("thioether_or_disulfide"),
                "source_entry_id": "source-1",
                "registry_version": "v1",
                "best_site_rank": 1,
                "best_donor_atom_index": 1,
                "best_donor_element": "S",
                "neutral_energy_hartree": -10.0,
                "li_reference_energy_hartree": -2.0,
                "complex_energy_hartree": -12.1,
                "delta_energy_hartree": -0.1,
                "delta_energy_ev": -2.7211386245988,
                "delta_energy_kcal_mol": -62.75094740631,
            }
        ]
    )

    li_sites = pd.DataFrame(
        [
            {
                "candidate_id": "CEI-TEST",
                "site_rank": 1,
                "status": "success",
                "connectivity_preserved": True,
                "fragmented": False,
            }
        ]
    )

    features = build_phase7_feature_table(
        candidates=candidates,
        xtb=xtb,
        li_association=li_association,
        li_sites=li_sites,
        expected_count=1,
    )

    assert len(features) == 1

    assert int(features.isna().sum().sum()) == 0

    assert (
        features.loc[
            0,
            "qc_li_connectivity_change_count",
        ]
        == 0
    )
