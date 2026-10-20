"""Tests for Li+ association preprocessing."""

from cei_scout.preprocessing.conformers import (
    ConformerSettings,
    generate_conformers,
)
from cei_scout.preprocessing.li_complexes import (
    compute_li_association_energy,
    generate_li_approaches,
)
from cei_scout.preprocessing.xtb_config import (
    XtbLiAssociationSettings,
)


def _settings() -> XtbLiAssociationSettings:
    return XtbLiAssociationSettings(
        charge=1,
        uhf=0,
        oxygen_distance_angstrom=2.0,
        nitrogen_distance_angstrom=2.1,
        sulfur_distance_angstrom=2.3,
        sulfur_max_total_valence=2,
    )


def test_low_valent_sulfur_is_li_donor() -> None:
    molecule, _ = generate_conformers(
        "CSC",
        ConformerSettings(
            random_seed=20260925,
            num_conformers=5,
            prune_rms_threshold_angstrom=0.5,
            num_threads=1,
            enforce_chirality=True,
            use_small_ring_torsions=True,
            use_macrocycle_torsions=True,
            use_macrocycle_14config=True,
            only_heavy_atoms_for_pruning=True,
        ),
    )

    approaches = generate_li_approaches(
        molecule,
        _settings(),
    )

    assert len(approaches) == 1
    assert approaches[0].metadata.donor_element == "S"


def test_li_association_energy_sign() -> None:
    result = compute_li_association_energy(
        complex_energy_hartree=-11.0,
        neutral_energy_hartree=-8.0,
        lithium_energy_hartree=-2.9,
    )

    assert result.delta_energy_hartree < 0
    assert result.delta_energy_ev < 0
    assert result.delta_energy_kcal_mol < 0
