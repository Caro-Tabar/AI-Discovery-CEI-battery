"""Li+ approach generation and association-energy utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry import Point3D

from cei_scout.preprocessing.xtb_config import (
    XtbLiAssociationSettings,
)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631


class LiApproachMetadata(BaseModel):
    """Provenance for one Li+ starting geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    site_rank: int
    donor_atom_index: int
    donor_element: Literal["O", "N", "S"]
    starting_distance_angstrom: float


@dataclass(frozen=True)
class LiApproachArtifact:
    """One molecular geometry containing an unbonded Li+ starting atom."""

    metadata: LiApproachMetadata
    molecule: Mol


class ConnectivityCheck(BaseModel):
    """Organic connectivity comparison after optimization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    connectivity_preserved: bool
    fragmented: bool
    removed_bond_count: int
    added_bond_count: int


class LiAssociationEnergy(BaseModel):
    """GFN2-xTB electronic Li+ association-energy proxy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    delta_energy_hartree: float
    delta_energy_ev: float
    delta_energy_kcal_mol: float


def read_xyz(
    path: Path,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[float, float, float], ...],
]:
    """Read symbols and coordinates from an XYZ file."""

    lines = path.read_text(
        encoding="utf-8",
    ).splitlines()

    atom_count = int(lines[0])

    atom_lines = lines[2 : 2 + atom_count]

    if len(atom_lines) != atom_count:
        raise ValueError(f"Incomplete XYZ file: {path}")

    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []

    for line in atom_lines:
        fields = line.split()

        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom line: {line!r}")

        symbols.append(fields[0])

        coordinates.append(
            (
                float(fields[1]),
                float(fields[2]),
                float(fields[3]),
            )
        )

    return tuple(symbols), tuple(coordinates)


def molecule_with_xyz_coordinates(
    reference_molecule: Mol,
    xyz_path: Path,
) -> Mol:
    """Transfer xTB-optimized coordinates onto known RDKit connectivity."""

    symbols, coordinates = read_xyz(xyz_path)

    if len(symbols) != reference_molecule.GetNumAtoms():
        raise ValueError("XYZ atom count does not match the reference molecule.")

    expected_symbols = tuple(atom.GetSymbol() for atom in reference_molecule.GetAtoms())

    if symbols != expected_symbols:
        raise ValueError("XYZ atom ordering does not match the reference molecule.")

    molecule = Chem.Mol(reference_molecule)

    molecule.RemoveAllConformers()

    conformer = Chem.Conformer(molecule.GetNumAtoms())

    conformer.Set3D(True)

    for atom_index, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(
            atom_index,
            Point3D(*coordinate),
        )

    molecule.AddConformer(
        conformer,
        assignId=True,
    )

    return molecule


def _distance_for_donor(
    element: str,
    settings: XtbLiAssociationSettings,
) -> float:
    if element == "O":
        return settings.oxygen_distance_angstrom

    if element == "N":
        return settings.nitrogen_distance_angstrom

    if element == "S":
        return settings.sulfur_distance_angstrom

    raise ValueError(f"Unsupported donor element: {element}")


def _is_donor(
    atom: Any,
    settings: XtbLiAssociationSettings,
) -> bool:
    element = atom.GetSymbol()

    if atom.GetFormalCharge() > 0:
        return False

    if element in {"O", "N"}:
        return True

    if element == "S":
        return int(atom.GetTotalValence()) <= settings.sulfur_max_total_valence
    return False


def _unit_outward_vector(
    molecule: Mol,
    donor_index: int,
) -> tuple[float, float, float]:
    conformer = molecule.GetConformer()

    donor = molecule.GetAtomWithIdx(donor_index)

    donor_position = conformer.GetAtomPosition(donor_index)

    neighbors = tuple(donor.GetNeighbors())

    if neighbors:
        center_x = sum(
            conformer.GetAtomPosition(neighbor.GetIdx()).x for neighbor in neighbors
        ) / len(neighbors)

        center_y = sum(
            conformer.GetAtomPosition(neighbor.GetIdx()).y for neighbor in neighbors
        ) / len(neighbors)

        center_z = sum(
            conformer.GetAtomPosition(neighbor.GetIdx()).z for neighbor in neighbors
        ) / len(neighbors)

    else:
        other_indices = tuple(
            index for index in range(molecule.GetNumAtoms()) if index != donor_index
        )

        center_x = sum(
            conformer.GetAtomPosition(index).x for index in other_indices
        ) / len(other_indices)

        center_y = sum(
            conformer.GetAtomPosition(index).y for index in other_indices
        ) / len(other_indices)

        center_z = sum(
            conformer.GetAtomPosition(index).z for index in other_indices
        ) / len(other_indices)

    vector = (
        donor_position.x - center_x,
        donor_position.y - center_y,
        donor_position.z - center_z,
    )

    norm = math.sqrt(sum(component * component for component in vector))

    if norm < 1.0e-10:
        return (1.0, 0.0, 0.0)

    return tuple(component / norm for component in vector)


def generate_li_approaches(
    neutral_molecule: Mol,
    settings: XtbLiAssociationSettings,
) -> tuple[LiApproachArtifact, ...]:
    """Generate one deterministic Li+ approach for each donor site."""

    approaches: list[LiApproachArtifact] = []

    neutral_conformer = neutral_molecule.GetConformer()

    donor_indices = tuple(
        atom.GetIdx()
        for atom in neutral_molecule.GetAtoms()
        if _is_donor(atom, settings)
    )

    for site_rank, donor_index in enumerate(
        donor_indices,
        start=1,
    ):
        donor_atom = neutral_molecule.GetAtomWithIdx(donor_index)

        donor_element = str(donor_atom.GetSymbol())

        if donor_element not in {"O", "N", "S"}:
            raise RuntimeError(f"Unexpected donor element: {donor_element}")

        distance = _distance_for_donor(
            donor_element,
            settings,
        )

        direction = _unit_outward_vector(
            neutral_molecule,
            donor_index,
        )

        donor_position = neutral_conformer.GetAtomPosition(donor_index)

        editable: Any = Chem.RWMol(neutral_molecule)

        lithium = Chem.Atom("Li")
        lithium.SetFormalCharge(1)

        lithium_index = editable.AddAtom(lithium)

        complex_molecule = editable.GetMol()

        complex_molecule.RemoveAllConformers()

        complex_conformer = Chem.Conformer(complex_molecule.GetNumAtoms())

        complex_conformer.Set3D(True)

        for atom_index in range(neutral_molecule.GetNumAtoms()):
            position = neutral_conformer.GetAtomPosition(atom_index)

            complex_conformer.SetAtomPosition(
                atom_index,
                Point3D(
                    position.x,
                    position.y,
                    position.z,
                ),
            )

        complex_conformer.SetAtomPosition(
            lithium_index,
            Point3D(
                donor_position.x + distance * direction[0],
                donor_position.y + distance * direction[1],
                donor_position.z + distance * direction[2],
            ),
        )

        complex_molecule.AddConformer(
            complex_conformer,
            assignId=True,
        )

        approaches.append(
            LiApproachArtifact(
                metadata=LiApproachMetadata(
                    site_rank=site_rank,
                    donor_atom_index=donor_index,
                    donor_element=donor_element,  # type: ignore[arg-type]
                    starting_distance_angstrom=distance,
                ),
                molecule=complex_molecule,
            )
        )

    return tuple(approaches)


def check_organic_connectivity(
    reference_molecule: Mol,
    optimized_xyz: Path,
) -> ConnectivityCheck:
    """Compare optimized organic connectivity with the frozen graph."""

    symbols, coordinates = read_xyz(optimized_xyz)

    reference_count = reference_molecule.GetNumAtoms()

    if len(symbols) == reference_count + 1:
        if symbols[-1] != "Li":
            raise ValueError("Expected Li to be the final atom of the complex XYZ.")

        symbols = symbols[:-1]
        coordinates = coordinates[:-1]

    elif len(symbols) != reference_count:
        raise ValueError("Optimized XYZ has an unexpected atom count.")

    expected_symbols = tuple(atom.GetSymbol() for atom in reference_molecule.GetAtoms())

    if symbols != expected_symbols:
        raise ValueError("Optimized XYZ atom ordering changed.")

    editable: Any = Chem.RWMol()

    for symbol in symbols:
        editable.AddAtom(Chem.Atom(symbol))

    inferred = editable.GetMol()

    conformer = Chem.Conformer(len(symbols))

    conformer.Set3D(True)

    for atom_index, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(
            atom_index,
            Point3D(*coordinate),
        )

    inferred.AddConformer(
        conformer,
        assignId=True,
    )

    inferred_editable: Any = Chem.RWMol(inferred)

    determine: Any = rdDetermineBonds

    determine.DetermineConnectivity(
        inferred_editable,
        False,
        0,
        1.3,
        True,
    )

    inferred = inferred_editable.GetMol()

    reference_edges = {
        tuple(
            sorted(
                (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                )
            )
        )
        for bond in reference_molecule.GetBonds()
    }

    inferred_edges = {
        tuple(
            sorted(
                (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                )
            )
        )
        for bond in inferred.GetBonds()
    }

    removed_edges = reference_edges - inferred_edges

    added_edges = inferred_edges - reference_edges

    fragments = Chem.GetMolFrags(
        inferred,
        asMols=False,
        sanitizeFrags=False,
    )

    fragmented = len(fragments) > 1

    return ConnectivityCheck(
        connectivity_preserved=(
            not removed_edges and not added_edges and not fragmented
        ),
        fragmented=fragmented,
        removed_bond_count=len(removed_edges),
        added_bond_count=len(added_edges),
    )


def compute_li_association_energy(
    *,
    complex_energy_hartree: float,
    neutral_energy_hartree: float,
    lithium_energy_hartree: float,
) -> LiAssociationEnergy:
    """Calculate the electronic Li+ association-energy proxy."""

    delta_energy = (
        complex_energy_hartree - neutral_energy_hartree - lithium_energy_hartree
    )

    return LiAssociationEnergy(
        delta_energy_hartree=delta_energy,
        delta_energy_ev=(delta_energy * HARTREE_TO_EV),
        delta_energy_kcal_mol=(delta_energy * HARTREE_TO_KCAL_MOL),
    )


def make_li_reference_molecule() -> Mol:
    """Create a one-atom Li+ reference geometry."""

    molecule = Chem.MolFromSmiles("[Li+]")

    if molecule is None:
        raise RuntimeError("RDKit could not construct Li+.")

    molecule.RemoveAllConformers()

    conformer = Chem.Conformer(1)

    conformer.Set3D(True)

    conformer.SetAtomPosition(
        0,
        Point3D(0.0, 0.0, 0.0),
    )

    molecule.AddConformer(
        conformer,
        assignId=True,
    )

    return molecule
