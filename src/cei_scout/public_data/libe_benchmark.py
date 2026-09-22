"""Construction of the retrospective LIBE benchmark."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

ALLOWED_ELEMENTS = frozenset({"C", "H", "N", "O", "S"})

MIN_MASS_G_MOL = 70.0
MAX_MASS_G_MOL = 220.0

HARTREE_TO_EV = 27.211386245988

ATOMIC_WEIGHTS_G_MOL = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
}


def molecular_weight_g_mol(record: dict[str, Any]) -> float:
    """Calculate molecular weight from the LIBE composition."""
    composition = record.get("composition")

    if not isinstance(composition, dict):
        raise ValueError("LIBE composition must be a JSON object.")

    mass = 0.0

    for element, count in composition.items():
        if element not in ATOMIC_WEIGHTS_G_MOL:
            raise ValueError(f"Unsupported element for benchmark mass: {element}")

        if not isinstance(count, (int, float)):
            raise ValueError(f"Invalid atom count for {element}: {count!r}")

        mass += ATOMIC_WEIGHTS_G_MOL[element] * float(count)

    return mass


def electronic_energy_ha(record: dict[str, Any]) -> float:
    """Return LIBE's raw Q-Chem electronic energy in Hartree."""
    thermo = record.get("thermo")

    if not isinstance(thermo, dict):
        raise ValueError("Missing LIBE thermo object.")

    raw = thermo.get("raw")

    if not isinstance(raw, dict):
        raise ValueError("Missing LIBE thermo.raw object.")

    value = raw.get("electronic_energy_Ha")

    if not isinstance(value, (int, float)):
        raise ValueError("Missing LIBE thermo.raw.electronic_energy_Ha.")

    return float(value)


def graph_from_record(
    record: dict[str, Any],
    *,
    heavy_atoms_only: bool = False,
) -> nx.Graph:
    """Build an element-labelled molecular connectivity graph."""
    species = record.get("species")
    bonds = record.get("bonds")

    if not isinstance(species, list):
        raise ValueError("LIBE species must be a list.")

    if bonds is None:
        bonds = []

    if not isinstance(bonds, list):
        raise ValueError("LIBE bonds must be a list.")

    graph: nx.Graph[int] = nx.Graph()

    for atom_index, element in enumerate(species):
        if not isinstance(element, str):
            raise ValueError("LIBE species entries must be strings.")

        if heavy_atoms_only and element == "H":
            continue

        graph.add_node(atom_index, element=element)

    for bond in bonds:
        if not (
            isinstance(bond, list)
            and len(bond) == 2
            and all(isinstance(index, int) for index in bond)
        ):
            raise ValueError(f"Invalid LIBE bond: {bond!r}")

        atom_a, atom_b = bond

        if atom_a in graph and atom_b in graph:
            graph.add_edge(atom_a, atom_b)

    return graph


def records_are_isomorphic(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    """Compare molecular connectivity while respecting element identity."""
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "element",
        "",
    )

    return nx.is_isomorphic(
        graph_from_record(first),
        graph_from_record(second),
        node_match=node_match,
    )


def neutral_parent_rejection_reason(
    record: dict[str, Any],
) -> str | None:
    """Return the first eligibility failure for a neutral parent."""
    if record.get("charge") != 0:
        return "not_neutral"

    if record.get("spin_multiplicity") != 1:
        return "not_closed_shell"

    elements = record.get("elements")

    if not isinstance(elements, list):
        return "invalid_elements"

    element_set = set(elements)

    if not element_set.issubset(ALLOWED_ELEMENTS):
        return "disallowed_element"

    try:
        mass = molecular_weight_g_mol(record)
    except ValueError:
        return "invalid_mass"

    if not MIN_MASS_G_MOL <= mass <= MAX_MASS_G_MOL:
        return "mass_out_of_range"

    return None


def _charge_index(
    records: list[dict[str, Any]],
) -> dict[tuple[str, int, int, int], list[dict[str, Any]]]:
    """Bucket records before exact graph-isomorphism matching."""
    index: dict[
        tuple[str, int, int, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        formula = record.get("formula_alphabetical")
        bonds = record.get("bonds") or []
        charge = record.get("charge")
        multiplicity = record.get("spin_multiplicity")

        if (
            isinstance(formula, str)
            and isinstance(bonds, list)
            and isinstance(charge, int)
            and isinstance(multiplicity, int)
        ):
            index[
                (
                    formula,
                    len(bonds),
                    charge,
                    multiplicity,
                )
            ].append(record)

    return dict(index)


def matching_charge_state(
    neutral: dict[str, Any],
    index: dict[
        tuple[str, int, int, int],
        list[dict[str, Any]],
    ],
    charge: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Match one-electron charged states by exact graph isomorphism."""
    if charge not in {-1, 1}:
        raise ValueError("Charge-state match must use -1 or +1.")

    formula = neutral["formula_alphabetical"]
    bonds = neutral.get("bonds") or []

    key = (
        formula,
        len(bonds),
        charge,
        2,
    )

    candidates = index.get(key, [])

    matches = [
        candidate
        for candidate in candidates
        if records_are_isomorphic(neutral, candidate)
    ]

    if not matches:
        if charge == 1:
            return None, "missing_cation_match"
        return None, "missing_anion_match"

    if len(matches) > 1:
        if charge == 1:
            return None, "ambiguous_cation_match"
        return None, "ambiguous_anion_match"

    return matches[0], None


def build_redox_record(
    neutral: dict[str, Any],
    cation: dict[str, Any],
    anion: dict[str, Any],
) -> dict[str, Any]:
    """Create one fully labelled retrospective benchmark record."""
    neutral_energy = electronic_energy_ha(neutral)
    cation_energy = electronic_energy_ha(cation)
    anion_energy = electronic_energy_ha(anion)

    ionization_energy_ev = (cation_energy - neutral_energy) * HARTREE_TO_EV

    electron_affinity_ev = (neutral_energy - anion_energy) * HARTREE_TO_EV

    return {
        "neutral_id": neutral["molecule_id"],
        "cation_id": cation["molecule_id"],
        "anion_id": anion["molecule_id"],
        "formula": neutral["formula_alphabetical"],
        "chemical_system": neutral["chemical_system"],
        "molecular_weight_g_mol": molecular_weight_g_mol(neutral),
        "neutral_energy_ha": neutral_energy,
        "cation_energy_ha": cation_energy,
        "anion_energy_ha": anion_energy,
        "adiabatic_ionization_energy_ev": ionization_energy_ev,
        "adiabatic_electron_affinity_ev": electron_affinity_ev,
        "oxidation_ease_ev": -ionization_energy_ev,
        "reduction_resistance_ev": -electron_affinity_ev,
    }


def scaffold_hash(record: dict[str, Any]) -> str:
    """Return an element-labelled heavy-atom connectivity hash."""
    graph = graph_from_record(
        record,
        heavy_atoms_only=True,
    )

    return nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr="element",
    )


def select_benchmark(
    records: list[dict[str, Any]],
    neutral_lookup: dict[str, dict[str, Any]],
    *,
    target_size: int = 300,
) -> list[dict[str, Any]]:
    """Select a deterministic family/scaffold-diverse benchmark."""
    if len(records) <= target_size:
        return sorted(
            records,
            key=lambda row: str(row["neutral_id"]),
        )

    strata: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in records:
        neutral_id = str(row["neutral_id"])
        neutral = neutral_lookup[neutral_id]

        family = str(row["chemical_system"])
        scaffold = scaffold_hash(neutral)

        row["family_stratum"] = family
        row["scaffold_hash"] = scaffold

        strata[(family, scaffold)].append(row)

    for bucket in strata.values():
        bucket.sort(key=lambda row: str(row["neutral_id"]))

    selected: list[dict[str, Any]] = []
    ordered_strata = sorted(strata)

    while len(selected) < target_size:
        added = False

        for stratum in ordered_strata:
            bucket = strata[stratum]

            if not bucket:
                continue

            selected.append(bucket.pop(0))
            added = True

            if len(selected) == target_size:
                break

        if not added:
            break

    return selected
