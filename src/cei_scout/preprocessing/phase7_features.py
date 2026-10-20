"""Build versioned Phase 7 molecular features and geometry provenance."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import Mol


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    """Require a set of columns in a dataframe."""

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing)}")


def _selected_candidates(
    candidates: pd.DataFrame,
    expected_count: int,
) -> pd.DataFrame:
    """Return the frozen selected candidate pool."""

    _require_columns(
        candidates,
        {
            "candidate_id",
            "preferred_name",
            "canonical_smiles",
            "molecular_formula",
            "molecular_weight_g_mol",
            "candidate_family",
            "source_entry_id",
            "source_url",
            "registry_version",
            "selection_status",
        },
        "candidate table",
    )

    selected = candidates.loc[candidates["selection_status"] == "selected"].copy()

    if len(selected) != expected_count:
        raise ValueError(
            f"Expected {expected_count} selected candidates, found {len(selected)}."
        )

    if selected["candidate_id"].duplicated().any():
        raise ValueError("Selected candidate IDs must be unique.")

    return selected.sort_values(
        "candidate_id",
        kind="stable",
    ).reset_index(drop=True)


def compute_rdkit_descriptor_row(
    candidate_id: str,
    smiles: str,
) -> dict[str, Any]:
    """Compute the preregistered 2D RDKit descriptors."""

    molecule: Mol | None = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"RDKit could not parse candidate {candidate_id}: {smiles!r}")

    descriptor_api: Any = Descriptors

    molecule_with_h = Chem.AddHs(molecule)

    element_counts = {
        symbol: sum(atom.GetSymbol() == symbol for atom in molecule_with_h.GetAtoms())
        for symbol in ("C", "H", "N", "O", "S")
    }

    return {
        "candidate_id": candidate_id,
        "rdkit_mol_wt_g_mol": float(descriptor_api.MolWt(molecule)),
        "rdkit_exact_mol_wt_g_mol": float(descriptor_api.ExactMolWt(molecule)),
        "rdkit_heavy_atom_count": int(descriptor_api.HeavyAtomCount(molecule)),
        "rdkit_rotatable_bond_count": int(descriptor_api.NumRotatableBonds(molecule)),
        "rdkit_tpsa_a2": float(descriptor_api.TPSA(molecule)),
        "rdkit_hbond_acceptors": int(descriptor_api.NumHAcceptors(molecule)),
        "rdkit_hbond_donors": int(descriptor_api.NumHDonors(molecule)),
        "rdkit_heteroatom_count": int(descriptor_api.NumHeteroatoms(molecule)),
        "rdkit_ring_count": int(descriptor_api.RingCount(molecule)),
        "rdkit_fraction_csp3": float(descriptor_api.FractionCSP3(molecule)),
        "rdkit_c_count": int(element_counts["C"]),
        "rdkit_h_count": int(element_counts["H"]),
        "rdkit_n_count": int(element_counts["N"]),
        "rdkit_o_count": int(element_counts["O"]),
        "rdkit_s_count": int(element_counts["S"]),
    }


def build_rdkit_descriptor_table(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate descriptors for all frozen candidates."""

    rows = [
        compute_rdkit_descriptor_row(
            candidate_id=str(record["candidate_id"]),
            smiles=str(record["canonical_smiles"]),
        )
        for record in candidates[
            [
                "candidate_id",
                "canonical_smiles",
            ]
        ].to_dict(orient="records")
    ]

    return pd.DataFrame(rows)


def _validate_exact_candidate_ids(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    table_name: str,
) -> None:
    """Require two tables to contain the same unique candidate IDs."""

    if comparison["candidate_id"].duplicated().any():
        raise ValueError(f"{table_name} contains duplicate candidate IDs.")

    expected_ids = set(reference["candidate_id"].astype(str))

    actual_ids = set(comparison["candidate_id"].astype(str))

    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)

        extra = sorted(actual_ids - expected_ids)

        raise ValueError(
            f"{table_name} candidate mismatch; missing={missing}, extra={extra}"
        )


def _validate_metadata_consistency(
    candidates: pd.DataFrame,
    table: pd.DataFrame,
    table_name: str,
) -> None:
    """Require identity/provenance fields to match candidate registry."""

    for column in (
        "candidate_family",
        "source_entry_id",
        "registry_version",
    ):
        reference = (
            candidates.set_index("candidate_id")[column].astype(str).sort_index()
        )

        comparison = table.set_index("candidate_id")[column].astype(str).sort_index()

        if not reference.equals(comparison):
            raise ValueError(
                f"{table_name} disagrees with candidate registry for {column}."
            )


def _build_site_qc(
    li_sites: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate Li-site execution/connectivity QC by candidate."""

    rows: list[dict[str, Any]] = []

    for candidate_id, group in li_sites.groupby(
        "candidate_id",
        sort=True,
    ):
        success = group["status"].astype(str).eq("success")

        connectivity_preserved = (
            group["connectivity_preserved"].fillna(False).astype(bool)
        )

        fragmented = group["fragmented"].fillna(False).astype(bool)

        rows.append(
            {
                "candidate_id": str(candidate_id),
                "qc_li_site_count": len(group),
                "qc_li_site_success_count": int(success.sum()),
                "qc_li_connectivity_change_count": int((~connectivity_preserved).sum()),
                "qc_li_fragmentation_count": int(fragmented.sum()),
            }
        )

    return pd.DataFrame(rows)


def build_phase7_feature_table(
    *,
    candidates: pd.DataFrame,
    xtb: pd.DataFrame,
    li_association: pd.DataFrame,
    li_sites: pd.DataFrame,
    expected_count: int = 60,
) -> pd.DataFrame:
    """Merge frozen identity, structural, xTB, and Li+ features."""

    selected = _selected_candidates(
        candidates,
        expected_count,
    )

    _require_columns(
        xtb,
        {
            "candidate_id",
            "candidate_family",
            "source_entry_id",
            "registry_version",
            "selection_rank",
            "attempt_number",
            "method",
            "gfn_level",
            "solvation_model",
            "total_energy_hartree",
            "homo_ev",
            "lumo_ev",
            "homo_lumo_gap_ev",
            "dipole_debye",
            "polarizability_au",
            "total_charge_e",
        },
        "neutral xTB table",
    )

    _require_columns(
        li_association,
        {
            "candidate_id",
            "candidate_family",
            "source_entry_id",
            "registry_version",
            "best_site_rank",
            "best_donor_atom_index",
            "best_donor_element",
            "neutral_energy_hartree",
            "li_reference_energy_hartree",
            "complex_energy_hartree",
            "delta_energy_hartree",
            "delta_energy_ev",
            "delta_energy_kcal_mol",
        },
        "Li association table",
    )

    _require_columns(
        li_sites,
        {
            "candidate_id",
            "site_rank",
            "status",
            "connectivity_preserved",
            "fragmented",
        },
        "Li site table",
    )

    _validate_exact_candidate_ids(
        selected,
        xtb,
        "neutral xTB table",
    )

    _validate_exact_candidate_ids(
        selected,
        li_association,
        "Li association table",
    )

    _validate_metadata_consistency(
        selected,
        xtb,
        "neutral xTB table",
    )

    _validate_metadata_consistency(
        selected,
        li_association,
        "Li association table",
    )

    energy_check = xtb[
        [
            "candidate_id",
            "total_energy_hartree",
        ]
    ].merge(
        li_association[
            [
                "candidate_id",
                "neutral_energy_hartree",
            ]
        ],
        on="candidate_id",
        validate="one_to_one",
    )

    maximum_energy_difference = float(
        (energy_check["total_energy_hartree"] - energy_check["neutral_energy_hartree"])
        .abs()
        .max()
    )

    if maximum_energy_difference > 1.0e-8:
        raise ValueError(
            "Li association table does not reference the frozen neutral xTB energies."
        )

    rdkit_descriptors = build_rdkit_descriptor_table(selected)

    base_columns = [
        "candidate_id",
        "preferred_name",
        "canonical_smiles",
        "molecular_formula",
        "molecular_weight_g_mol",
        "candidate_family",
        "source_entry_id",
        "source_url",
        "registry_version",
        "selection_status",
    ]

    xtb_features = xtb[
        [
            "candidate_id",
            "selection_rank",
            "attempt_number",
            "method",
            "gfn_level",
            "solvation_model",
            "total_energy_hartree",
            "homo_ev",
            "lumo_ev",
            "homo_lumo_gap_ev",
            "dipole_debye",
            "polarizability_au",
            "total_charge_e",
        ]
    ].rename(
        columns={
            "selection_rank": ("xtb_selection_rank"),
            "attempt_number": ("xtb_attempt_number"),
            "method": "xtb_method",
            "gfn_level": ("xtb_gfn_level"),
            "solvation_model": ("xtb_solvation_model"),
            "total_energy_hartree": ("xtb_neutral_total_energy_hartree"),
            "homo_ev": "xtb_homo_ev",
            "lumo_ev": "xtb_lumo_ev",
            "homo_lumo_gap_ev": ("xtb_homo_lumo_gap_ev"),
            "dipole_debye": ("xtb_dipole_debye"),
            "polarizability_au": ("xtb_polarizability_au"),
            "total_charge_e": ("xtb_total_charge_e"),
        }
    )

    li_features = li_association[
        [
            "candidate_id",
            "best_site_rank",
            "best_donor_atom_index",
            "best_donor_element",
            "complex_energy_hartree",
            "delta_energy_hartree",
            "delta_energy_ev",
            "delta_energy_kcal_mol",
        ]
    ].rename(
        columns={
            "best_site_rank": ("xtb_li_best_site_rank"),
            "best_donor_atom_index": ("xtb_li_best_donor_atom_index"),
            "best_donor_element": ("xtb_li_best_donor_element"),
            "complex_energy_hartree": ("xtb_li_complex_energy_hartree"),
            "delta_energy_hartree": ("xtb_li_delta_energy_hartree"),
            "delta_energy_ev": ("xtb_li_delta_energy_ev"),
            "delta_energy_kcal_mol": ("xtb_li_delta_energy_kcal_mol"),
        }
    )

    site_qc = _build_site_qc(li_sites)

    features = (
        selected[base_columns]
        .merge(
            rdkit_descriptors,
            on="candidate_id",
            validate="one_to_one",
        )
        .merge(
            xtb_features,
            on="candidate_id",
            validate="one_to_one",
        )
        .merge(
            li_features,
            on="candidate_id",
            validate="one_to_one",
        )
        .merge(
            site_qc,
            on="candidate_id",
            validate="one_to_one",
        )
        .sort_values(
            "candidate_id",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    required_complete = [
        "candidate_id",
        "canonical_smiles",
        "candidate_family",
        "rdkit_mol_wt_g_mol",
        "rdkit_heavy_atom_count",
        "rdkit_tpsa_a2",
        "rdkit_hbond_acceptors",
        "rdkit_hbond_donors",
        "rdkit_fraction_csp3",
        "xtb_homo_ev",
        "xtb_lumo_ev",
        "xtb_homo_lumo_gap_ev",
        "xtb_dipole_debye",
        "xtb_polarizability_au",
        "xtb_total_charge_e",
        "xtb_li_delta_energy_ev",
        "qc_li_site_count",
        "qc_li_site_success_count",
    ]

    missing_count = int(features[required_complete].isna().sum().sum())

    if missing_count:
        raise ValueError(
            f"Phase 7 feature table has {missing_count} missing required values."
        )

    return features


def _resolve_path(
    root: Path,
    value: str,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return root / path


def _display_path(
    root: Path,
    path: Path,
) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def freeze_geometry_provenance(
    *,
    root: Path,
    feature_table: pd.DataFrame,
    neutral_attempts: pd.DataFrame,
    geometry_root: Path,
) -> pd.DataFrame:
    """Freeze selected neutral/Li geometries with hashes and provenance."""

    successful = neutral_attempts.loc[neutral_attempts["status"] == "success"].copy()

    if successful["candidate_id"].duplicated().any():
        raise ValueError("Expected one successful neutral xTB attempt per candidate.")

    expected_ids = set(feature_table["candidate_id"].astype(str))

    actual_ids = set(successful["candidate_id"].astype(str))

    if expected_ids != actual_ids:
        raise ValueError(
            "Neutral geometry provenance does not cover the complete Phase 7 pool."
        )

    attempts_by_id = successful.set_index("candidate_id")

    neutral_directory = geometry_root / "neutral"

    li_directory = geometry_root / "li_complex"

    neutral_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    li_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []

    for record in feature_table[
        [
            "candidate_id",
            "registry_version",
            "xtb_method",
            "xtb_gfn_level",
        ]
    ].to_dict(orient="records"):
        candidate_id = str(record["candidate_id"])

        work_directory = _resolve_path(
            root,
            str(
                attempts_by_id.loc[
                    candidate_id,
                    "work_directory",
                ]
            ),
        )

        neutral_source = work_directory / "optimization" / "xtbopt.xyz"

        li_source = (
            root
            / "results"
            / "xtb_li_v1"
            / "selected_complexes"
            / f"{candidate_id}.xyz"
        )

        systems = (
            (
                "neutral",
                neutral_source,
                neutral_directory / f"{candidate_id}.xyz",
                0,
                0,
            ),
            (
                "li_complex",
                li_source,
                li_directory / f"{candidate_id}.xyz",
                1,
                0,
            ),
        )

        for (
            system,
            source,
            destination,
            charge,
            uhf,
        ) in systems:
            if not source.is_file():
                raise FileNotFoundError(f"Missing frozen geometry source: {source}")

            source_hash = sha256_file(source)

            shutil.copy2(
                source,
                destination,
            )

            frozen_hash = sha256_file(destination)

            if source_hash != frozen_hash:
                raise RuntimeError(
                    "Geometry checksum changed during provenance freeze."
                )

            rows.append(
                {
                    "candidate_id": (candidate_id),
                    "registry_version": str(record["registry_version"]),
                    "system": system,
                    "system_charge": charge,
                    "unpaired_electrons": uhf,
                    "method": str(record["xtb_method"]),
                    "gfn_level": int(record["xtb_gfn_level"]),
                    "source_path": (
                        _display_path(
                            root,
                            source,
                        )
                    ),
                    "frozen_path": (
                        _display_path(
                            root,
                            destination,
                        )
                    ),
                    "sha256": frozen_hash,
                    "file_size_bytes": int(destination.stat().st_size),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(
        [
            "candidate_id",
            "system",
        ]
    )

    if len(manifest) != (2 * len(feature_table)):
        raise RuntimeError("Expected exactly two frozen geometries per candidate.")

    return manifest.reset_index(drop=True)
