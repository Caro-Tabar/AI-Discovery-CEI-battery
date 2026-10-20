"""Run Li+ association screening across the frozen candidate pool."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from cei_scout.preprocessing.li_complexes import (
    check_organic_connectivity,
    compute_li_association_energy,
    generate_li_approaches,
    make_li_reference_molecule,
    molecule_with_xyz_coordinates,
)
from cei_scout.preprocessing.xtb_config import (
    load_xtb_config,
)
from cei_scout.preprocessing.xtb_runner import (
    load_selected_geometries,
    run_xtb_attempt,
    run_xtb_energy_only,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs/xtb.yaml"

CONFORMER_ROOT = ROOT / "results/conformers_v1"

SDF_PATH = CONFORMER_ROOT / "selected_conformers.sdf"

MANIFEST_PATH = CONFORMER_ROOT / "selected_conformers.parquet"

NEUTRAL_ROOT = ROOT / "results/xtb_neutral_v1"

ATTEMPTS_PATH = NEUTRAL_ROOT / "xtb_attempts.parquet"

DESCRIPTORS_PATH = NEUTRAL_ROOT / "xtb_descriptors.parquet"

OUTPUT_ROOT = ROOT / "results/xtb_li_v1"


def _resolve_work_directory(
    value: str,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return ROOT / path


def main() -> None:
    config = load_xtb_config(CONFIG_PATH)

    geometries, _manifest = load_selected_geometries(
        SDF_PATH,
        MANIFEST_PATH,
    )

    neutral_attempts = pd.read_parquet(ATTEMPTS_PATH)

    neutral_descriptors = pd.read_parquet(DESCRIPTORS_PATH)

    successful_attempts = neutral_attempts.loc[
        neutral_attempts["status"] == "success"
    ].copy()

    if len(neutral_descriptors) != 60:
        raise ValueError("Expected 60 neutral descriptor rows.")

    # Li+ reference energy
    lithium = make_li_reference_molecule()

    lithium_result = run_xtb_energy_only(
        system_id="Li+",
        molecule=lithium,
        conformer_id=0,
        system_charge=config.li_association.charge,
        unpaired_electrons=config.li_association.uhf,
        config=config,
        work_directory=(OUTPUT_ROOT / "li_reference"),
    )

    if (
        lithium_result.status != "success"
        or lithium_result.total_energy_hartree is None
    ):
        raise RuntimeError("Li+ reference calculation failed.")

    lithium_energy = lithium_result.total_energy_hartree

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_ROOT / "li_reference.json").write_text(
        json.dumps(
            {
                "method": config.method.name,
                "charge": (config.li_association.charge),
                "uhf": (config.li_association.uhf),
                "total_energy_hartree": (lithium_energy),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    site_rows: list[dict[str, Any]] = []

    association_rows: list[dict[str, Any]] = []

    pool_rows: list[dict[str, Any]] = []

    selected_directory = OUTPUT_ROOT / "selected_complexes"

    selected_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for neutral_row in neutral_descriptors.sort_values("candidate_id").to_dict(
        orient="records"
    ):
        candidate_id = str(neutral_row["candidate_id"])

        rank = int(neutral_row["selection_rank"])

        reference_molecule = geometries[(candidate_id, rank)]

        candidate_attempts = successful_attempts.loc[
            successful_attempts["candidate_id"] == candidate_id
        ]

        if len(candidate_attempts) != 1:
            raise ValueError(
                f"Expected one successful neutral attempt for {candidate_id}."
            )

        work_directory = _resolve_work_directory(
            str(candidate_attempts.iloc[0]["work_directory"])
        )

        neutral_xyz = work_directory / "optimization" / "xtbopt.xyz"

        # P07-18 also verifies that the neutral
        # optimization itself preserved connectivity.
        neutral_connectivity = check_organic_connectivity(
            reference_molecule,
            neutral_xyz,
        )

        if not (neutral_connectivity.connectivity_preserved):
            pool_rows.append(
                {
                    "candidate_id": (candidate_id),
                    "prospective_eligible": (False),
                    "exclusion_reason": ("neutral_connectivity_change"),
                }
            )

            continue

        neutral_molecule = molecule_with_xyz_coordinates(
            reference_molecule,
            neutral_xyz,
        )

        approaches = generate_li_approaches(
            neutral_molecule,
            config.li_association,
        )

        if not approaches:
            pool_rows.append(
                {
                    "candidate_id": (candidate_id),
                    "prospective_eligible": (False),
                    "exclusion_reason": ("no_li_donor_site"),
                }
            )

            continue

        valid_sites: list[dict[str, Any]] = []

        for approach in approaches:
            metadata = approach.metadata

            directory_label = (
                f"site_{metadata.site_rank:02d}_"
                f"{metadata.donor_element}_"
                f"{metadata.donor_atom_index}"
            )

            result = run_xtb_attempt(
                candidate_id=candidate_id,
                selection_rank=(metadata.site_rank),
                attempt_number=1,
                molecule=approach.molecule,
                conformer_id=0,
                config=config,
                output_root=OUTPUT_ROOT,
                system_charge=(config.li_association.charge),
                unpaired_electrons=(config.li_association.uhf),
                directory_label=directory_label,
            )

            row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "site_rank": (metadata.site_rank),
                "donor_atom_index": (metadata.donor_atom_index),
                "donor_element": (metadata.donor_element),
                "starting_distance_angstrom": (metadata.starting_distance_angstrom),
                "status": result.status,
                "failure_stage": (result.failure_stage),
                "message": result.message,
                "work_directory": (result.work_directory),
            }

            if result.status == "success" and result.descriptors is not None:
                optimized_xyz = (
                    Path(result.work_directory) / "optimization" / "xtbopt.xyz"
                )

                connectivity = check_organic_connectivity(
                    neutral_molecule,
                    optimized_xyz,
                )

                row.update(
                    {
                        "complex_energy_hartree": (
                            result.descriptors.total_energy_hartree
                        ),
                        "connectivity_preserved": (connectivity.connectivity_preserved),
                        "fragmented": (connectivity.fragmented),
                        "removed_bond_count": (connectivity.removed_bond_count),
                        "added_bond_count": (connectivity.added_bond_count),
                    }
                )

                if connectivity.connectivity_preserved:
                    valid_sites.append(
                        {
                            **row,
                            "optimized_xyz": str(optimized_xyz),
                        }
                    )

            site_rows.append(row)

        if not valid_sites:
            pool_rows.append(
                {
                    "candidate_id": (candidate_id),
                    "prospective_eligible": (False),
                    "exclusion_reason": ("no_valid_li_complex"),
                }
            )

            continue

        # All sites have identical composition,
        # so comparing their xTB total energies
        # is valid within this candidate.
        best_site = min(
            valid_sites,
            key=lambda row: float(row["complex_energy_hartree"]),
        )

        association = compute_li_association_energy(
            complex_energy_hartree=float(best_site["complex_energy_hartree"]),
            neutral_energy_hartree=float(neutral_row["total_energy_hartree"]),
            lithium_energy_hartree=(lithium_energy),
        )

        association_rows.append(
            {
                "candidate_id": (candidate_id),
                "candidate_family": str(neutral_row["candidate_family"]),
                "source_entry_id": str(neutral_row["source_entry_id"]),
                "registry_version": str(neutral_row["registry_version"]),
                "best_site_rank": (best_site["site_rank"]),
                "best_donor_atom_index": (best_site["donor_atom_index"]),
                "best_donor_element": (best_site["donor_element"]),
                "neutral_energy_hartree": (float(neutral_row["total_energy_hartree"])),
                "li_reference_energy_hartree": (lithium_energy),
                "complex_energy_hartree": (float(best_site["complex_energy_hartree"])),
                **association.model_dump(),
            }
        )

        shutil.copy2(
            Path(str(best_site["optimized_xyz"])),
            (selected_directory / f"{candidate_id}.xyz"),
        )

        pool_rows.append(
            {
                "candidate_id": candidate_id,
                "prospective_eligible": True,
                "exclusion_reason": None,
            }
        )

    pd.DataFrame(site_rows).to_parquet(
        OUTPUT_ROOT / "li_site_attempts.parquet",
        index=False,
    )

    pd.DataFrame(association_rows).to_parquet(
        OUTPUT_ROOT / "li_association.parquet",
        index=False,
    )

    frozen_pool = pd.DataFrame(pool_rows).sort_values("candidate_id")

    frozen_pool.to_parquet(
        OUTPUT_ROOT / "prospective_pool_v1.parquet",
        index=False,
    )

    print(f"Candidates evaluated: {len(frozen_pool)}")

    print(f"Prospective eligible: {int(frozen_pool['prospective_eligible'].sum())}")

    print(f"Excluded after xTB QC: {int((~frozen_pool['prospective_eligible']).sum())}")

    print(f"Li association rows: {len(association_rows)}")


if __name__ == "__main__":
    main()
