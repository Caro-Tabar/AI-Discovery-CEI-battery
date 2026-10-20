"""Run neutral GFN2-xTB screening across the frozen candidate pool."""

from pathlib import Path
from typing import Any

import pandas as pd

from cei_scout.preprocessing.xtb_config import load_xtb_config
from cei_scout.preprocessing.xtb_runner import (
    load_selected_geometries,
    run_xtb_attempt,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs/xtb.yaml"

CONFORMER_DIRECTORY = ROOT / "results/conformers_v1"

SDF_PATH = CONFORMER_DIRECTORY / "selected_conformers.sdf"

MANIFEST_PATH = CONFORMER_DIRECTORY / "selected_conformers.parquet"

OUTPUT_ROOT = ROOT / "results/xtb_neutral_v1"


def main() -> None:
    config = load_xtb_config(CONFIG_PATH)

    geometries, manifest = load_selected_geometries(
        SDF_PATH,
        MANIFEST_PATH,
    )

    candidate_ids = sorted(
        str(candidate_id) for candidate_id in manifest["candidate_id"].unique()
    )

    if len(candidate_ids) != 60:
        raise ValueError(f"Expected 60 candidates, found {len(candidate_ids)}.")

    rank_order = (
        config.retry.first_conformer_rank,
        config.retry.retry_conformer_rank,
    )

    attempt_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []

    failed_candidates: list[str] = []

    for candidate_id in candidate_ids:
        candidate_rows = manifest.loc[manifest["candidate_id"] == candidate_id]

        metadata = candidate_rows.iloc[0].to_dict()

        successful_attempt = None
        attempts_run = 0

        for attempt_number, rank in enumerate(
            rank_order,
            start=1,
        ):
            if attempt_number > config.retry.max_attempts:
                break

            geometry_key = (
                candidate_id,
                rank,
            )

            if geometry_key not in geometries:
                continue

            molecule = geometries[geometry_key]

            conformer_id = molecule.GetConformer().GetId()

            attempts_run += 1

            result = run_xtb_attempt(
                candidate_id=candidate_id,
                selection_rank=rank,
                attempt_number=attempt_number,
                molecule=molecule,
                conformer_id=conformer_id,
                config=config,
                output_root=OUTPUT_ROOT,
            )

            attempt_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_family": str(metadata["candidate_family"]),
                    "source_entry_id": str(metadata["source_entry_id"]),
                    "registry_version": str(metadata["registry_version"]),
                    "selection_rank": rank,
                    "attempt_number": attempt_number,
                    "status": result.status,
                    "failure_stage": result.failure_stage,
                    "message": result.message,
                    "work_directory": result.work_directory,
                }
            )

            if result.status != "success":
                continue

            if result.descriptors is None:
                raise RuntimeError("Successful xTB attempt has no descriptors.")

            successful_attempt = result

            descriptor_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_family": str(metadata["candidate_family"]),
                    "source_entry_id": str(metadata["source_entry_id"]),
                    "registry_version": str(metadata["registry_version"]),
                    "selection_rank": rank,
                    "attempt_number": attempt_number,
                    "method": config.method.name,
                    "gfn_level": config.method.gfn_level,
                    "solvation_model": config.solvation.model,
                    **result.descriptors.model_dump(),
                }
            )

            break

        if successful_attempt is None:
            failed_candidates.append(candidate_id)

            summary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "attempts_run": attempts_run,
                    "successful_selection_rank": None,
                }
            )
        else:
            summary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "success",
                    "attempts_run": attempts_run,
                    "successful_selection_rank": (successful_attempt.selection_rank),
                }
            )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(attempt_rows).to_parquet(
        OUTPUT_ROOT / "xtb_attempts.parquet",
        index=False,
    )

    pd.DataFrame(summary_rows).to_parquet(
        OUTPUT_ROOT / "xtb_candidate_summary.parquet",
        index=False,
    )

    pd.DataFrame(descriptor_rows).to_parquet(
        OUTPUT_ROOT / "xtb_descriptors.parquet",
        index=False,
    )

    print(f"Candidates processed: {len(candidate_ids)}")
    print(f"Successful candidates: {len(candidate_ids) - len(failed_candidates)}")
    print(f"Failed candidates: {len(failed_candidates)}")

    if failed_candidates:
        print("Failed candidate IDs:")

        for candidate_id in failed_candidates:
            print(f"  {candidate_id}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()
