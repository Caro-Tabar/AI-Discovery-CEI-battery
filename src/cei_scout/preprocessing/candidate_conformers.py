"""Candidate-pool conformer generation and selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdchem import Mol

from cei_scout.preprocessing.config import (
    ConformerPipelineConfig,
    ConformerSelectionSettings,
)
from cei_scout.preprocessing.conformers import generate_conformers
from cei_scout.preprocessing.forcefield import (
    ConformerEnergy,
    prescreen_conformers,
)


class CandidateInput(BaseModel):
    """Frozen candidate identity required for conformer preprocessing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    canonical_smiles: str
    candidate_family: str
    source_entry_id: str
    registry_version: str


class SelectedConformer(BaseModel):
    """One retained low-energy, structurally distinct conformer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    conformer_id: int
    selection_rank: int = Field(ge=1, le=3)

    force_field: Literal["MMFF94s", "UFF"]
    force_field_converged: bool
    force_field_status: int
    force_field_energy_kcal_mol: float

    relative_force_field_energy_kcal_mol: float | None = None
    min_rmsd_to_selected_angstrom: float | None = None


class CandidatePreprocessingResult(BaseModel):
    """Summary of preprocessing for one frozen candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    canonical_smiles: str
    candidate_family: str
    source_entry_id: str
    registry_version: str

    generated_conformer_count: int
    force_field_result_count: int
    selected_conformer_count: int

    status: Literal["ok", "failed"]
    error: str | None = None

    selected: tuple[SelectedConformer, ...] = ()


@dataclass(frozen=True)
class CandidatePreprocessingArtifact:
    """Result plus the RDKit molecule containing its optimized conformers."""

    result: CandidatePreprocessingResult
    molecule: Mol | None


def load_candidate_pool(
    path: Path,
    expected_count: int = 60,
) -> tuple[CandidateInput, ...]:
    """Load and validate the frozen selected candidate pool."""

    frame = pd.read_parquet(path)

    required_columns = {
        "candidate_id",
        "canonical_smiles",
        "candidate_family",
        "source_entry_id",
        "registry_version",
        "selection_status",
    }

    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Candidate table is missing required columns: {sorted(missing_columns)}"
        )

    selected_frame = frame.loc[
        frame["selection_status"] == "selected",
        [
            "candidate_id",
            "canonical_smiles",
            "candidate_family",
            "source_entry_id",
            "registry_version",
        ],
    ].copy()

    if len(selected_frame) != expected_count:
        raise ValueError(
            f"Expected {expected_count} selected candidates, "
            f"found {len(selected_frame)}."
        )

    if selected_frame["candidate_id"].duplicated().any():
        raise ValueError("Candidate IDs must be unique.")

    if selected_frame["canonical_smiles"].isna().any():
        raise ValueError("Every candidate must have canonical SMILES.")

    selected_frame = selected_frame.sort_values(
        "candidate_id",
        kind="stable",
    )

    return tuple(
        CandidateInput(
            candidate_id=str(row.candidate_id),
            canonical_smiles=str(row.canonical_smiles),
            candidate_family=str(row.candidate_family),
            source_entry_id=str(row.source_entry_id),
            registry_version=str(row.registry_version),
        )
        for row in selected_frame.itertuples(index=False)
    )


def _heavy_atom_rmsd_lookup(
    molecule: Mol,
) -> dict[tuple[int, int], float]:
    """Return symmetry-aware heavy-atom RMSDs between conformers."""

    conformer_ids = tuple(conformer.GetId() for conformer in molecule.GetConformers())

    if len(conformer_ids) < 2:
        return {}

    aligner: Any = rdMolAlign
    parameters: Any = aligner.BestAlignmentParams()

    parameters.ignoreHs = True
    parameters.numThreads = 1
    parameters.symmetrizeConjugatedTerminalGroups = True

    raw_rmsds: Any = aligner.GetAllConformerBestRMS(
        molecule,
        parameters,
    )

    expected_values = len(conformer_ids) * (len(conformer_ids) - 1) // 2

    if len(raw_rmsds) != expected_values:
        raise RuntimeError("Unexpected number of pairwise conformer RMSD values.")

    lookup: dict[tuple[int, int], float] = {}
    index = 0

    for upper_index in range(1, len(conformer_ids)):
        for lower_index in range(upper_index):
            first_id = conformer_ids[upper_index]
            second_id = conformer_ids[lower_index]

            rmsd = float(raw_rmsds[index])

            lookup[(first_id, second_id)] = rmsd
            lookup[(second_id, first_id)] = rmsd

            index += 1

    return lookup


def select_low_energy_distinct_conformers(
    candidate_id: str,
    molecule: Mol,
    ranked_results: tuple[ConformerEnergy, ...],
    settings: ConformerSelectionSettings,
) -> tuple[SelectedConformer, ...]:
    """Greedily retain low-energy structurally distinct conformers."""

    if not ranked_results:
        raise RuntimeError("No force-field conformer results are available.")

    rmsd_lookup = _heavy_atom_rmsd_lookup(molecule)

    selected_results: list[tuple[ConformerEnergy, float | None]] = []

    for result in ranked_results:
        if not selected_results:
            minimum_rmsd = None
        else:
            minimum_rmsd = min(
                rmsd_lookup[
                    (
                        result.conformer_id,
                        selected_result.conformer_id,
                    )
                ]
                for selected_result, _ in selected_results
            )

            if minimum_rmsd < settings.min_heavy_atom_rmsd_angstrom:
                continue

        selected_results.append((result, minimum_rmsd))

        if len(selected_results) == settings.max_selected_conformers:
            break

    reference_result = selected_results[0][0]

    selected: list[SelectedConformer] = []

    for rank, (result, minimum_rmsd) in enumerate(
        selected_results,
        start=1,
    ):
        relative_energy: float | None

        if result.converged and reference_result.converged:
            relative_energy = result.energy_kcal_mol - reference_result.energy_kcal_mol
        else:
            relative_energy = None

        selected.append(
            SelectedConformer(
                candidate_id=candidate_id,
                conformer_id=result.conformer_id,
                selection_rank=rank,
                force_field=result.force_field,
                force_field_converged=result.converged,
                force_field_status=result.status,
                force_field_energy_kcal_mol=result.energy_kcal_mol,
                relative_force_field_energy_kcal_mol=relative_energy,
                min_rmsd_to_selected_angstrom=minimum_rmsd,
            )
        )

    return tuple(selected)


def preprocess_candidate(
    candidate: CandidateInput,
    config: ConformerPipelineConfig,
) -> CandidatePreprocessingArtifact:
    """Generate, minimize, and select conformers for one candidate."""

    try:
        molecule, conformer_ids = generate_conformers(
            candidate.canonical_smiles,
            config.embedding,
        )

        force_field_results = prescreen_conformers(
            molecule,
            config.prescreen,
        )

        selected = select_low_energy_distinct_conformers(
            candidate.candidate_id,
            molecule,
            force_field_results,
            config.selection,
        )

    except (ValueError, RuntimeError) as error:
        return CandidatePreprocessingArtifact(
            result=CandidatePreprocessingResult(
                candidate_id=candidate.candidate_id,
                canonical_smiles=candidate.canonical_smiles,
                candidate_family=candidate.candidate_family,
                source_entry_id=candidate.source_entry_id,
                registry_version=candidate.registry_version,
                generated_conformer_count=0,
                force_field_result_count=0,
                selected_conformer_count=0,
                status="failed",
                error=str(error),
            ),
            molecule=None,
        )

    return CandidatePreprocessingArtifact(
        result=CandidatePreprocessingResult(
            candidate_id=candidate.candidate_id,
            canonical_smiles=candidate.canonical_smiles,
            candidate_family=candidate.candidate_family,
            source_entry_id=candidate.source_entry_id,
            registry_version=candidate.registry_version,
            generated_conformer_count=len(conformer_ids),
            force_field_result_count=len(force_field_results),
            selected_conformer_count=len(selected),
            status="ok",
            selected=selected,
        ),
        molecule=molecule,
    )


def preprocess_candidate_pool(
    candidates: tuple[CandidateInput, ...],
    config: ConformerPipelineConfig,
) -> tuple[CandidatePreprocessingArtifact, ...]:
    """Preprocess the full frozen candidate pool."""

    return tuple(preprocess_candidate(candidate, config) for candidate in candidates)


def write_conformer_pool_outputs(
    artifacts: tuple[CandidatePreprocessingArtifact, ...],
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    """Write selected conformers and machine-readable manifests."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    sdf_path = output_directory / "selected_conformers.sdf"
    manifest_path = output_directory / "selected_conformers.parquet"
    summary_path = output_directory / "candidate_summary.parquet"

    writer_factory: Any = Chem.SDWriter
    writer: Any = writer_factory(str(sdf_path))

    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    try:
        for artifact in artifacts:
            result = artifact.result

            summary_rows.append(
                result.model_dump(
                    exclude={"selected"},
                )
            )

            if artifact.molecule is None:
                continue

            for selected in result.selected:
                artifact.molecule.SetProp(
                    "candidate_id",
                    result.candidate_id,
                )
                artifact.molecule.SetProp(
                    "candidate_family",
                    result.candidate_family,
                )
                artifact.molecule.SetProp(
                    "source_entry_id",
                    result.source_entry_id,
                )
                artifact.molecule.SetProp(
                    "registry_version",
                    result.registry_version,
                )
                artifact.molecule.SetProp(
                    "selection_rank",
                    str(selected.selection_rank),
                )
                artifact.molecule.SetProp(
                    "force_field",
                    selected.force_field,
                )
                artifact.molecule.SetProp(
                    "force_field_energy_kcal_mol",
                    str(selected.force_field_energy_kcal_mol),
                )

                writer.write(
                    artifact.molecule,
                    confId=selected.conformer_id,
                )

                manifest_rows.append(
                    {
                        "candidate_family": result.candidate_family,
                        "source_entry_id": result.source_entry_id,
                        "registry_version": result.registry_version,
                        **selected.model_dump(),
                    }
                )
    finally:
        writer.close()

    pd.DataFrame(manifest_rows).to_parquet(
        manifest_path,
        index=False,
    )

    pd.DataFrame(summary_rows).to_parquet(
        summary_path,
        index=False,
    )

    return sdf_path, manifest_path, summary_path
