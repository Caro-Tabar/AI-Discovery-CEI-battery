"""Force-field prescreening of molecular conformers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem.rdchem import Mol

ForceFieldName = Literal["MMFF94s", "UFF"]


class ForceFieldSettings(BaseModel):
    """Validated settings for conformer force-field prescreening."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_force_field: Literal["MMFF94s"] = "MMFF94s"
    fallback_force_field: Literal["UFF"] = "UFF"
    fallback_if_mmff_unavailable: bool = True
    max_iterations: int = Field(default=500, ge=1)
    num_threads: Literal[1] = 1


class ConformerEnergy(BaseModel):
    """Force-field optimization result for one conformer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conformer_id: int
    force_field: ForceFieldName
    converged: bool
    status: int
    energy_kcal_mol: float


def _optimize_conformers(
    molecule: Mol,
    settings: ForceFieldSettings,
    force_field: ForceFieldName,
) -> tuple[ConformerEnergy, ...]:
    """Optimize all conformers with one selected force field."""

    helpers: Any = rdForceFieldHelpers

    if force_field == "MMFF94s":
        raw_results: Any = helpers.MMFFOptimizeMoleculeConfs(
            molecule,
            numThreads=settings.num_threads,
            maxIters=settings.max_iterations,
            mmffVariant="MMFF94s",
        )
    else:
        raw_results = helpers.UFFOptimizeMoleculeConfs(
            molecule,
            numThreads=settings.num_threads,
            maxIters=settings.max_iterations,
        )

    conformer_ids = tuple(conformer.GetId() for conformer in molecule.GetConformers())

    if len(raw_results) != len(conformer_ids):
        raise RuntimeError("Force-field result count does not match conformer count.")

    results = tuple(
        ConformerEnergy(
            conformer_id=conformer_ids[index],
            force_field=force_field,
            converged=int(raw_results[index][0]) == 0,
            status=int(raw_results[index][0]),
            energy_kcal_mol=float(raw_results[index][1]),
        )
        for index in range(len(conformer_ids))
    )

    return tuple(
        sorted(
            results,
            key=lambda result: (
                not result.converged,
                result.energy_kcal_mol,
            ),
        )
    )


def prescreen_conformers(
    molecule: Mol,
    settings: ForceFieldSettings,
) -> tuple[ConformerEnergy, ...]:
    """Optimize and energy-rank an RDKit conformer ensemble."""

    helpers: Any = rdForceFieldHelpers

    if helpers.MMFFHasAllMoleculeParams(molecule):
        return _optimize_conformers(
            molecule,
            settings,
            "MMFF94s",
        )

    if not settings.fallback_if_mmff_unavailable:
        raise RuntimeError(
            "MMFF94s parameters are unavailable and UFF fallback is disabled."
        )

    if not helpers.UFFHasAllMoleculeParams(molecule):
        raise RuntimeError("Neither MMFF94s nor UFF has parameters for this molecule.")

    return _optimize_conformers(
        molecule,
        settings,
        "UFF",
    )
