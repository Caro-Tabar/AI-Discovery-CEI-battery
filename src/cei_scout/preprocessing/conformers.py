"""Deterministic RDKit conformer generation for CEI-Scout."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdchem import Mol


class ConformerSettings(BaseModel):
    """Validated ETKDG settings for conformer generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["ETKDGv3"] = "ETKDGv3"
    random_seed: int = Field(ge=0, le=2_147_483_647)
    num_conformers: int = Field(ge=1)
    prune_rms_threshold_angstrom: float = Field(gt=0)
    num_threads: Literal[1] = 1
    enforce_chirality: bool = True
    use_small_ring_torsions: bool = True
    use_macrocycle_torsions: bool = True
    use_macrocycle_14config: bool = True
    only_heavy_atoms_for_pruning: bool = True


def _build_etkdg_parameters(
    settings: ConformerSettings,
) -> rdDistGeom.EmbedParameters:
    """Build ETKDGv3 parameters from validated settings."""

    params = rdDistGeom.ETKDGv3()

    # RDKit's C++-backed Python type information does not correctly expose
    # the scalar types of EmbedParameters properties to mypy.
    untyped_params: Any = params

    untyped_params.randomSeed = settings.random_seed
    untyped_params.pruneRmsThresh = settings.prune_rms_threshold_angstrom
    untyped_params.numThreads = settings.num_threads
    untyped_params.enforceChirality = settings.enforce_chirality

    untyped_params.useSmallRingTorsions = settings.use_small_ring_torsions
    untyped_params.useMacrocycleTorsions = settings.use_macrocycle_torsions
    untyped_params.useMacrocycle14config = settings.use_macrocycle_14config

    untyped_params.onlyHeavyAtomsForRMS = settings.only_heavy_atoms_for_pruning
    untyped_params.useSymmetryForPruning = True
    untyped_params.clearConfs = True

    return params


def generate_conformers(
    smiles: str,
    settings: ConformerSettings,
) -> tuple[Mol, tuple[int, ...]]:
    """Generate a deterministic hydrogen-explicit conformer ensemble.

    Args:
        smiles: Sanitizable molecular SMILES.
        settings: Validated ETKDG generation settings.

    Returns:
        Hydrogen-explicit RDKit molecule and generated conformer IDs.

    Raises:
        ValueError: If RDKit cannot parse the SMILES.
        RuntimeError: If ETKDG generates no conformers.
    """
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    molecule = Chem.AddHs(molecule)

    params = _build_etkdg_parameters(settings)

    raw_conformer_ids: Any = rdDistGeom.EmbedMultipleConfs(
        molecule,
        numConfs=settings.num_conformers,
        params=params,
    )

    conformer_ids = tuple(
        int(raw_conformer_ids[index]) for index in range(len(raw_conformer_ids))
    )
    if not conformer_ids:
        raise RuntimeError(f"ETKDG generated no conformers for SMILES {smiles!r}")

    return molecule, conformer_ids
