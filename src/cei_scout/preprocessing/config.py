"""Validated preprocessing configuration loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cei_scout.preprocessing.conformers import ConformerSettings
from cei_scout.preprocessing.forcefield import ForceFieldSettings


class ConformerSelectionSettings(BaseModel):
    """Settings controlling final conformer diversity selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_selected_conformers: int = Field(default=3, ge=1, le=3)
    min_heavy_atom_rmsd_angstrom: float = Field(default=0.5, gt=0)


class ConformerPipelineConfig(BaseModel):
    """Validated complete conformer preprocessing configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    embedding: ConformerSettings
    prescreen: ForceFieldSettings
    selection: ConformerSelectionSettings


def load_conformer_pipeline_config(
    path: Path,
) -> ConformerPipelineConfig:
    """Load and validate the conformer preprocessing YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        raw_config: Any = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError("Conformer configuration must contain a YAML mapping.")

    return ConformerPipelineConfig.model_validate(raw_config)
