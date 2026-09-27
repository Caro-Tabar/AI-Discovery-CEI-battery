"""Validated configuration for GFN2-xTB screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class XtbMethodSettings(BaseModel):
    """Electronic-structure method settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["GFN2-xTB"]
    gfn_level: Literal[2]


class XtbSolvationSettings(BaseModel):
    """Solvation settings for the xTB screening layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["none"]
    solvent: None = None


class XtbOptimizationSettings(BaseModel):
    """Geometry-optimization settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["tight"]
    max_cycles: int = Field(ge=1)


class XtbParallelSettings(BaseModel):
    """Thread settings passed to xTB through environment variables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    omp_num_threads: int = Field(ge=1)
    mkl_num_threads: int = Field(ge=1)
    omp_max_active_levels: int = Field(ge=1)


class XtbRetrySettings(BaseModel):
    """Retry policy for failed neutral optimizations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: Literal[2]
    first_conformer_rank: Literal[1]
    retry_conformer_rank: Literal[2]
    change_method_on_retry: Literal[False]
    change_solvation_on_retry: Literal[False]


class XtbExecutionSettings(BaseModel):
    """Operational execution limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: int = Field(ge=1)


class XtbConfig(BaseModel):
    """Complete validated neutral xTB configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    method: XtbMethodSettings
    solvation: XtbSolvationSettings
    optimization: XtbOptimizationSettings
    parallel: XtbParallelSettings
    retry: XtbRetrySettings
    execution: XtbExecutionSettings
    li_association: XtbLiAssociationSettings


def load_xtb_config(path: Path) -> XtbConfig:
    """Load and validate the xTB YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        raw_config: Any = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError("xTB configuration must contain a YAML mapping.")

    return XtbConfig.model_validate(raw_config)


class XtbLiAssociationSettings(BaseModel):
    """Starting-geometry rules for Li+ association screening."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    charge: Literal[1]
    uhf: Literal[0]

    oxygen_distance_angstrom: float = Field(gt=0)
    nitrogen_distance_angstrom: float = Field(gt=0)
    sulfur_distance_angstrom: float = Field(gt=0)

    sulfur_max_total_valence: int = Field(ge=1)
