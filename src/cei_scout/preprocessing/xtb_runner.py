"""Execute reproducible neutral GFN2-xTB calculations."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from cei_scout.preprocessing.xtb_config import XtbConfig
from cei_scout.preprocessing.xtb_parser import (
    XtbDescriptors,
    parse_xtb_output,
    parse_xtb_total_energy,
)


class XtbAttemptResult(BaseModel):
    """Outcome of one xTB optimization/property attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    selection_rank: int
    attempt_number: int

    status: Literal["success", "failed"]
    failure_stage: str | None = None
    message: str | None = None

    work_directory: str

    descriptors: XtbDescriptors | None = None


def load_selected_geometries(
    sdf_path: Path,
    manifest_path: Path,
) -> tuple[
    dict[tuple[str, int], Mol],
    pd.DataFrame,
]:
    """Load selected conformer geometries and validate them against the manifest."""

    manifest = pd.read_parquet(manifest_path)

    required_columns = {
        "candidate_id",
        "selection_rank",
        "candidate_family",
        "source_entry_id",
        "registry_version",
        "force_field",
        "force_field_converged",
        "force_field_energy_kcal_mol",
    }

    missing_columns = required_columns.difference(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Conformer manifest is missing columns: {sorted(missing_columns)}"
        )

    supplier: Any = Chem.SDMolSupplier(
        str(sdf_path),
        removeHs=False,
    )

    geometries: dict[tuple[str, int], Mol] = {}

    for molecule in supplier:
        if molecule is None:
            raise ValueError("Failed to parse a selected conformer SDF record.")

        if not molecule.HasProp("candidate_id"):
            raise ValueError("SDF record is missing candidate_id.")

        if not molecule.HasProp("selection_rank"):
            raise ValueError("SDF record is missing selection_rank.")

        candidate_id = molecule.GetProp("candidate_id")
        selection_rank = int(molecule.GetProp("selection_rank"))

        key = (candidate_id, selection_rank)

        if key in geometries:
            raise ValueError(
                f"Duplicate selected geometry for {candidate_id} rank {selection_rank}."
            )

        geometries[key] = molecule

    manifest_keys = {
        (
            str(record["candidate_id"]),
            int(record["selection_rank"]),
        )
        for record in manifest[
            [
                "candidate_id",
                "selection_rank",
            ]
        ].to_dict(orient="records")
    }

    geometry_keys = set(geometries)

    if geometry_keys != manifest_keys:
        raise ValueError("SDF geometries and selected-conformer manifest do not match.")

    return geometries, manifest


def _write_xyz(
    molecule: Mol,
    conformer_id: int,
    path: Path,
    comment: str,
) -> None:
    """Write one explicit conformer as an XYZ file."""

    conformer = molecule.GetConformer(conformer_id)

    lines = [
        str(molecule.GetNumAtoms()),
        comment,
    ]

    for atom_index, atom in enumerate(molecule.GetAtoms()):
        position = conformer.GetAtomPosition(atom_index)

        lines.append(
            f"{atom.GetSymbol():<3} "
            f"{position.x: .12f} "
            f"{position.y: .12f} "
            f"{position.z: .12f}"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _xtb_environment(config: XtbConfig) -> dict[str, str]:
    """Construct the environment controlling xTB thread counts."""

    environment = os.environ.copy()

    environment["OMP_NUM_THREADS"] = str(config.parallel.omp_num_threads)
    environment["MKL_NUM_THREADS"] = str(config.parallel.mkl_num_threads)
    environment["OMP_MAX_ACTIVE_LEVELS"] = str(config.parallel.omp_max_active_levels)

    return environment


def _run_logged_command(
    command: list[str],
    work_directory: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> tuple[int | None, bool]:
    """Execute one external command with persistent stdout/stderr logs."""

    work_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (work_directory / "command.txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )

    stdout_path = work_directory / "xtb.out"
    stderr_path = work_directory / "xtb.err"

    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        try:
            result = subprocess.run(
                command,
                cwd=work_directory,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, True

    return result.returncode, False


def run_xtb_attempt(
    *,
    candidate_id: str,
    selection_rank: int,
    attempt_number: int,
    molecule: Mol,
    conformer_id: int,
    config: XtbConfig,
    output_root: Path,
    system_charge: int = 0,
    unpaired_electrons: int = 0,
    directory_label: str | None = None,
) -> XtbAttemptResult:
    """Run one neutral GFN2-xTB optimization and property calculation."""

    if directory_label is None:
        directory_label = f"attempt_{attempt_number}_rank_{selection_rank}"

    attempt_directory = output_root / candidate_id / directory_label

    optimization_directory = attempt_directory / "optimization"
    property_directory = attempt_directory / "properties"

    optimization_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = optimization_directory / "input.xyz"

    _write_xyz(
        molecule,
        conformer_id,
        input_path,
        (
            f"{candidate_id} rank {selection_rank}; "
            f"charge={system_charge}; uhf={unpaired_electrons}"
        ),
    )

    environment = _xtb_environment(config)

    optimization_command = [
        "xtb",
        "input.xyz",
        "--gfn",
        str(config.method.gfn_level),
        "--chrg",
        str(system_charge),
        "--uhf",
        str(unpaired_electrons),
        "--opt",
        config.optimization.level,
        "--cycles",
        str(config.optimization.max_cycles),
    ]

    return_code, timed_out = _run_logged_command(
        optimization_command,
        optimization_directory,
        environment,
        config.execution.timeout_seconds,
    )

    if timed_out:
        return XtbAttemptResult(
            candidate_id=candidate_id,
            selection_rank=selection_rank,
            attempt_number=attempt_number,
            status="failed",
            failure_stage="optimization_timeout",
            message="xTB optimization exceeded the configured timeout.",
            work_directory=str(attempt_directory),
        )

    optimization_output = (optimization_directory / "xtb.out").read_text(
        encoding="utf-8",
        errors="replace",
    )

    optimized_geometry = optimization_directory / "xtbopt.xyz"

    if (
        return_code != 0
        or "GEOMETRY OPTIMIZATION CONVERGED" not in optimization_output
        or not optimized_geometry.exists()
    ):
        return XtbAttemptResult(
            candidate_id=candidate_id,
            selection_rank=selection_rank,
            attempt_number=attempt_number,
            status="failed",
            failure_stage="optimization",
            message="GFN2-xTB geometry optimization did not converge.",
            work_directory=str(attempt_directory),
        )

    property_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    property_geometry = property_directory / "optimized.xyz"

    shutil.copy2(
        optimized_geometry,
        property_geometry,
    )

    property_command = [
        "xtb",
        "optimized.xyz",
        "--gfn",
        str(config.method.gfn_level),
        "--chrg",
        str(system_charge),
        "--uhf",
        str(unpaired_electrons),
        "--json",
        "--pop",
        "--dipole",
        "--alpha",
    ]

    return_code, timed_out = _run_logged_command(
        property_command,
        property_directory,
        environment,
        config.execution.timeout_seconds,
    )

    if timed_out:
        return XtbAttemptResult(
            candidate_id=candidate_id,
            selection_rank=selection_rank,
            attempt_number=attempt_number,
            status="failed",
            failure_stage="property_timeout",
            message="xTB property calculation exceeded the timeout.",
            work_directory=str(attempt_directory),
        )

    if return_code != 0:
        return XtbAttemptResult(
            candidate_id=candidate_id,
            selection_rank=selection_rank,
            attempt_number=attempt_number,
            status="failed",
            failure_stage="property",
            message="GFN2-xTB property calculation failed.",
            work_directory=str(attempt_directory),
        )

    property_output = (property_directory / "xtb.out").read_text(
        encoding="utf-8",
        errors="replace",
    )

    try:
        descriptors = parse_xtb_output(property_output)
    except ValueError as error:
        return XtbAttemptResult(
            candidate_id=candidate_id,
            selection_rank=selection_rank,
            attempt_number=attempt_number,
            status="failed",
            failure_stage="parsing",
            message=str(error),
            work_directory=str(attempt_directory),
        )

    return XtbAttemptResult(
        candidate_id=candidate_id,
        selection_rank=selection_rank,
        attempt_number=attempt_number,
        status="success",
        work_directory=str(attempt_directory),
        descriptors=descriptors,
    )


class XtbEnergyResult(BaseModel):
    """Outcome of an xTB single-point energy calculation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_id: str
    status: Literal["success", "failed"]

    system_charge: int
    unpaired_electrons: int

    work_directory: str
    total_energy_hartree: float | None = None
    message: str | None = None


def run_xtb_energy_only(
    *,
    system_id: str,
    molecule: Mol,
    conformer_id: int,
    system_charge: int,
    unpaired_electrons: int,
    config: XtbConfig,
    work_directory: Path,
) -> XtbEnergyResult:
    """Run one GFN2-xTB single-point calculation and retain its energy."""

    work_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = work_directory / "input.xyz"

    _write_xyz(
        molecule,
        conformer_id,
        input_path,
        (f"{system_id}; charge={system_charge}; uhf={unpaired_electrons}"),
    )

    environment = _xtb_environment(config)

    command = [
        "xtb",
        "input.xyz",
        "--gfn",
        str(config.method.gfn_level),
        "--chrg",
        str(system_charge),
        "--uhf",
        str(unpaired_electrons),
    ]

    return_code, timed_out = _run_logged_command(
        command,
        work_directory,
        environment,
        config.execution.timeout_seconds,
    )

    if timed_out:
        return XtbEnergyResult(
            system_id=system_id,
            status="failed",
            system_charge=system_charge,
            unpaired_electrons=unpaired_electrons,
            work_directory=str(work_directory),
            message="xTB single-point calculation timed out.",
        )

    if return_code != 0:
        return XtbEnergyResult(
            system_id=system_id,
            status="failed",
            system_charge=system_charge,
            unpaired_electrons=unpaired_electrons,
            work_directory=str(work_directory),
            message="xTB single-point calculation failed.",
        )

    output = (work_directory / "xtb.out").read_text(
        encoding="utf-8",
        errors="replace",
    )

    try:
        energy = parse_xtb_total_energy(output)
    except ValueError as error:
        return XtbEnergyResult(
            system_id=system_id,
            status="failed",
            system_charge=system_charge,
            unpaired_electrons=unpaired_electrons,
            work_directory=str(work_directory),
            message=str(error),
        )

    return XtbEnergyResult(
        system_id=system_id,
        status="success",
        system_charge=system_charge,
        unpaired_electrons=unpaired_electrons,
        work_directory=str(work_directory),
        total_energy_hartree=energy,
    )
