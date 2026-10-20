"""Run Phase 8 screening DFT for all frozen candidate geometries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from cei_scout.dft.molecule import build_molecule
from cei_scout.dft.scf import build_screening_scf

CANDIDATE_RE = re.compile(r"CEI-\d{4}")

FUNCTIONAL = "wb97x-v"
BASIS = "def2-svpd"
PCM_MODEL = "IEF-PCM"
DIELECTRIC = 20.0
GRID_LEVEL = 3

STATES = (
    ("neutral", 0, 0),
    ("oxidized", +1, 1),
    ("reduced", -1, 1),
)

FIELDNAMES = (
    "candidate_id",
    "state",
    "attempt",
    "charge",
    "spin",
    "electron_count",
    "reference",
    "status",
    "energy_eh",
    "runtime_seconds",
    "s2",
    "multiplicity",
    "xyz_path",
    "xyz_sha256",
    "functional",
    "basis",
    "pcm_model",
    "dielectric",
    "grid_level",
    "vv10_nlc_active",
    "started_utc",
    "finished_utc",
    "error",
)


def sha256_file(path: Path) -> str:
    """Return SHA-256 digest for provenance."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def candidate_id_from_path(path: Path) -> str:
    """Extract CEI candidate ID from path or XYZ comment line."""
    match = CANDIDATE_RE.search(str(path))
    if match:
        return match.group(0)

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not read XYZ file: {path}") from exc

    if len(lines) >= 2:
        match = CANDIDATE_RE.search(lines[1])
        if match:
            return match.group(0)

    raise ValueError(f"Could not determine candidate ID for {path}")


def discover_geometries(root: Path) -> dict[str, Path]:
    """Discover exactly one XYZ geometry for each candidate."""
    if not root.is_dir():
        raise NotADirectoryError(f"XYZ root does not exist: {root}")

    geometries: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in sorted(root.glob("CEI-*/attempt_1_rank_1/optimization/xtbopt.xyz")):
        candidate_id = candidate_id_from_path(path)

        if candidate_id in geometries:
            duplicates.setdefault(candidate_id, [geometries[candidate_id]])
            duplicates[candidate_id].append(path)
        else:
            geometries[candidate_id] = path

    if duplicates:
        details = "\n".join(
            f"{candidate_id}: {paths}"
            for candidate_id, paths in sorted(duplicates.items())
        )
        raise ValueError(
            "Multiple XYZ files found for one or more candidates.\n"
            "Point --xyz-root at the frozen neutral geometry directory only.\n"
            f"{details}"
        )

    if len(geometries) != 60:
        raise ValueError(
            f"Expected exactly 60 candidate geometries, found {len(geometries)}."
        )

    return dict(sorted(geometries.items()))


def load_existing_results(
    output_path: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, str]],
    dict[tuple[str, str], int],
]:
    """Load latest recorded attempt and attempt count for each state."""
    latest: dict[tuple[str, str], dict[str, str]] = {}
    attempts: dict[tuple[str, str], int] = {}

    if not output_path.exists():
        return latest, attempts

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            key = (row["candidate_id"], row["state"])
            latest[key] = row
            attempts[key] = max(
                attempts.get(key, 0),
                int(row.get("attempt", "1")),
            )

    return latest, attempts


def append_result(output_path: Path, row: dict[str, Any]) -> None:
    """Append one result and force it to disk immediately."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

        # Make the checkpoint durable before moving to the next state.
        handle.flush()
        os.fsync(handle.fileno())


def run_state(
    *,
    candidate_id: str,
    state: str,
    charge: int,
    spin: int,
    xyz_path: Path,
    xyz_sha256: str,
    attempt: int,
    log_dir: Path,
    force_newton: bool = False,
) -> dict[str, Any]:
    """Run one state without allowing failure to stop the batch."""
    started = datetime.now(UTC).isoformat()
    start = perf_counter()

    electron_count: int | str = ""
    reference = ""
    energy_eh: float | str = ""
    s2: float | str = ""
    multiplicity: float | str = ""
    nlc_active: bool | str = ""
    status = "failed"
    error = ""

    try:
        mol = build_molecule(
            xyz_path,
            basis=BASIS,
            charge=charge,
            spin=spin,
        )

        electron_count = mol.nelectron

        mf = build_screening_scf(
            mol,
            functional=FUNCTIONAL,
            dielectric_constant=DIELECTRIC,
            grid_level=GRID_LEVEL,
        )

        reference = type(mf).__name__
        nlc_active = bool(mf.do_nlc())

        if force_newton:
            print("Previous standard SCF attempt was unconverged.")
            print("Starting retry directly with second-order SCF.")

            mf = mf.newton()
            mf.max_cycle = 100

            energy_eh = float(mf.kernel())

        else:
            energy_eh = float(mf.kernel())
            # If the standard DIIS SCF does not converge, retry from its
            # current orbitals with PySCF's second-order Newton solver.
            if not mf.converged:
                print("Standard SCF unconverged; retrying with second-order SCF.")

                previous_mo_coeff = mf.mo_coeff
                previous_mo_occ = mf.mo_occ

                mf = mf.newton()
                mf.max_cycle = 100

                energy_eh = float(
                    mf.kernel(
                        previous_mo_coeff,
                        previous_mo_occ,
                    )
                )

        if spin != 0:
            s2_value, multiplicity_value = mf.spin_square()
            s2 = float(s2_value)
            multiplicity = float(multiplicity_value)

        if mf.converged:
            status = "success"
        else:
            status = "unconverged"
            error = "SCF did not converge."

    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{candidate_id}_{state}_attempt{attempt}.log"

        log_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )

    runtime = perf_counter() - start
    finished = datetime.now(UTC).isoformat()

    return {
        "candidate_id": candidate_id,
        "state": state,
        "attempt": attempt,
        "charge": charge,
        "spin": spin,
        "electron_count": electron_count,
        "reference": reference,
        "status": status,
        "energy_eh": energy_eh,
        "runtime_seconds": runtime,
        "s2": s2,
        "multiplicity": multiplicity,
        "xyz_path": str(xyz_path),
        "xyz_sha256": xyz_sha256,
        "functional": FUNCTIONAL,
        "basis": BASIS,
        "pcm_model": PCM_MODEL,
        "dielectric": DIELECTRIC,
        "grid_level": GRID_LEVEL,
        "vv10_nlc_active": nlc_active,
        "started_utc": started,
        "finished_utc": finished,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--xyz-root",
        type=Path,
        required=True,
        help="Directory containing exactly one frozen neutral XYZ per candidate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase8_v1/dft_state_results.csv"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate geometry discovery without running DFT.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previous failed or unconverged states.",
    )

    args = parser.parse_args()

    geometries = discover_geometries(args.xyz_root)

    print(f"Validated candidate geometries: {len(geometries)}")

    if args.dry_run:
        print("Dry run passed. No DFT calculations were started.")
        return

    latest, attempts = load_existing_results(args.output)

    log_dir = args.output.parent / "logs"

    total_states = len(geometries) * len(STATES)
    session_start = perf_counter()
    session_completed = 0

    for candidate_id, xyz_path in geometries.items():
        geometry_hash = sha256_file(xyz_path)

        for state, charge, spin in STATES:
            key = (candidate_id, state)

            previous = latest.get(key)

            if previous is not None:
                previous_hash = previous.get("xyz_sha256", "")

                if previous_hash and previous_hash != geometry_hash:
                    raise RuntimeError(
                        f"Geometry hash changed for {candidate_id}. "
                        "Refusing to mix results from different geometries."
                    )

                previous_status = previous.get("status", "")

                if previous_status == "success":
                    print(f"SKIP {candidate_id} {state}: already successful")
                    continue

                if not args.retry_failed:
                    print(
                        f"SKIP {candidate_id} {state}: "
                        f"previous status={previous_status}"
                    )
                    continue

            attempt = attempts.get(key, 0) + 1

            print(
                f"\nRUN {candidate_id} {state} "
                f"(charge={charge}, spin={spin}, attempt={attempt})"
            )

            force_newton = (
                previous is not None
                and previous.get("status") == "unconverged"
                and args.retry_failed
            )

            row = run_state(
                candidate_id=candidate_id,
                state=state,
                charge=charge,
                spin=spin,
                xyz_path=xyz_path,
                xyz_sha256=geometry_hash,
                attempt=attempt,
                log_dir=log_dir,
                force_newton=force_newton,
            )

            append_result(args.output, row)

            latest[key] = {name: str(row[name]) for name in FIELDNAMES}
            attempts[key] = attempt
            session_completed += 1

            print(
                f"{row['status'].upper()} | "
                f"E={row['energy_eh']} Eh | "
                f"{float(row['runtime_seconds']) / 60:.2f} min"
            )

            elapsed_hours = (perf_counter() - session_start) / 3600

            if session_completed:
                mean_hours = elapsed_hours / session_completed
                remaining = total_states - len(
                    [
                        value
                        for value in latest.values()
                        if value.get("status") == "success"
                    ]
                )
                eta_hours = mean_hours * remaining

                print(
                    f"Session average: {mean_hours * 60:.2f} min/state | "
                    f"rough remaining time: {eta_hours:.1f} h"
                )

    final_latest, _ = load_existing_results(args.output)

    counts = {
        "success": 0,
        "unconverged": 0,
        "failed": 0,
    }

    for row in final_latest.values():
        status = row.get("status", "")
        if status in counts:
            counts[status] += 1

    print("\n=== PHASE 8 SCREENING SUMMARY ===")
    print(f"Successful:   {counts['success']}")
    print(f"Unconverged:  {counts['unconverged']}")
    print(f"Failed:       {counts['failed']}")
    print(f"Results:      {args.output}")


if __name__ == "__main__":
    main()
