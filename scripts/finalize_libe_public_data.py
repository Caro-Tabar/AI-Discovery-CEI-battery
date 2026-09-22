#!/usr/bin/env python3
"""Freeze the LIBE benchmark and quantify prospective-pool domain shift."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_DOI = "10.6084/m9.figshare.14226464.v2"
SOURCE_VERSION = "2"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_METHOD = "omegaB97X-V/def2-TZVPPD/SMD"

ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/manifests/libe_benchmark_selection_v0.json"),
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=Path("data/processed/candidates_v1.parquet"),
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("data/processed/libe_benchmark_v1.parquet"),
    )
    parser.add_argument(
        "--domain-shift-csv",
        type=Path,
        default=Path("results/tables/libe_candidate_domain_shift_v1.csv"),
    )
    parser.add_argument(
        "--domain-shift-json",
        type=Path,
        default=Path("results/reports/libe_domain_shift_v1.json"),
    )

    return parser.parse_args()


def _numeric_summary(values: list[float]) -> dict[str, float]:
    """Return compact descriptive statistics."""
    if not values:
        raise ValueError("Cannot summarize an empty numeric sequence.")

    ordered = sorted(values)

    return {
        "min": ordered[0],
        "median": median(ordered),
        "mean": mean(ordered),
        "max": ordered[-1],
    }


def _candidate_weight(row: dict[str, Any]) -> float:
    """Resolve the Phase 5 molecular-weight field."""
    possible_names = (
        "molecular_weight_g_mol",
        "molecular_weight",
        "mol_weight",
        "mw",
    )

    for name in possible_names:
        value = row.get(name)

        if isinstance(value, (int, float)):
            return float(value)

    raise ValueError(
        "Could not find a numeric molecular-weight column in "
        f"candidate record. Available columns: {sorted(row)}"
    )


def _formula_elements(formula: str) -> set[str]:
    """Extract element symbols from a molecular formula."""
    return set(ELEMENT_PATTERN.findall(formula))


def _candidate_elements(row: dict[str, Any]) -> set[str]:
    """Resolve candidate element membership robustly."""
    value = row.get("elements")

    if isinstance(value, (list, tuple, set)):
        return {str(element) for element in value}

    if isinstance(value, str):
        stripped = value.strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None

        if isinstance(parsed, (list, tuple, set)):
            return {str(element) for element in parsed}

        if "-" in stripped:
            return {token for token in stripped.split("-") if token}

    for name in (
        "formula",
        "molecular_formula",
        "formula_alphabetical",
    ):
        formula = row.get(name)

        if isinstance(formula, str):
            return _formula_elements(formula)

    raise ValueError(
        "Could not determine elements from candidate record. "
        f"Available columns: {sorted(row)}"
    )


def _benchmark_elements(row: dict[str, Any]) -> set[str]:
    """Return elements from the LIBE chemical-system field."""
    chemical_system = row.get("chemical_system")

    if not isinstance(chemical_system, str):
        raise ValueError("LIBE benchmark record lacks chemical_system.")

    return {token for token in chemical_system.split("-") if token}


def _presence_fraction(
    element_sets: list[set[str]],
    element: str,
) -> float:
    """Return fraction of molecules containing an element."""
    if not element_sets:
        raise ValueError("Element-set collection is empty.")

    return sum(element in elements for elements in element_sets) / len(element_sets)


def write_benchmark_parquet(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    """Write the compact attributed LIBE benchmark."""
    rows = sorted(
        rows,
        key=lambda row: str(row["neutral_id"]),
    )

    table = pa.Table.from_pylist(rows)

    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"source_dataset": b"LIBE",
            b"source_version": SOURCE_VERSION.encode(),
            b"source_doi": SOURCE_DOI.encode(),
            b"source_license": SOURCE_LICENSE.encode(),
            b"source_method": SOURCE_METHOD.encode(),
            b"energy_source_field": (b"thermo.raw.electronic_energy_Ha"),
            b"benchmark_role": (b"retrospective active-learning benchmark"),
        }
    )

    table = table.replace_schema_metadata(metadata)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        table,
        destination,
        compression="zstd",
    )


def write_domain_shift(
    benchmark_rows: list[dict[str, Any]],
    candidate_path: Path,
    csv_path: Path,
    json_path: Path,
) -> None:
    """Compare available shared chemistry descriptors."""
    candidate_table = pq.read_table(candidate_path)
    candidate_rows = candidate_table.to_pylist()

    if len(candidate_rows) != 60:
        raise ValueError(
            "Expected exactly 60 frozen candidate records; "
            f"found {len(candidate_rows)}."
        )

    benchmark_weights = [float(row["molecular_weight_g_mol"]) for row in benchmark_rows]
    candidate_weights = [_candidate_weight(row) for row in candidate_rows]

    benchmark_elements = [_benchmark_elements(row) for row in benchmark_rows]
    candidate_elements = [_candidate_elements(row) for row in candidate_rows]

    benchmark_weight_summary = _numeric_summary(benchmark_weights)
    candidate_weight_summary = _numeric_summary(candidate_weights)

    elements = ("C", "H", "N", "O", "S")

    benchmark_presence = {
        element: _presence_fraction(
            benchmark_elements,
            element,
        )
        for element in elements
    }

    candidate_presence = {
        element: _presence_fraction(
            candidate_elements,
            element,
        )
        for element in elements
    }

    summary = {
        "libe_benchmark_count": len(benchmark_rows),
        "candidate_pool_count": len(candidate_rows),
        "shared_numeric_descriptor": ("molecular_weight_g_mol"),
        "molecular_weight_g_mol": {
            "libe_benchmark": benchmark_weight_summary,
            "candidate_pool": candidate_weight_summary,
        },
        "element_presence_fraction": {
            "libe_benchmark": benchmark_presence,
            "candidate_pool": candidate_presence,
        },
        "sulfur_domain_shift": {
            "libe_fraction": benchmark_presence["S"],
            "candidate_fraction": candidate_presence["S"],
        },
        "interpretation": (
            "The retrospective LIBE benchmark is chemically "
            "out-of-domain relative to the sulfur-additive "
            "prospective pool. This benchmark therefore tests "
            "active-learning mechanics and uncertainty behavior; "
            "it is not treated as a directly representative "
            "training distribution for prospective candidates."
        ),
        "limitation": (
            "Only descriptors shared before Phase 7 are compared "
            "here. RDKit/xTB feature-space domain shift will become "
            "measurable after prospective features are generated."
        ),
    }

    rows: list[dict[str, Any]] = []

    for statistic in ("min", "median", "mean", "max"):
        libe_value = benchmark_weight_summary[statistic]
        candidate_value = candidate_weight_summary[statistic]

        rows.append(
            {
                "metric": (f"molecular_weight_g_mol_{statistic}"),
                "libe_benchmark": libe_value,
                "candidate_pool": candidate_value,
                "candidate_minus_libe": (candidate_value - libe_value),
            }
        )

    for element in elements:
        libe_value = benchmark_presence[element]
        candidate_value = candidate_presence[element]

        rows.append(
            {
                "metric": f"fraction_contains_{element}",
                "libe_benchmark": libe_value,
                "candidate_pool": candidate_value,
                "candidate_minus_libe": (candidate_value - libe_value),
            }
        )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "libe_benchmark",
                "candidate_pool",
                "candidate_minus_libe",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Freeze benchmark and create domain-shift outputs."""
    args = parse_args()

    benchmark_rows = json.loads(args.selection.read_text(encoding="utf-8"))

    if not isinstance(benchmark_rows, list):
        raise ValueError("Benchmark selection must be a JSON list.")

    if len(benchmark_rows) != 133:
        raise ValueError(
            f"Expected 133 frozen LIBE benchmark records; found {len(benchmark_rows)}."
        )

    write_benchmark_parquet(
        benchmark_rows,
        args.benchmark_output,
    )

    write_domain_shift(
        benchmark_rows,
        args.candidate_pool,
        args.domain_shift_csv,
        args.domain_shift_json,
    )

    print(
        f"Wrote {len(benchmark_rows)} LIBE benchmark records to {args.benchmark_output}"
    )
    print(f"Wrote domain-shift report to {args.domain_shift_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
