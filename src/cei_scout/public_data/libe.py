"""Parsing utilities for the pinned LIBE public dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

LIBE_ID_PATTERN = re.compile(r"^libe-\d{6}$")

REQUIRED_FIELDS = frozenset(
    {
        "molecule_id",
        "charge",
        "spin_multiplicity",
        "species",
        "xyz",
        "composition",
        "elements",
        "chemical_system",
        "formula_alphabetical",
        "thermo",
    }
)


def _records_from_payload(payload: object) -> list[object]:
    """Return raw records from the supported LIBE JSON containers."""
    if isinstance(payload, list):
        return cast(list[object], payload)

    if isinstance(payload, dict):
        return list(payload.values())

    raise ValueError("LIBE JSON must contain a top-level list or object.")


def validate_libe_record(record: dict[str, Any]) -> None:
    """Validate ingestion-level LIBE invariants without altering values."""
    missing = REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError(f"LIBE record is missing required fields: {sorted(missing)}")

    molecule_id = record["molecule_id"]
    if not isinstance(molecule_id, str):
        raise ValueError("LIBE molecule_id must be a string.")

    if LIBE_ID_PATTERN.fullmatch(molecule_id) is None:
        raise ValueError(f"Invalid LIBE molecule_id format: {molecule_id!r}")

    if not isinstance(record["charge"], int):
        raise ValueError(f"{molecule_id}: charge must be an integer.")

    if not isinstance(record["spin_multiplicity"], int):
        raise ValueError(f"{molecule_id}: spin_multiplicity must be an integer.")

    species = record["species"]
    xyz = record["xyz"]

    if not isinstance(species, list):
        raise ValueError(f"{molecule_id}: species must be a list.")

    if not isinstance(xyz, list):
        raise ValueError(f"{molecule_id}: xyz must be a list.")

    if len(species) != len(xyz):
        raise ValueError(f"{molecule_id}: species and xyz lengths differ.")

    if not isinstance(record["thermo"], dict):
        raise ValueError(f"{molecule_id}: thermo must be a JSON object.")


def load_libe_records(path: Path) -> list[dict[str, Any]]:
    """Load LIBE records while preserving source identifiers and values."""
    with path.open(encoding="utf-8") as handle:
        payload: object = json.load(handle)

    raw_records = _records_from_payload(payload)

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"LIBE record {index} is not a JSON object.")

        record = cast(dict[str, Any], raw_record)
        validate_libe_record(record)

        molecule_id = cast(str, record["molecule_id"])

        if molecule_id in seen_ids:
            raise ValueError(f"Duplicate LIBE molecule_id: {molecule_id}")

        seen_ids.add(molecule_id)
        records.append(record)

    return records
