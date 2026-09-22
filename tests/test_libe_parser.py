"""Tests for LIBE public-data ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from cei_scout.public_data.libe import (
    load_libe_records,
    validate_libe_record,
)

FIXTURE = Path("tests/fixtures/libe_v2_minimal.json")


def test_load_libe_records_preserves_source_ids() -> None:
    """The parser must retain LIBE's original molecule identifiers."""
    raw = cast(
        list[dict[str, Any]],
        json.loads(FIXTURE.read_text(encoding="utf-8")),
    )

    parsed = load_libe_records(FIXTURE)

    assert [record["molecule_id"] for record in parsed] == [
        record["molecule_id"] for record in raw
    ]


def test_load_libe_records_preserves_scientific_fields() -> None:
    """Ingestion must not modify charge, spin, geometry, or thermo."""
    raw = cast(
        list[dict[str, Any]],
        json.loads(FIXTURE.read_text(encoding="utf-8")),
    )

    parsed = load_libe_records(FIXTURE)

    for source, result in zip(raw, parsed, strict=True):
        assert result["charge"] == source["charge"]
        assert result["spin_multiplicity"] == source["spin_multiplicity"]
        assert result["species"] == source["species"]
        assert result["xyz"] == source["xyz"]
        assert result["thermo"] == source["thermo"]


def test_validate_libe_record_rejects_missing_identifier() -> None:
    """Records without their upstream identity must fail ingestion."""
    raw = cast(
        list[dict[str, Any]],
        json.loads(FIXTURE.read_text(encoding="utf-8")),
    )
    invalid = dict(raw[0])
    invalid.pop("molecule_id")

    with pytest.raises(ValueError, match="missing required fields"):
        validate_libe_record(invalid)


def test_validate_libe_record_rejects_geometry_mismatch() -> None:
    """Species and coordinate arrays must stay aligned."""
    raw = cast(
        list[dict[str, Any]],
        json.loads(FIXTURE.read_text(encoding="utf-8")),
    )
    invalid = dict(raw[0])
    invalid["xyz"] = invalid["xyz"][:-1]

    with pytest.raises(ValueError, match="species and xyz lengths differ"):
        validate_libe_record(invalid)
