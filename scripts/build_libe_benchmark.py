#!/usr/bin/env python3
"""Build the filtered retrospective LIBE benchmark."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cei_scout.public_data.libe import load_libe_records
from cei_scout.public_data.libe_benchmark import (
    _charge_index,
    build_redox_record,
    matching_charge_state,
    molecular_weight_g_mol,
    neutral_parent_rejection_reason,
    scaffold_hash,
    select_benchmark,
)

SOURCE = Path("data/external/libe/v2/libe.json")

AUDIT_PATH = Path("results/reports/libe_filter_audit_v0.csv")
SUMMARY_PATH = Path("results/reports/libe_filter_summary_v0.json")
MATCHED_PATH = Path("data/interim/libe_matched_v0.json")
SELECTION_PATH = Path("results/manifests/libe_benchmark_selection_v0.json")


def main() -> int:
    """Build and audit the retrospective benchmark."""
    records = load_libe_records(SOURCE)

    index = _charge_index(records)

    neutral_records = [record for record in records if record.get("charge") == 0]

    neutral_lookup: dict[str, dict[str, Any]] = {
        str(record["molecule_id"]): record for record in neutral_records
    }

    audit_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []

    for neutral in neutral_records:
        neutral_id = str(neutral["molecule_id"])

        reason = neutral_parent_rejection_reason(neutral)

        try:
            mass = molecular_weight_g_mol(neutral)
        except ValueError:
            mass = None

        if reason is not None:
            audit_rows.append(
                {
                    "neutral_id": neutral_id,
                    "status": "rejected",
                    "reason": reason,
                    "molecular_weight_g_mol": mass,
                    "cation_id": "",
                    "anion_id": "",
                }
            )
            continue

        cation, cation_reason = matching_charge_state(
            neutral,
            index,
            1,
        )

        if cation_reason is not None:
            audit_rows.append(
                {
                    "neutral_id": neutral_id,
                    "status": "rejected",
                    "reason": cation_reason,
                    "molecular_weight_g_mol": mass,
                    "cation_id": "",
                    "anion_id": "",
                }
            )
            continue

        anion, anion_reason = matching_charge_state(
            neutral,
            index,
            -1,
        )

        if anion_reason is not None:
            audit_rows.append(
                {
                    "neutral_id": neutral_id,
                    "status": "rejected",
                    "reason": anion_reason,
                    "molecular_weight_g_mol": mass,
                    "cation_id": str(cation["molecule_id"]),
                    "anion_id": "",
                }
            )
            continue

        assert cation is not None
        assert anion is not None

        try:
            matched = build_redox_record(
                neutral,
                cation,
                anion,
            )
        except ValueError:
            audit_rows.append(
                {
                    "neutral_id": neutral_id,
                    "status": "rejected",
                    "reason": "missing_or_invalid_energy",
                    "molecular_weight_g_mol": mass,
                    "cation_id": str(cation["molecule_id"]),
                    "anion_id": str(anion["molecule_id"]),
                }
            )
            continue

        matched["family_stratum"] = str(neutral["chemical_system"])
        matched["scaffold_hash"] = scaffold_hash(neutral)

        matched_rows.append(matched)

        audit_rows.append(
            {
                "neutral_id": neutral_id,
                "status": "accepted",
                "reason": "",
                "molecular_weight_g_mol": mass,
                "cation_id": str(cation["molecule_id"]),
                "anion_id": str(anion["molecule_id"]),
            }
        )

    selected = select_benchmark(
        matched_rows,
        neutral_lookup,
        target_size=300,
    )

    reason_counts = Counter(
        row["reason"] for row in audit_rows if row["status"] == "rejected"
    )

    summary = {
        "source_record_count": len(records),
        "neutral_record_count": len(neutral_records),
        "fully_matched_record_count": len(matched_rows),
        "benchmark_target_size": 300,
        "benchmark_selected_count": len(selected),
        "used_all_eligible_records": len(matched_rows) < 300,
        "rejection_counts": dict(sorted(reason_counts.items())),
        "family_counts_matched": dict(
            sorted(Counter(row["family_stratum"] for row in matched_rows).items())
        ),
        "family_counts_selected": dict(
            sorted(Counter(row["family_stratum"] for row in selected).items())
        ),
        "unique_scaffolds_matched": len({row["scaffold_hash"] for row in matched_rows}),
        "unique_scaffolds_selected": len({row["scaffold_hash"] for row in selected}),
    }

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATCHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "neutral_id",
                "status",
                "reason",
                "molecular_weight_g_mol",
                "cation_id",
                "anion_id",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    MATCHED_PATH.write_text(
        json.dumps(
            matched_rows,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    SELECTION_PATH.write_text(
        json.dumps(
            selected,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
