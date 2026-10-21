"""Transparent descriptor-based ranking of the frozen CEI candidate pool."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

OUTPUT_FIELDS = [
    "candidate_id",
    "preferred_name",
    "canonical_smiles",
    "molecular_formula",
    "candidate_family",
    "dft_status",
    "unconverged_state",
    "vertical_ionization_energy_ev",
    "vertical_electron_affinity_ev",
    "oxidation_propensity_rank",
    "electron_attachment_rank",
    "oxidation_resistance_rank",
    "dual_redox_rank_sum",
    "dual_redox_priority_rank",
    "xtb_li_delta_energy_ev",
    "manual_review_flag",
    "spin_review_flag",
    "electron_affinity_iqr_flag",
    "qc_li_connectivity_change_count",
    "qc_li_fragmentation_count",
]


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read candidate rows with strict ID and SMILES uniqueness."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        rows = list(reader)

    ids = [row["candidate_id"] for row in rows]
    smiles = [row["canonical_smiles"] for row in rows]

    if (
        not all(ids)
        or len(ids) != len(set(ids))
        or not all(smiles)
        or len(smiles) != len(set(smiles))
    ):
        raise ValueError(f"Duplicate or blank identity: {path}")

    return {row["candidate_id"]: row for row in rows}


def finite_number(row: dict[str, str], field: str) -> float:
    """Require a finite numerical screening descriptor."""
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value: {row['candidate_id']} / {field}")
    return value


def ordinal_ranks(
    rows: list[dict[str, str]],
    field: str,
    descending: bool = False,
) -> dict[str, int]:
    """Rank by physical descriptor, breaking exact ties by ID."""
    ordered = sorted(
        rows,
        key=lambda row: (
            -finite_number(row, field) if descending else finite_number(row, field),
            row["candidate_id"],
        ),
    )
    return {row["candidate_id"]: index for index, row in enumerate(ordered, start=1)}


def build_ranking(
    config_path: Path,
) -> tuple[list[dict], dict]:
    """Preserve the 60-molecule pool and rank 56 DFT-complete rows."""
    config_path = config_path.resolve()
    root = config_path.parent.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    paths = {
        name: root / cfg[name]
        for name in (
            "prospective_dataset",
            "dft_dataset",
            "dft_exclusions",
        )
    }

    profile = cfg["ranking"]
    if (
        profile["profile"] != "exploratory_dual_redox_propensity"
        or profile["oxidation_column"] != "vertical_ionization_energy_ev"
        or profile["oxidation_direction"] != "ascending"
        or profile["electron_attachment_column"] != "vertical_electron_affinity_ev"
        or profile["electron_attachment_direction"] != "descending"
        or profile["composite"] != "equal_ordinal_rank_sum"
        or profile["composite_tie_break"] != "oxidation_rank_then_candidate_id"
        or cfg["li_association"]["column"] != "xtb_li_delta_energy_ev"
        or cfg["li_association"]["policy"] != "report_only"
        or cfg["missing_dft"]["policy"] != "unranked_retain_in_frozen_pool"
        or cfg["qc"]["retain_flagged_candidates"] is not True
        or cfg["qc"]["preserve_dft_review_flags"] is not True
        or cfg["qc"]["preserve_li_connectivity_flags"] is not True
    ):
        raise ValueError("Unsupported ranking policy")

    prospective = load_rows(paths["prospective_dataset"])
    dft = load_rows(paths["dft_dataset"])

    exclusion_policy = yaml.safe_load(
        paths["dft_exclusions"].read_text(encoding="utf-8")
    )
    exclusions = exclusion_policy["candidates"]

    missing = set(prospective) - set(dft)

    if (
        len(prospective) != 60
        or len(dft) != 56
        or not set(dft).issubset(prospective)
        or missing != set(exclusions)
    ):
        raise ValueError("Frozen pool or DFT-exclusion mismatch")

    for candidate_id, row in dft.items():
        original = prospective[candidate_id]

        for field in (
            "canonical_smiles",
            "preferred_name",
            "candidate_family",
            "molecular_formula",
        ):
            if row[field] != original[field]:
                raise ValueError(f"Candidate identity mismatch: {candidate_id}")

    complete = list(dft.values())
    ie_field = profile["oxidation_column"]
    ea_field = profile["electron_attachment_column"]

    ie_ranks = ordinal_ranks(complete, ie_field)
    ea_ranks = ordinal_ranks(complete, ea_field, descending=True)
    resistance_ranks = ordinal_ranks(complete, ie_field, descending=True)

    combined = sorted(
        dft,
        key=lambda cid: (
            ie_ranks[cid] + ea_ranks[cid],
            ie_ranks[cid],
            cid,
        ),
    )
    combined_ranks = {cid: index for index, cid in enumerate(combined, start=1)}

    output = []

    for cid in sorted(prospective):
        original = prospective[cid]
        completed = cid in dft
        row = dft.get(cid, {})
        # All molecular identities and available Phase 7 values survive.
        record = {
            "candidate_id": cid,
            "preferred_name": original["preferred_name"],
            "canonical_smiles": original["canonical_smiles"],
            "molecular_formula": original["molecular_formula"],
            "candidate_family": original["candidate_family"],
            "dft_status": ("complete" if completed else "incomplete"),
            "unconverged_state": (
                "" if completed else exclusions[cid]["unconverged_state"]
            ),
            "vertical_ionization_energy_ev": (
                finite_number(row, ie_field) if completed else ""
            ),
            "vertical_electron_affinity_ev": (
                finite_number(row, ea_field) if completed else ""
            ),
            "oxidation_propensity_rank": (ie_ranks[cid] if completed else ""),
            "electron_attachment_rank": (ea_ranks[cid] if completed else ""),
            "oxidation_resistance_rank": (resistance_ranks[cid] if completed else ""),
            "dual_redox_rank_sum": (ie_ranks[cid] + ea_ranks[cid] if completed else ""),
            "dual_redox_priority_rank": (combined_ranks[cid] if completed else ""),
            "xtb_li_delta_energy_ev": finite_number(
                original, cfg["li_association"]["column"]
            ),
            "manual_review_flag": (row["manual_review_flag"] if completed else ""),
            "spin_review_flag": (row["spin_review_flag"] if completed else ""),
            "electron_affinity_iqr_flag": (
                row["electron_affinity_iqr_flag"] if completed else ""
            ),
            "qc_li_connectivity_change_count": original[
                "qc_li_connectivity_change_count"
            ],
            "qc_li_fragmentation_count": original["qc_li_fragmentation_count"],
        }
        output.append(record)

    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    hashes["ranking_config"] = hashlib.sha256(config_path.read_bytes()).hexdigest()

    metadata = {
        "version": 1,
        "phase": 9,
        "population": {
            "prospective": 60,
            "dft_complete": 56,
            "dft_incomplete": sorted(missing),
        },
        "scoring": profile,
        "input_sha256": hashes,
        "interpretation": (
            "Ranks describe computed vertical redox tendencies. "
            "They do not establish CEI performance or experimental "
            "electrochemical potentials. The combined rank uses "
            "equal-weight ordinal positions, not calibrated physics."
        ),
        "li_association_policy": "reported, not scored",
        "qc_policy": "flags preserved; no automatic exclusions",
    }

    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase9_ranking_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase9_v1"),
    )
    args = parser.parse_args()

    rows, metadata = build_ranking(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output = args.output_dir / "species_ranking_v1.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = args.output_dir / "species_ranking_metadata_v1.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ranked = sorted(
        (r for r in rows if r["dft_status"] == "complete"),
        key=lambda r: r["dual_redox_priority_rank"],
    )

    print("PHASE 9 SPECIES RANKING COMPLETE")
    print(f"Total candidates: {len(rows)}")
    print(f"DFT-ranked: {len(ranked)}")
    print(f"DFT-incomplete: {len(rows) - len(ranked)}")
    print("\nTOP 10 — EXPLORATORY DUAL-REDOX PRIORITY")
    for row in ranked[:10]:
        print(
            f"{row['dual_redox_priority_rank']:>2}. "
            f"{row['candidate_id']} "
            f"{row['preferred_name'][:27]:<27} "
            f"IE={row['vertical_ionization_energy_ev']:.3f} "
            f"EA={row['vertical_electron_affinity_ev']:.3f} "
            f"review={row['manual_review_flag']}"
        )


if __name__ == "__main__":
    main()
