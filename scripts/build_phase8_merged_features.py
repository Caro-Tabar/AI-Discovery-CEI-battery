"""Merge Phase 7 features with validated Phase 8 DFT descriptors."""

from pathlib import Path

import pandas as pd

PHASE7_PATH = Path("data/processed/phase7_features_v1.csv")
DFT_PATH = Path("results/phase8_v1/dft_features_qc_v1.csv")

OUTPUT_CSV = Path("data/processed/phase8_features_v1.csv")
OUTPUT_PARQUET = Path("data/processed/phase8_features_v1.parquet")

EXPECTED_PHASE7_COUNT = 60
EXPECTED_PHASE8_COUNT = 56

EXPECTED_DFT_EXCLUSIONS = {
    "CEI-0021",
    "CEI-0051",
    "CEI-0059",
    "CEI-0060",
}


def main() -> None:
    phase7 = pd.read_csv(PHASE7_PATH)
    dft = pd.read_csv(DFT_PATH)

    if len(phase7) != EXPECTED_PHASE7_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PHASE7_COUNT} Phase 7 candidates, found {len(phase7)}."
        )

    if len(dft) != EXPECTED_PHASE8_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PHASE8_COUNT} DFT-complete candidates, "
            f"found {len(dft)}."
        )

    if phase7["candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate candidate IDs in Phase 7 features.")

    if dft["candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate candidate IDs in Phase 8 DFT features.")

    phase7_ids = set(phase7["candidate_id"])
    dft_ids = set(dft["candidate_id"])

    missing_from_dft = phase7_ids - dft_ids
    unexpected_dft = dft_ids - phase7_ids

    if missing_from_dft != EXPECTED_DFT_EXCLUSIONS:
        raise RuntimeError(
            "Unexpected Phase 8 DFT exclusions.\n"
            f"Expected: {sorted(EXPECTED_DFT_EXCLUSIONS)}\n"
            f"Observed: {sorted(missing_from_dft)}"
        )

    if unexpected_dft:
        raise RuntimeError(
            "Phase 8 contains candidate IDs absent from Phase 7: "
            f"{sorted(unexpected_dft)}"
        )

    merged = phase7.merge(
        dft,
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_PHASE8_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PHASE8_COUNT} merged candidates, found {len(merged)}."
        )

    # Preserve Phase 7 candidate ordering.
    order = {
        candidate_id: index for index, candidate_id in enumerate(phase7["candidate_id"])
    }

    merged["_candidate_order"] = merged["candidate_id"].map(order)

    merged = (
        merged.sort_values("_candidate_order")
        .drop(columns="_candidate_order")
        .reset_index(drop=True)
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(OUTPUT_CSV, index=False)
    merged.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"Phase 7 candidates: {len(phase7)}")
    print(f"DFT-complete candidates: {len(dft)}")
    print(f"Merged candidates: {len(merged)}")
    print(f"Merged columns: {len(merged.columns)}")

    print("\nExcluded from DFT-complete dataset:")
    excluded = phase7[phase7["candidate_id"].isin(EXPECTED_DFT_EXCLUSIONS)][
        [
            "candidate_id",
            "preferred_name",
            "molecular_formula",
            "candidate_family",
        ]
    ]

    print(excluded.to_string(index=False))

    print(f"\nCSV: {OUTPUT_CSV}")
    print(f"Parquet: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
