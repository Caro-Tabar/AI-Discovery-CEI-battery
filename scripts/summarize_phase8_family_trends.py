"""Summarize Phase 8 DFT descriptors by chemical family."""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/phase8_features_v1.csv")
OUTPUT_PATH = Path("results/phase8_v1/dft_family_summary_v1.csv")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    summary = (
        df.groupby("candidate_family")
        .agg(
            candidate_count=("candidate_id", "size"),
            ionization_mean_ev=(
                "vertical_ionization_energy_ev",
                "mean",
            ),
            ionization_median_ev=(
                "vertical_ionization_energy_ev",
                "median",
            ),
            ionization_min_ev=(
                "vertical_ionization_energy_ev",
                "min",
            ),
            ionization_max_ev=(
                "vertical_ionization_energy_ev",
                "max",
            ),
            electron_affinity_mean_ev=(
                "vertical_electron_affinity_ev",
                "mean",
            ),
            electron_affinity_median_ev=(
                "vertical_electron_affinity_ev",
                "median",
            ),
            electron_affinity_min_ev=(
                "vertical_electron_affinity_ev",
                "min",
            ),
            electron_affinity_max_ev=(
                "vertical_electron_affinity_ev",
                "max",
            ),
            xtb_li_association_mean_ev=(
                "xtb_li_delta_energy_ev",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("ionization_mean_ev")
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("\n=== DFT FAMILY SUMMARY ===")
    print(summary.round(3).to_string(index=False))

    print("\n=== EASIEST VERTICAL OXIDATION ===")
    print(
        df.nsmallest(10, "vertical_ionization_energy_ev")[
            [
                "candidate_id",
                "preferred_name",
                "candidate_family",
                "vertical_ionization_energy_ev",
                "vertical_electron_affinity_ev",
                "manual_review_flag",
            ]
        ].to_string(index=False)
    )

    print("\n=== HIGHEST ELECTRON AFFINITY ===")
    print(
        df.nlargest(10, "vertical_electron_affinity_ev")[
            [
                "candidate_id",
                "preferred_name",
                "candidate_family",
                "vertical_ionization_energy_ev",
                "vertical_electron_affinity_ev",
                "manual_review_flag",
            ]
        ].to_string(index=False)
    )

    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
