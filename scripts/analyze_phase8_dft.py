"""Analyze and visualize Phase 8 DFT screening descriptors."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = Path("data/processed/phase8_features_v1.csv")
OUTPUT_DIR = Path("results/phase8_v1/plots")
CORRELATION_PATH = Path("results/phase8_v1/dft_correlations_v1.csv")


def save_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    """Create scatter plot and label manual-review candidates."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(df[x], df[y], alpha=0.8)

    flagged = df[df["manual_review_flag"]]

    for _, row in flagged.iterrows():
        ax.annotate(
            row["candidate_id"],
            (row[x], row[y]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # DFT redox descriptor distributions.
    for column, xlabel, filename in [
        (
            "vertical_ionization_energy_ev",
            "Vertical ionization energy (eV)",
            "vertical_ionization_energy.png",
        ),
        (
            "vertical_electron_affinity_ev",
            "Vertical electron affinity (eV)",
            "vertical_electron_affinity.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(df[column], bins=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Candidate count")
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close(fig)

    # Cheap xTB orbital descriptors versus higher-fidelity DFT ΔSCF values.
    save_scatter(
        df,
        x="xtb_homo_ev",
        y="vertical_ionization_energy_ev",
        xlabel="xTB HOMO energy (eV)",
        ylabel="DFT vertical ionization energy (eV)",
        filename="xtb_homo_vs_dft_ionization.png",
    )

    save_scatter(
        df,
        x="xtb_lumo_ev",
        y="vertical_electron_affinity_ev",
        xlabel="xTB LUMO energy (eV)",
        ylabel="DFT vertical electron affinity (eV)",
        filename="xtb_lumo_vs_dft_electron_affinity.png",
    )

    comparisons = [
        (
            "xtb_homo_ev",
            "vertical_ionization_energy_ev",
        ),
        (
            "xtb_lumo_ev",
            "vertical_electron_affinity_ev",
        ),
        (
            "xtb_homo_lumo_gap_ev",
            "vertical_ionization_energy_ev",
        ),
        (
            "xtb_li_delta_energy_ev",
            "vertical_electron_affinity_ev",
        ),
    ]

    rows = []

    for x, y in comparisons:
        rows.append(
            {
                "descriptor_x": x,
                "descriptor_y": y,
                "pearson_r": df[x].corr(df[y], method="pearson"),
                "spearman_rho": df[x].corr(df[y], method="spearman"),
            }
        )

    correlations = pd.DataFrame(rows)
    correlations.to_csv(CORRELATION_PATH, index=False)

    print("\nCorrelations:")
    print(correlations.to_string(index=False))

    print("\nLowest vertical ionization energies:")
    print(
        df.nsmallest(5, "vertical_ionization_energy_ev")[
            [
                "candidate_id",
                "preferred_name",
                "vertical_ionization_energy_ev",
            ]
        ].to_string(index=False)
    )

    print("\nHighest vertical electron affinities:")
    print(
        df.nlargest(5, "vertical_electron_affinity_ev")[
            [
                "candidate_id",
                "preferred_name",
                "vertical_electron_affinity_ev",
            ]
        ].to_string(index=False)
    )

    print(f"\nPlots: {OUTPUT_DIR}")
    print(f"Correlations: {CORRELATION_PATH}")


if __name__ == "__main__":
    main()
