"""Quality control for Phase 8 DFT-derived features."""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("results/phase8_v1/dft_features_v1.csv")
OUTPUT_PATH = Path("results/phase8_v1/dft_features_qc_v1.csv")

EXPECTED_COUNT = 56
IDEAL_DOUBLET_S2 = 0.75

# Heuristic review threshold only; this does NOT automatically exclude a candidate.
S2_REVIEW_THRESHOLD = 0.05

REQUIRED_COLUMNS = [
    "candidate_id",
    "energy_neutral_eh",
    "energy_oxidized_eh",
    "energy_reduced_eh",
    "vertical_ionization_energy_ev",
    "vertical_electron_attachment_delta_e_ev",
    "vertical_electron_affinity_ev",
    "oxidized_s2",
    "reduced_s2",
    "functional",
    "basis",
    "pcm_model",
    "dielectric",
]


def iqr_flags(series: pd.Series) -> pd.Series:
    """Return statistical IQR review flags without excluding observations."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (series < lower) | (series > upper)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise RuntimeError(f"Missing required columns: {sorted(missing_columns)}")

    if len(df) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT} DFT-complete candidates, found {len(df)}."
        )

    if df["candidate_id"].duplicated().any():
        duplicates = df.loc[
            df["candidate_id"].duplicated(),
            "candidate_id",
        ].tolist()
        raise RuntimeError(f"Duplicate candidate IDs: {duplicates}")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise RuntimeError("Missing values detected in required DFT fields.")

    # All candidates must use one consistent screening method.
    for column in [
        "functional",
        "basis",
        "pcm_model",
        "dielectric",
    ]:
        if df[column].nunique() != 1:
            raise RuntimeError(
                f"Inconsistent Phase 8 method settings in column: {column}"
            )

    # Our two electron-attachment conventions must be exact sign opposites.
    sign_consistent = (
        df["vertical_electron_attachment_delta_e_ev"]
        + df["vertical_electron_affinity_ev"]
    ).abs() < 1e-8

    if not sign_consistent.all():
        raise RuntimeError(
            "Electron-affinity and attachment-energy signs are inconsistent."
        )

    # Ionization should require positive energy for this screening definition.
    df["ionization_sign_flag"] = df["vertical_ionization_energy_ev"] <= 0.0

    # UKS doublet spin-quality diagnostics.
    df["oxidized_s2_deviation"] = (df["oxidized_s2"] - IDEAL_DOUBLET_S2).abs()

    df["reduced_s2_deviation"] = (df["reduced_s2"] - IDEAL_DOUBLET_S2).abs()

    df["spin_review_flag"] = (df["oxidized_s2_deviation"] > S2_REVIEW_THRESHOLD) | (
        df["reduced_s2_deviation"] > S2_REVIEW_THRESHOLD
    )

    # Statistical review only — these are not automatic exclusions.
    df["ionization_iqr_flag"] = iqr_flags(df["vertical_ionization_energy_ev"])
    df["electron_affinity_iqr_flag"] = iqr_flags(df["vertical_electron_affinity_ev"])

    df["manual_review_flag"] = (
        df["ionization_sign_flag"]
        | df["spin_review_flag"]
        | df["ionization_iqr_flag"]
        | df["electron_affinity_iqr_flag"]
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Candidates checked: {len(df)}")

    print("\nMethod:")
    print("Functional:", df["functional"].iloc[0])
    print("Basis:", df["basis"].iloc[0])
    print("PCM:", df["pcm_model"].iloc[0])
    print("Dielectric:", df["dielectric"].iloc[0])

    print("\nVertical ionization energy (eV):")
    print(df["vertical_ionization_energy_ev"].describe())

    print("\nVertical electron affinity (eV):")
    print(df["vertical_electron_affinity_ev"].describe())

    print("\nSpin diagnostics:")
    print(
        df[
            [
                "oxidized_s2",
                "reduced_s2",
            ]
        ].describe()
    )

    flagged = df[df["manual_review_flag"]]

    print(f"\nCandidates flagged for manual review: {len(flagged)}")

    if not flagged.empty:
        print(
            flagged[
                [
                    "candidate_id",
                    "vertical_ionization_energy_ev",
                    "vertical_electron_affinity_ev",
                    "oxidized_s2",
                    "reduced_s2",
                    "ionization_sign_flag",
                    "spin_review_flag",
                    "ionization_iqr_flag",
                    "electron_affinity_iqr_flag",
                ]
            ].to_string(index=False)
        )

    print(f"\nQC output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
