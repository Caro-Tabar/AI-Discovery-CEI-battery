"""Generate Phase 7 feature-distribution and anomaly QC outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = ROOT / "data/processed/phase7_features_v1.parquet"

LI_SITE_PATH = ROOT / "results/xtb_li_v1/li_site_attempts.parquet"

OUTPUT_DIRECTORY = ROOT / "results/phase7_v1/qc"


QC_FEATURES = (
    "rdkit_tpsa_a2",
    "rdkit_rotatable_bond_count",
    "xtb_homo_ev",
    "xtb_lumo_ev",
    "xtb_homo_lumo_gap_ev",
    "xtb_dipole_debye",
    "xtb_polarizability_au",
    "xtb_li_delta_energy_ev",
)


def _iqr_outliers(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return review-only IQR outlier flags."""

    rows: list[dict[str, Any]] = []

    for column in QC_FEATURES:
        series = frame[column].astype(float)

        q1 = float(series.quantile(0.25))

        q3 = float(series.quantile(0.75))

        iqr = q3 - q1

        if iqr == 0.0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (series < lower) | (series > upper)

        for record in frame.loc[
            mask,
            [
                "candidate_id",
                "preferred_name",
                "candidate_family",
                column,
            ],
        ].to_dict(orient="records"):
            rows.append(
                {
                    "candidate_id": (record["candidate_id"]),
                    "preferred_name": (record["preferred_name"]),
                    "candidate_family": (record["candidate_family"]),
                    "descriptor": column,
                    "value": float(record[column]),
                    "q1": q1,
                    "q3": q3,
                    "lower_fence": lower,
                    "upper_fence": upper,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "preferred_name",
            "candidate_family",
            "descriptor",
            "value",
            "q1",
            "q3",
            "lower_fence",
            "upper_fence",
        ],
    )


def _save_histogram(
    frame: pd.DataFrame,
    column: str,
    title: str,
    x_label: str,
    path: Path,
) -> None:
    """Save one descriptor histogram."""

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.hist(
        frame[column].astype(float),
        bins=12,
    )

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Candidate count")

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
    )

    plt.close(figure)


def _save_scatter(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Save HOMO versus Li-association scatter QC."""

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.scatter(
        frame["xtb_homo_ev"],
        frame["xtb_li_delta_energy_ev"],
    )

    axis.set_xlabel("GFN2-xTB HOMO energy (eV)")

    axis.set_ylabel("GFN2-xTB Li+ association proxy (eV)")

    axis.set_title("Electronic descriptor vs Li+ association")

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
    )

    plt.close(figure)


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)

    li_sites = pd.read_parquet(LI_SITE_PATH)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    outliers = _iqr_outliers(features)

    outliers.to_csv(
        OUTPUT_DIRECTORY / "iqr_outliers.csv",
        index=False,
    )

    connectivity_anomalies = (
        li_sites.loc[
            (li_sites["status"].astype(str).ne("success"))
            | (~li_sites["connectivity_preserved"].fillna(False).astype(bool))
            | (li_sites["fragmented"].fillna(False).astype(bool))
        ]
        .copy()
        .sort_values(
            [
                "candidate_id",
                "site_rank",
            ]
        )
    )

    connectivity_anomalies.to_csv(
        OUTPUT_DIRECTORY / "connectivity_anomalies.csv",
        index=False,
    )

    ranges = {
        column: {
            "minimum": float(features[column].min()),
            "maximum": float(features[column].max()),
            "mean": float(features[column].mean()),
        }
        for column in QC_FEATURES
    }

    summary = {
        "candidate_count": len(features),
        "unique_candidate_count": int(features["candidate_id"].nunique()),
        "missing_value_count": int(features.isna().sum().sum()),
        "iqr_outlier_flag_count": len(outliers),
        "connectivity_anomaly_count": len(connectivity_anomalies),
        "descriptor_ranges": ranges,
    }

    (OUTPUT_DIRECTORY / "qc_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _save_histogram(
        features,
        "xtb_homo_lumo_gap_ev",
        "GFN2-xTB HOMO-LUMO gap",
        "Gap (eV)",
        OUTPUT_DIRECTORY / "gap_histogram.png",
    )

    _save_histogram(
        features,
        "xtb_dipole_debye",
        "GFN2-xTB molecular dipole",
        "Dipole (Debye)",
        OUTPUT_DIRECTORY / "dipole_histogram.png",
    )

    _save_histogram(
        features,
        "xtb_polarizability_au",
        "GFN2-xTB molecular polarizability",
        "Polarizability (a.u.)",
        OUTPUT_DIRECTORY / "polarizability_histogram.png",
    )

    _save_histogram(
        features,
        "xtb_li_delta_energy_ev",
        "GFN2-xTB Li+ association proxy",
        "Association energy proxy (eV)",
        OUTPUT_DIRECTORY / "li_association_histogram.png",
    )

    _save_histogram(
        features,
        "rdkit_tpsa_a2",
        "RDKit topological polar surface area",
        "TPSA (angstrom^2)",
        OUTPUT_DIRECTORY / "tpsa_histogram.png",
    )

    _save_scatter(
        features,
        OUTPUT_DIRECTORY / "homo_vs_li_association.png",
    )

    print(f"Candidates: {len(features)}")

    print(f"Missing values: {summary['missing_value_count']}")

    print(f"IQR review flags: {len(outliers)}")

    print(f"Connectivity anomalies: {len(connectivity_anomalies)}")

    if not connectivity_anomalies.empty:
        print("\nConnectivity anomaly sites:")

        print(
            connectivity_anomalies[
                [
                    "candidate_id",
                    "site_rank",
                    "donor_element",
                    "removed_bond_count",
                    "added_bond_count",
                    "work_directory",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
