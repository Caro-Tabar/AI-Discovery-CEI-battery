"""Build the frozen Phase 7 feature and geometry-provenance dataset."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import rdBase

from cei_scout.preprocessing.phase7_features import (
    build_phase7_feature_table,
    freeze_geometry_provenance,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_PATH = ROOT / "data/processed/candidates_v1.parquet"

XTB_PATH = ROOT / "results/xtb_neutral_v1/xtb_descriptors.parquet"

XTB_ATTEMPTS_PATH = ROOT / "results/xtb_neutral_v1/xtb_attempts.parquet"

LI_ASSOCIATION_PATH = ROOT / "results/xtb_li_v1/li_association.parquet"

LI_SITES_PATH = ROOT / "results/xtb_li_v1/li_site_attempts.parquet"

DATA_OUTPUT_DIRECTORY = ROOT / "data/processed"

RESULT_OUTPUT_DIRECTORY = ROOT / "results/phase7_v1"


def _xtb_version() -> str:
    """Return the installed xTB semantic version."""

    result = subprocess.run(
        ["xtb", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )

    text = result.stdout + "\n" + result.stderr

    match = re.search(
        r"xtb version\s+([0-9.]+)",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return "unparsed"

    return match.group(1)


def main() -> None:
    candidates = pd.read_parquet(CANDIDATES_PATH)

    xtb = pd.read_parquet(XTB_PATH)

    neutral_attempts = pd.read_parquet(XTB_ATTEMPTS_PATH)

    li_association = pd.read_parquet(LI_ASSOCIATION_PATH)

    li_sites = pd.read_parquet(LI_SITES_PATH)

    features = build_phase7_feature_table(
        candidates=candidates,
        xtb=xtb,
        li_association=li_association,
        li_sites=li_sites,
        expected_count=60,
    )

    DATA_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_parquet = DATA_OUTPUT_DIRECTORY / "phase7_features_v1.parquet"

    feature_csv = DATA_OUTPUT_DIRECTORY / "phase7_features_v1.csv"

    features.to_parquet(
        feature_parquet,
        index=False,
    )

    features.to_csv(
        feature_csv,
        index=False,
    )

    geometry_manifest = freeze_geometry_provenance(
        root=ROOT,
        feature_table=features,
        neutral_attempts=neutral_attempts,
        geometry_root=(RESULT_OUTPUT_DIRECTORY / "geometries"),
    )

    geometry_manifest_path = RESULT_OUTPUT_DIRECTORY / "geometry_manifest.parquet"

    geometry_manifest.to_parquet(
        geometry_manifest_path,
        index=False,
    )

    li_reference_values = li_association["li_reference_energy_hartree"].unique()

    if len(li_reference_values) != 1:
        raise ValueError("Expected one frozen Li+ reference energy.")

    metadata = {
        "schema_version": "1.0",
        "feature_version": ("phase7_features_v1"),
        "candidate_count": len(features),
        "geometry_record_count": len(geometry_manifest),
        "software": {
            "python": (platform.python_version()),
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
            "xtb": _xtb_version(),
        },
        "input_sha256": {
            "candidates_v1.parquet": (sha256_file(CANDIDATES_PATH)),
            "xtb_descriptors.parquet": (sha256_file(XTB_PATH)),
            "xtb_attempts.parquet": (sha256_file(XTB_ATTEMPTS_PATH)),
            "li_association.parquet": (sha256_file(LI_ASSOCIATION_PATH)),
            "li_site_attempts.parquet": (sha256_file(LI_SITES_PATH)),
        },
        "li_reference_energy_hartree": (float(li_reference_values[0])),
        "units": {
            "molecular_weight": "g/mol",
            "tpsa": "angstrom^2",
            "xtb_orbital_energy": "eV",
            "xtb_gap": "eV",
            "xtb_dipole": "Debye",
            "xtb_polarizability": "atomic_units",
            "xtb_total_energy": "Hartree",
            "li_association": {
                "hartree": "Hartree",
                "ev": "eV",
                "kcal_mol": "kcal/mol",
            },
        },
        "scientific_cautions": [
            (
                "Neutral xTB total energies are "
                "composition-dependent and must not be "
                "ranked directly across different "
                "molecular formulas."
            ),
            (
                "Li association values are gas-phase "
                "GFN2-xTB electronic-energy proxies, "
                "not binding free energies."
            ),
            (
                "RDKit descriptors are 2D/topological "
                "or fragment-derived descriptors and "
                "do not replace electronic-structure "
                "calculations."
            ),
        ],
    }

    metadata_path = DATA_OUTPUT_DIRECTORY / "phase7_features_v1_metadata.json"

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Feature rows: {len(features)}")

    print(f"Feature columns: {len(features.columns)}")

    print(f"Frozen geometry records: {len(geometry_manifest)}")

    print(f"RDKit version: {rdBase.rdkitVersion}")

    print(f"xTB version: {metadata['software']['xtb']}")


if __name__ == "__main__":
    main()
