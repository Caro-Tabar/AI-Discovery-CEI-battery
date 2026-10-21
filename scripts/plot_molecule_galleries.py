"""Render the frozen CEI candidate pool as GitHub-ready molecular galleries."""

import csv
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/phase9_v1/species_ranking_v1.csv"
OUTPUT = ROOT / "docs/assets/molecules"

FAMILIES = {
    "sulfite": "Sulfites",
    "sulfate_or_sultone": "Sulfates and sultones",
    "sulfone_or_sulfoxide": "Sulfones and sulfoxides",
    "thioether_or_disulfide": "Thioethers and disulfides",
    "sulfonate_or_sulfonamide": "Sulfonates and sulfonamides",
    "mixed_s_o_n": "Mixed sulfur/oxygen/nitrogen",
}


def load_candidates() -> list[dict[str, str]]:
    """Read the complete frozen molecular population."""
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    ids = [row["candidate_id"] for row in rows]

    expected = {f"CEI-{i:04d}" for i in range(1, 61)}
    if len(rows) != 60 or set(ids) != expected:
        raise ValueError("Frozen 60-candidate pool mismatch")

    if sum(row["dft_status"] == "complete" for row in rows) != 56:
        raise ValueError("Expected 56 DFT-complete molecules")

    return rows


def render_gallery(
    rows: list[dict[str, str]],
    filename: str,
    ranked: bool = False,
) -> None:
    """Render real RDKit molecular structures as PNG and SVG."""
    molecules = []
    legends = []

    for row in rows:
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {row['candidate_id']}")

        molecules.append(molecule)

        if ranked:
            legend = f"{row['candidate_id']}  |  Rank {row['dual_redox_priority_rank']}"
        else:
            legend = row["candidate_id"]
            if row["dft_status"] != "complete":
                legend += "  [DFT incomplete]"

        legends.append(legend)

    options = {
        "molsPerRow": 4,
        "subImgSize": (370, 250),
        "legends": legends,
    }

    png = Draw.MolsToGridImage(molecules, useSVG=False, **options)
    png.save(OUTPUT / f"{filename}.png")

    svg = Draw.MolsToGridImage(molecules, useSVG=True, **options)
    (OUTPUT / f"{filename}.svg").write_text(svg, encoding="utf-8")

    print(f"Created {filename}: {len(molecules)} structures")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_candidates()

    top10 = sorted(
        (row for row in rows if row["dft_status"] == "complete"),
        key=lambda row: int(row["dual_redox_priority_rank"]),
    )[:10]

    render_gallery(top10, "top10_redox", ranked=True)

    groups = defaultdict(list)
    for row in rows:
        groups[row["candidate_family"]].append(row)

    if set(groups) != set(FAMILIES):
        raise ValueError("Unexpected chemical-family assignment")

    for family in FAMILIES:
        members = sorted(groups[family], key=lambda row: row["candidate_id"])
        render_gallery(members, f"family_{family}")

    print("MOLECULAR GALLERIES COMPLETE")
    print("60 unique molecules across six chemical families")


if __name__ == "__main__":
    main()
