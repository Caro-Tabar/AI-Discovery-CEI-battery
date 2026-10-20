"""Run EC and EMC smoke tests through the neutral GFN2-xTB workflow."""

from pathlib import Path

from cei_scout.preprocessing.config import (
    load_conformer_pipeline_config,
)
from cei_scout.preprocessing.conformers import generate_conformers
from cei_scout.preprocessing.forcefield import prescreen_conformers
from cei_scout.preprocessing.xtb_config import load_xtb_config
from cei_scout.preprocessing.xtb_runner import run_xtb_attempt

ROOT = Path(__file__).resolve().parents[1]

CONFORMER_CONFIG = ROOT / "configs/conformers.yaml"
XTB_CONFIG = ROOT / "configs/xtb.yaml"

OUTPUT_ROOT = ROOT / "results/xtb_smoke"

MOLECULES = {
    "EC": "C1COC(=O)O1",
    "EMC": "CCOC(=O)OC",
}


def main() -> None:
    conformer_config = load_conformer_pipeline_config(CONFORMER_CONFIG)
    xtb_config = load_xtb_config(XTB_CONFIG)

    failures = 0

    for name, smiles in MOLECULES.items():
        molecule, _ = generate_conformers(
            smiles,
            conformer_config.embedding,
        )

        ranked = prescreen_conformers(
            molecule,
            conformer_config.prescreen,
        )

        best = ranked[0]

        result = run_xtb_attempt(
            candidate_id=name,
            selection_rank=1,
            attempt_number=1,
            molecule=molecule,
            conformer_id=best.conformer_id,
            config=xtb_config,
            output_root=OUTPUT_ROOT,
        )

        print(f"{name}: {result.status}")

        if result.descriptors is not None:
            print(f"  energy = {result.descriptors.total_energy_hartree:.8f} Eh")
            print(f"  gap = {result.descriptors.homo_lumo_gap_ev:.4f} eV")

        if result.status != "success":
            failures += 1
            print(f"  stage = {result.failure_stage}")
            print(f"  reason = {result.message}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
