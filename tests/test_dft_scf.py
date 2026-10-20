from pathlib import Path

from cei_scout.dft.molecule import build_molecule
from cei_scout.dft.scf import build_screening_scf


def _write_water_xyz(tmp_path: Path) -> Path:
    xyz_path = tmp_path / "water.xyz"
    xyz_path.write_text(
        """3
water
O  0.000000  0.000000  0.000000
H  0.000000  0.757000  0.587000
H  0.000000 -0.757000  0.587000
""",
        encoding="utf-8",
    )
    return xyz_path


def test_build_screening_rks(tmp_path: Path) -> None:
    mol = build_molecule(
        _write_water_xyz(tmp_path),
        basis="def2-svpd",
        charge=0,
        spin=0,
    )

    mf = build_screening_scf(mol)

    assert mf.xc.lower().replace("_", "-") == "wb97x-v"
    assert mf.grids.level == 3
    assert mf.with_solvent.method == "IEF-PCM"
    assert mf.with_solvent.eps == 20.0
    assert mf.with_df is not None


def test_build_screening_uks(tmp_path: Path) -> None:
    mol = build_molecule(
        _write_water_xyz(tmp_path),
        basis="def2-svpd",
        charge=1,
        spin=1,
    )

    mf = build_screening_scf(mol)

    assert mf.mol.spin == 1
    assert mf.with_df is not None
