from pathlib import Path

from cei_scout.dft.molecule import build_molecule, select_reference_method


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


def test_build_closed_shell_molecule(tmp_path: Path) -> None:
    xyz_path = _write_water_xyz(tmp_path)

    mol = build_molecule(
        xyz_path,
        basis="def2-svpd",
        charge=0,
        spin=0,
    )

    assert mol.nelectron == 10
    assert mol.nelec == (5, 5)
    assert select_reference_method(mol) == "RKS"


def test_build_open_shell_cation(tmp_path: Path) -> None:
    xyz_path = _write_water_xyz(tmp_path)

    mol = build_molecule(
        xyz_path,
        basis="def2-svpd",
        charge=1,
        spin=1,
    )

    assert mol.nelectron == 9
    assert mol.nelec == (5, 4)
    assert select_reference_method(mol) == "UKS"
