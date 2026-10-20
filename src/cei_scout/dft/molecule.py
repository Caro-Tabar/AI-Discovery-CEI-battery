"""Construction and validation of PySCF molecular systems."""

from pathlib import Path
from typing import Literal

from pyscf import gto  # type: ignore[import-untyped]

ReferenceMethod = Literal["RKS", "UKS"]


def build_molecule(
    xyz_path: Path,
    *,
    basis: str,
    charge: int,
    spin: int,
) -> gto.Mole:
    """Build a PySCF molecule from an XYZ geometry.

    Args:
        xyz_path: XYZ geometry in Angstrom.
        basis: Orbital basis-set name understood by PySCF.
        charge: Total molecular charge.
        spin: PySCF spin, defined as N_alpha - N_beta = 2S.

    Returns:
        Built and validated PySCF Mole object.
    """
    if not xyz_path.is_file():
        raise FileNotFoundError(f"XYZ geometry not found: {xyz_path}")

    mol = gto.M(
        atom=str(xyz_path),
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        verbose=0,
    )

    n_alpha, n_beta = mol.nelec

    if n_alpha + n_beta != mol.nelectron:
        raise ValueError("Alpha/beta populations do not match total electron count.")

    if n_alpha - n_beta != spin:
        raise ValueError("Alpha/beta populations do not match requested spin.")

    return mol


def select_reference_method(mol: gto.Mole) -> ReferenceMethod:
    """Select restricted or unrestricted KS treatment from molecular spin."""
    return "RKS" if mol.spin == 0 else "UKS"
