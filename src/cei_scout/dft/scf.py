"""Construction and execution of screening-level DFT calculations."""

from typing import Any

from pyscf import dft  # type: ignore[import-untyped]


def build_screening_scf(
    mol: Any,
    *,
    functional: str = "wb97x-v",
    dielectric_constant: float = 20.0,
    grid_level: int = 3,
) -> Any:
    """Build the locked Phase 8 screening DFT calculation.

    The calculation uses:
    - RKS for spin-zero species, UKS otherwise
    - omegaB97X-V
    - density fitting
    - IEF-PCM with the requested dielectric constant
    """

    mf = dft.RKS(mol, xc=functional) if mol.spin == 0 else dft.UKS(mol, xc=functional)

    # DFT exchange-correlation terms are integrated numerically.
    mf.grids.level = grid_level
    mf.nlcgrids.level = grid_level

    # Approximate expensive two-electron integrals with an auxiliary basis.
    mf = mf.density_fit()

    # Add the generic electrolyte-like dielectric environment.
    mf = mf.PCM()
    mf.with_solvent.method = "IEF-PCM"
    mf.with_solvent.eps = dielectric_constant

    return mf
