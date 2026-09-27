"""Parse scalar molecular descriptors from GFN2-xTB output."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


class XtbDescriptors(BaseModel):
    """Scalar descriptors extracted from a neutral GFN2-xTB calculation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_energy_hartree: float
    homo_ev: float
    lumo_ev: float
    homo_lumo_gap_ev: float
    dipole_debye: float
    polarizability_au: float
    total_charge_e: float


def _require_float(
    pattern: str,
    text: str,
    label: str,
    flags: int = 0,
) -> float:
    match = re.search(pattern, text, flags)

    if match is None:
        raise ValueError(f"Could not parse xTB {label}.")

    return float(match.group(1))


def parse_xtb_total_energy(text: str) -> float:
    """Parse the final total electronic energy from xTB stdout."""

    return _require_float(
        rf"\|\s*TOTAL ENERGY\s+({_FLOAT})\s+Eh\s*\|",
        text,
        "total energy",
    )


def parse_xtb_output(text: str) -> XtbDescriptors:
    """Parse required screening descriptors from xTB stdout."""

    total_energy = parse_xtb_total_energy(text)

    homo = _require_float(
        rf"^\s*\d+\s+.*?\s+({_FLOAT})\s+\(HOMO\)\s*$",
        text,
        "HOMO energy",
        re.MULTILINE,
    )

    lumo = _require_float(
        rf"^\s*\d+\s+.*?\s+({_FLOAT})\s+\(LUMO\)\s*$",
        text,
        "LUMO energy",
        re.MULTILINE,
    )

    gap = _require_float(
        rf"HOMO-LUMO\s+GAP\s+(?:{_FLOAT}\s+Eh\s+)?({_FLOAT})\s+eV",
        text,
        "HOMO-LUMO gap",
        re.IGNORECASE,
    )

    dipole = _require_float(
        rf"molecular dipole:.*?"
        rf"^\s*full:\s+{_FLOAT}\s+{_FLOAT}\s+{_FLOAT}\s+({_FLOAT})\s*$",
        text,
        "dipole moment",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    polarizability = _require_float(
        rf"Mol\.\s+.*?\(0\)\s*/au\s*:\s*({_FLOAT})",
        text,
        "molecular polarizability",
    )

    total_charge = _require_float(
        rf"total charge\s+({_FLOAT})\s+e",
        text,
        "total charge",
        re.IGNORECASE,
    )

    calculated_gap = lumo - homo

    if abs(calculated_gap - gap) > 0.05:
        raise ValueError("Parsed HOMO/LUMO energies are inconsistent with the xTB gap.")

    return XtbDescriptors(
        total_energy_hartree=total_energy,
        homo_ev=homo,
        lumo_ev=lumo,
        homo_lumo_gap_ev=gap,
        dipole_debye=dipole,
        polarizability_au=polarizability,
        total_charge_e=total_charge,
    )
