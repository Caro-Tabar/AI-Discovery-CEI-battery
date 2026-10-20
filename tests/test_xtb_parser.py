"""Tests for parsing GFN2-xTB scalar descriptors."""

from cei_scout.preprocessing.xtb_parser import parse_xtb_output

XTB_OUTPUT = """
     4        2.0000           -0.4463919             -12.1469 (HOMO)
     5                          0.0826818                2.2499 (LUMO)

             HL-Gap            0.5290737 Eh            14.3968 eV

Mol. alpha(0) /au        :          9.429351

molecular dipole:
                x           y           z       tot (Debye)
 full:        0.696       0.000       0.492       2.167

:: total charge               0.000000000000 e     ::

 -------------------------------------------------
| TOTAL ENERGY               -5.070322476938 Eh   |
| HOMO-LUMO GAP               0.529073700000 Eh   14.3968 eV |
 -------------------------------------------------
"""


def test_parse_xtb_output() -> None:
    result = parse_xtb_output(XTB_OUTPUT)

    assert result.total_energy_hartree == -5.070322476938
    assert result.homo_ev == -12.1469
    assert result.lumo_ev == 2.2499
    assert result.homo_lumo_gap_ev == 14.3968
    assert result.dipole_debye == 2.167
    assert result.polarizability_au == 9.429351
    assert result.total_charge_e == 0.0
