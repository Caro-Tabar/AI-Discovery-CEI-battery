# LIBE retrospective benchmark targets

## Purpose

The LIBE benchmark is a fully labelled retrospective test of
multi-objective active-learning behavior.

It does not provide prospective CEI-Scout candidate labels and is not
treated as numerically interchangeable with the later CEI-Scout DFT
workflow.

## Parent-species filter

A benchmark parent must be:

- charge 0;
- spin multiplicity 1;
- composed only of C, H, N, O, and S;
- between 70 and 220 g/mol.

A parent is retained only when an isomorphic charge +1 doublet and
charge -1 doublet are both present.

Molecular identity across charge states is defined by element-labelled
graph isomorphism using the LIBE `species` and `bonds` fields.

## Energy source

The source quantity is:

`thermo.raw.electronic_energy_Ha`

No thermochemical correction or LIBE free-energy field is substituted.

## Retrospective labels

For independently optimized neutral, cationic, and anionic structures:

`adiabatic_ionization_energy_ev = E(+1) - E(0)`

`adiabatic_electron_affinity_ev = E(0) - E(-1)`

with Hartree converted to electron-volts.

For later multi-objective acquisition, two explicitly oriented
maximization targets are also stored:

`oxidation_ease_ev = -adiabatic_ionization_energy_ev`

`reduction_resistance_ev = -adiabatic_electron_affinity_ev`

These are electronic-energy differences between separately optimized
solution-phase LIBE structures.

They are not vertical ionization energies, electrode potentials,
experimental redox voltages, or CEI-formation predictions.

## Benchmark stratification

For the retrospective subset:

- `chemical_system` is used as the operational family stratum;
- an element-labelled heavy-atom connectivity hash is used as the
  scaffold stratum.

This intentionally avoids assigning sulfur functional-group labels that
cannot be determined reliably from LIBE's simple connectivity field
without bond-order information.

Selection is deterministic round-robin across family/scaffold strata.

If fewer than 300 fully matched records survive, every eligible record
is retained, following the preregistered contingency rule.
