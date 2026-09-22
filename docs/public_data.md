# Public LIBE data

## Role in CEI-Scout

CEI-Scout uses the Lithium-Ion Battery Electrolyte (LIBE) dataset as a
fully labelled retrospective benchmark for testing active-learning
behavior and as public quantum-chemical property context.

LIBE is not used as a substitute for prospective CEI-Scout candidate
DFT calculations.

## Source and license

Dataset: Lithium-Ion Battery Electrolyte (LIBE) dataset

Authors: Evan Walter Clark Spotte-Smith, Samuel M. Blau, Xiaowei Xie,
Hetal Patel, Mingjian Wen, Brandon Wood, Shyam Dwaraknath, and
Kristin Persson

Figshare article: 14226464

Pinned version: 2

Version DOI: 10.6084/m9.figshare.14226464.v2

License: CC BY 4.0

Upstream file: `libe.json`

The exact retrieved file is identified in
`results/manifests/libe_source_v2.json` by its SHA256 checksum.

The raw LIBE file is not stored in Git.

## Published calculation context

LIBE contains 17,190 validated molecular records generated at the
ωB97X-V/def2-TZVPPD/SMD level of theory.

The dataset contains molecular structures, connectivity,
charge/spin information, electronic and thermochemical properties,
and vibrational information.

CEI-Scout preserves the original upstream `molecule_id` values.

## CEI-Scout transformations

The raw LIBE dataset is first ingested without changing source values.

For the retrospective benchmark, CEI-Scout requires parent species to
be:

- neutral;
- closed-shell singlets;
- composed only of C, H, N, O, and S;
- between 70 and 220 g/mol.

Eligible neutral parents are matched to unique isomorphic +1 doublet
and -1 doublet records.

Electronic-energy differences use the published raw Q-Chem field:

`thermo.raw.electronic_energy_Ha`

Because the neutral, cation, and anion structures were separately
optimized, CEI-Scout labels these derived quantities as adiabatic
electronic ionization energy and adiabatic electronic electron
affinity. They are not described as vertical ionization energies or
experimental redox potentials.

## Benchmark size

The preregistered target size was 300 molecules.

Only 133 records satisfied all parent eligibility, identity, charge
state, and data-quality requirements.

Under the project's predetermined contingency rule, all 133 eligible
records are retained. No identity, charge-state, elemental, mass, or
data-quality filter was relaxed to reach an arbitrary sample size.

## Domain shift

The 133-record retrospective benchmark is chemically different from
the prospective CEI-Scout candidate pool.

In particular, the prospective pool is deliberately composed of
sulfur-containing additives, whereas no sulfur-containing parent
survived the complete LIBE neutral/+1/-1 matching procedure.

The retrospective benchmark is therefore used to evaluate
active-learning mechanics, sample efficiency, uncertainty behavior,
and leakage controls. It is not claimed to be a directly
representative training distribution for the prospective sulfur
candidate pool.

`results/reports/libe_domain_shift_v1.json` records the quantitative
comparison.

A more complete feature-space domain-shift analysis becomes possible
after Phase 7 generates common RDKit and xTB descriptors.

## Reproducibility

The upstream archive is downloaded from the pinned Figshare version
with `scripts/download_libe.py`.

The downloader verifies the exact source using the registered SHA256.

The raw archive remains outside Git.

The parser, filtering and charge-state matching code generate the
compact derived benchmark reproducibly from the pinned source.

## Limitations

LIBE was designed around electrolyte and interphase chemistry and has
a chemical distribution strongly influenced by its principal
molecules and fragmentation/recombination procedure.

The derived CEI-Scout benchmark is smaller and chemically narrower
than the original LIBE dataset because it additionally requires
neutral CHNOS parents within the CEI-Scout size range and uniquely
matched oxidized and reduced charge states.

Performance on the retrospective LIBE benchmark must therefore not be
interpreted as direct evidence that a surrogate will extrapolate
accurately to the prospective sulfur-additive search space.
