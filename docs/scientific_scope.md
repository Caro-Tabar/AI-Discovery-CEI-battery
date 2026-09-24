# CEI-Scout scientific scope

## Purpose

CEI-Scout is a computational screening project for non-fluorinated,
sulfur-containing molecular additives in a high-voltage LNMO electrolyte.

The project combines public data, molecular descriptors, screening-level
quantum chemistry, uncertainty-aware active learning, and short classical
molecular-dynamics simulations. Its purpose is to identify computationally
promising trade-offs and demonstrate a reproducible materials-discovery
workflow—not to establish experimental performance.

## Scientific question

Can active learning find sulfur additives predicted to oxidize before EC/EMC
without excessively strengthening Li+ coordination or degrading relative
transport?

## Screening hypothesis

Within a chemically constrained pool of sulfur-containing molecules, some
candidate additives will exhibit a more favorable calculated oxidation tendency
than the EC/EMC solvent references without simultaneously producing undesirable
calculated Li+ association or relative transport behavior.

A physics-informed, uncertainty-aware active-learning workflow is expected to
identify these trade-offs using fewer prospective DFT evaluations than an
exhaustive calculation of the complete candidate pool.

This is a testable computational screening hypothesis. A negative or
inconclusive result remains a valid project outcome.

## Fixed chemical scope

The baseline electrolyte is:

- 1.0 M LiPF6
- EC:EMC at 3:7 by weight

“Non-fluorinated” applies only to the candidate additives. LiPF6 contains
fluorine, so CEI-Scout does not study or claim a fluorine-free electrolyte.

The finite candidate pool contains 60 neutral CHNOS molecules with:

- Molecular weight from 70 to 220 g/mol
- No halogens
- No metals
- No radicals

The detailed candidate inclusion and exclusion rules will be registered before
the pool is constructed.

## Assumptions

1. Relative screening trends are more defensible than absolute experimental
   predictions at the selected computational level.
2. Screening-level xTB and DFT calculations can provide useful relative
   oxidation and Li+-association rankings within the defined chemical domain.
3. A calculated vertical ionization quantity is a proxy for relative oxidation
   tendency, not direct proof of oxidation voltage, reaction products, or CEI
   formation.
4. Electronic Li+-association energies provide an internal coordination proxy;
   they are not solution-phase binding free energies.
5. Active-learning uncertainty is used to guide finite-pool sampling and does
   not represent every source of chemical or model uncertainty.
6. Short classical MD simulations can compare relative coordination and
   transport behavior under matched conditions but cannot describe reactive CEI
   chemistry.
7. Conclusions apply only to the registered candidate domain, computational
   methods, electrolyte composition, and simulation conditions.
8. Calculation failures are retained and reported as outcomes rather than
   silently removed.

Quantitative objective definitions, sign conventions, soft screening targets,
and MD concern criteria are registered in the subsequent Phase 4 tasks.

## Non-goals

CEI-Scout does not:

- Prove CEI formation, composition, or performance.
- Perform reactive molecular dynamics.
- Model an explicit LNMO surface or slab.
- Predict cell or electrolyte lifetime.
- Make experimental or commercial recommendations.

## Quantitative DFT screening definitions

The following definitions are registered before any candidate DFT labels are
calculated. Every energy entering a comparison must use the same electronic
method, basis set, solvent model, and energy-unit conversion.

Here, \(X\) denotes a candidate molecule and \(R_X^0\) denotes its optimized
neutral geometry.

### Vertical ionization energy

The calculated vertical ionization energy of molecule \(X\) is:

```text
VIE(X) = E(X+, R_X^0) - E(X, R_X^0)

```

The machine-facing property name is:

```text
vertical_ionization_energy_ev
```

A lower VIE indicates that removing an electron is calculated to require less
energy. This is interpreted only as a greater relative oxidation tendency under
the registered computational model.

It is not an experimentally referenced oxidation voltage and does not prove
that oxidation, decomposition, or CEI formation will occur.

### Preferential-oxidation objective

The more easily oxidized baseline solvent reference is the one with the lower
calculated VIE:

```text
solvent_reference_vie_ev = min(VIE(EC), VIE(EMC))
```

The oxidation-selectivity objective is:

```text
oxidation_selectivity_ev(X)
    = solvent_reference_vie_ev - VIE(X)
```

Interpretation:

- A positive value means the candidate VIE is lower than both solvent-reference
  VIEs.
- Zero means the candidate matches the lower EC/EMC reference.
- A negative value means at least one solvent reference has a lower VIE.
- Larger values are preferred.

The registered soft oxidation target is at least +0.20 eV:

```text
oxidation_selectivity_ev >= +0.20
```

This means the candidate VIE should be at least 0.20 eV below the lower of the
two EC/EMC reference VIEs.

“Preferential oxidation” means only this calculated relative comparison. It is
not a claim about experimental oxidation voltage, reaction rate, decomposition
products, electrode-surface reactions, or CEI formation.

### Electronic Li+ association energy

The electronic Li+ association energy is:

```text
association_energy_ev(X)
    = E(Li+·X) - E(Li+) - E(X)
```

All three energies must use the same registered electronic method and solvent
model.

Interpretation:

- A negative value represents electronically favorable association.
- A more negative value represents stronger calculated association.
- A positive value represents unfavorable association under this convention.

This electronic-energy proxy is not a binding free energy. It does not include
complete entropic, concentration, or dynamical-solvent effects.

### EC-relative Li+ coordination objective

The coordination objective is defined relative to EC:

```text
li_coordination_objective_ev(X)
    = association_energy_ev(X) - association_energy_ev(EC)
```

Interpretation:

- A positive value means the candidate binds Li+ less strongly than EC.
- Zero means the candidate and EC have equal calculated association energies.
- A negative value means the candidate binds Li+ more strongly than EC.
- Larger values are preferred.

For example, suppose:

```text
association_energy_ev(EC) = -1.00 eV
association_energy_ev(X)  = -1.15 eV
```

Then:

```text
li_coordination_objective_ev(X)
    = -1.15 eV - (-1.00 eV)
    = -0.15 eV
```

The candidate therefore binds Li+ 0.15 eV more strongly than EC.

The registered soft coordination target is at least -0.20 eV:

```text
li_coordination_objective_ev >= -0.20
```

Because -0.15 eV is greater than -0.20 eV, the example candidate remains within
the soft target. Values below -0.20 eV indicate binding more than 0.20 eV
stronger than EC.

### Active-learning objective orientation

The active-learning model uses two separate maximization objectives:

```text
maximize:
    1. oxidation_selectivity_ev
    2. li_coordination_objective_ev
```

Both objectives are deliberately oriented so that larger values are preferred.
They remain separate in the Pareto analysis and are not combined into a hidden
weighted score.

The +0.20 eV and -0.20 eV values are pre-registered soft screening targets.
They are not universal physical constants or hard eligibility filters.
Candidates outside either target remain recorded and may still be informative
because of uncertainty, Pareto trade-offs, or chemical-family diversity.

The targets must not be changed after candidate results are observed unless the
change is explicitly justified and recorded in the decision log.


## Pre-registered MD screening rules

The MD stage compares each finalist system with the matched baseline under the
same force-field family, ion-charge scaling, temperature, pressure, simulation
length, analysis method, and uncertainty procedure.

These screens provide relative diagnostics. They do not predict absolute
electrolyte conductivity, lifetime, or CEI chemistry.

### Li+ transport screen

The relative Li+ transport quantity is:

```text
li_diffusion_ratio
    = D_Li(finalist system) / D_Li(baseline system)
```

Interpretation:

- A value of 1.0 indicates equal estimated Li+ self-diffusion.
- A value below 1.0 indicates slower estimated Li+ diffusion in the finalist.
- A value above 1.0 indicates faster estimated Li+ diffusion in the finalist.

A finalist receives a transport-concern flag only when:

1. The central `li_diffusion_ratio` estimate is below 0.70.
2. The upper bound of its registered uncertainty interval is also below 0.70.

If the central estimate is below 0.70 but the uncertainty interval crosses
0.70, the result is labeled inconclusive rather than concerning.

The uncertainty calculation must account for time correlation and independent
replicas. Individual trajectory frames must not be treated as independent
samples.

This is a relative transport screen, not an estimate of ionic conductivity or
cell-rate capability.

### Li+ coordination screen

For each system, the combined carbonate-solvent coordination is:

```text
carbonate_coordination_number
    = CN(Li+-EC) + CN(Li+-EMC)
```

Coordination numbers are obtained by integrating the registered Li+-species RDF
to its documented first-shell cutoff.

The solvent-displacement quantity is:

```text
carbonate_displacement
    = carbonate_coordination_number(baseline)
      - carbonate_coordination_number(finalist)
```

A positive value indicates that EC and EMC have been displaced from the Li+
first coordination shell in the finalist system.

A coordination concern is flagged for scientific review when:

1. The finalist shows a replicated reduction in combined EC/EMC coordination.
2. The uncertainty-supported change is consistent across the matched replicas.
3. The additive occupies a measurable portion of the Li+ first shell.

The magnitude, uncertainty, RDF shape, individual EC and EMC contributions, and
additive coordination are all reported. No universal numerical displacement
threshold is imposed because coordination depends on composition, force field,
and the registered RDF cutoff.

This flag is not an automatic rejection and is not evidence of favorable or
unfavorable CEI formation.
