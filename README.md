# CEI-Scout

Physics-informed active learning of non-fluorinated sulfur additives for
high-voltage LNMO electrolytes.

## Scientific question

Can active learning find sulfur additives predicted to oxidize before EC/EMC
without excessively strengthening Li+ coordination or degrading relative
transport?

## Status

CEI-Scout is currently in the repository-scaffolding stage. The development
environment, package structure, automated quality checks, and continuous
integration workflow are configured. Scientific workflows and results have not
yet been implemented.

## Scope clarification

“Non-fluorinated” refers only to the candidate additives. The baseline
electrolyte is 1.0 M LiPF6 in EC:EMC at 3:7 by weight, so this project does not
claim to study a fluorine-free electrolyte.

## Non-goals

This project does not:

- Prove CEI formation, composition, or performance.
- Perform reactive molecular dynamics.
- Model an explicit LNMO slab.
- Predict cell or electrolyte lifetime.
- Make experimental recommendations.
