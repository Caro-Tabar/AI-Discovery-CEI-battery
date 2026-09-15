# CEI-Scout decision log

This file records scientific and engineering choices before candidate results
are observed. Changes must be appended with a date, justification, and affected
outputs; prior decisions must not be silently rewritten.

## Locked decisions — 2026-09-14

| Decision | Locked choice |
| --- | --- |
| Project name | CEI-Scout |
| Public repository | `Caro-Tabar/AI-Discovery-CEI-battery` |
| Scientific target | Non-fluorinated sulfur-containing molecular additives for a 4.7 V LNMO electrolyte |
| Baseline formulation | 1.0 M LiPF6 in EC:EMC at 3:7 by weight |
| Scientific question | Can active learning find sulfur additives predicted to oxidize before EC/EMC without excessively strengthening Li+ coordination or degrading relative transport? |
| Candidate pool | 60 neutral CHNOS molecules, 70–220 g/mol, without halogens, metals, or radicals |
| Candidate families | Sulfites; sulfates/sultones; sulfones/sulfoxides; thioethers/disulfides; sulfonates/sulfonamides; mixed S/O/N motifs |
| Public-data role | LIBE supplies the retrospective active-learning benchmark and property context |
| Cheap chemistry layer | RDKit descriptors and conformers plus GFN2-xTB optimization and properties |
| Screening DFT | PySCF, ωB97X-V/def2-SVPD, density fitting, IEF-PCM with ε = 20.0, and single points on xTB geometries |
| Finalist DFT | PBE0-D3(BJ)/def2-SVP geometry sensitivity followed by ωB97X-V/def2-TZVPD single points |
| DFT oxidation label | ΔSCF vertical ionization energy at the neutral geometry |
| DFT coordination label | Li+–additive electronic association energy relative to EC, not a free energy |
| DFT controls | EC, EMC, DTD/ethylene sulfate, and sulfolane |
| Prospective DFT budget | Six diverse seeds plus three batches of three, totaling 15 candidate labels excluding controls |
| ML representation | Curated RDKit and xTB numeric descriptors reduced to at most eight PCA components |
| Main surrogate | One exact Gaussian process per objective using BoTorch and GPyTorch |
| Acquisition | Finite-pool batch qLogNEHVI with batch size three and a recorded reference point |
| ML baselines | Random sampling, maximin diversity, uncertainty-only sampling, and ExtraTrees |
| Primary MD route | OpenMM with OpenFF Interchange/Sage, explicit Li+/PF6− handling, and 0.8 ion-charge scaling |
| MD fallback | If any finalist lacks OpenFF coverage or baseline density error exceeds 10%, use GAFF2 for every compared system |
| MD formulation | Baseline plus 5 wt% additive stress-test systems |
| MD budget | Baseline plus three finalists, two replicas each, 1 ns NPT equilibration and 5 ns NVT production |
| Core MD outputs | Density, Li+ RDFs, coordination, solvent-shell composition, residence, Li diffusion ratio, PF6 contact pairing, and additive clustering |
| Decision method | Pareto ranking with uncertainty and pre-registered soft gates; no hidden weighted score |
| Code environment | Ubuntu 24.04 under WSL2, Python 3.11, micromamba, and `pyproject.toml` |
| Workflow style | Installable `src/` package, validated YAML configs, Typer CLI, tests, logging, and presentation-only notebooks |
| Repository license | MIT for original code; CC BY 4.0 for original data and figures; third-party licenses retained |
| Large-file policy | No raw LIBE archive or full MD trajectories in Git; use checksum-driven downloads and compact derived artifacts |
