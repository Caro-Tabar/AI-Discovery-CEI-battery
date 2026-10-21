# CEI-Scout

**Physics-informed molecular screening and machine learning for cathode-electrolyte interphase (CEI) additive discovery.**

60 molecules · 6 chemical families · 56 DFT-complete candidates · 6 ML models

[**Molecules**](#molecular-library) · [**DFT**](#vertical-dft-screening) · [**Machine learning**](#machine-learning-benchmark) · [**Ranking**](#candidate-prioritization) · [**Active learning**](#retrospective-active-learning)

---

## Screening workflow

```mermaid
flowchart LR
    A["60 sulfur-containing molecules"] --> B["RDKit conformers"]
    B --> C["GFN2-xTB screening"]
    C --> D["Vertical DFT: 56 complete"]
    D --> E["Six ML models"]
    D --> F["Redox ranking"]
    E --> G["Uncertainty analysis"]
    E --> H["Retrospective active learning"]
```

A small, reproducible computational workflow motivated by additive discovery for high-voltage LNMO electrolytes.

## Molecular library

![Top ten redox-priority molecular structures](docs/assets/molecules/top10_redox.png)

*Top ten molecules by exploratory combined redox rank. Structures are RDKit 2D representations, not optimized DFT geometries.*

<details>
<summary><strong>View the complete 60-molecule library by chemical family</strong></summary>

### Sulfites

![Sulfite molecular structures](docs/assets/molecules/family_sulfite.png)

### Sulfates and sultones

![Sulfate and sultone molecular structures](docs/assets/molecules/family_sulfate_or_sultone.png)

### Sulfones and sulfoxides

![Sulfone and sulfoxide molecular structures](docs/assets/molecules/family_sulfone_or_sulfoxide.png)

### Thioethers and disulfides

![Thioether and disulfide molecular structures](docs/assets/molecules/family_thioether_or_disulfide.png)

### Sulfonates and sulfonamides

![Sulfonate and sulfonamide molecular structures](docs/assets/molecules/family_sulfonate_or_sulfonamide.png)

### Mixed sulfur / oxygen / nitrogen

![Mixed S O N molecular structures](docs/assets/molecules/family_mixed_s_o_n.png)

</details>

[Original interactive candidate-review HTML](results/reports/candidate_pool_review.html) — download and open locally to inspect the full molecular review.

## GFN2-xTB screening

Neutral conformers were screened using molecular mechanics and GFN2-xTB. Li+ association was evaluated as a gas-phase electronic-energy proxy.

<table>
<tr>
<td width="50%"><img src="docs/assets/phase7/gap_histogram.png" alt="xTB HOMO-LUMO gap distribution" width="100%"></td>
<td width="50%"><img src="docs/assets/phase7/li_association_histogram.png" alt="xTB lithium association distribution" width="100%"></td>
</tr>
</table>

*Electronic gaps and Li+ association provide complementary screening descriptors. Association energies are not binding free energies.*

<details>
<summary><strong>Additional molecular-descriptor and QC plots</strong></summary>

<table>
<tr>
<td><img src="docs/assets/phase7/dipole_histogram.png" alt="Dipole distribution" width="100%"></td>
<td><img src="docs/assets/phase7/polarizability_histogram.png" alt="Polarizability distribution" width="100%"></td>
</tr>
<tr>
<td><img src="docs/assets/phase7/tpsa_histogram.png" alt="Topological polar surface area" width="100%"></td>
<td><img src="docs/assets/phase7/homo_vs_li_association.png" alt="HOMO versus Li association" width="100%"></td>
</tr>
</table>

</details>

## Vertical DFT screening

**PySCF · omegaB97X-V / def2-SVPD · IEF-PCM, dielectric constant 20**

180 electronic-state calculations attempted; 176 converged. The 56 retained DFT-complete candidates have neutral, oxidized, and reduced energies evaluated at a common neutral geometry.

<table>
<tr>
<td><img src="docs/assets/phase8/vertical_ionization_energy.png" alt="Vertical ionization energy distribution" width="100%"></td>
<td><img src="docs/assets/phase8/vertical_electron_affinity.png" alt="Vertical electron affinity distribution" width="100%"></td>
</tr>
<tr>
<td><img src="docs/assets/phase8/xtb_homo_vs_dft_ionization.png" alt="xTB HOMO versus DFT ionization energy" width="100%"></td>
<td><img src="docs/assets/phase8/xtb_lumo_vs_dft_electron_affinity.png" alt="xTB LUMO versus DFT electron affinity" width="100%"></td>
</tr>
</table>

*xTB frontier-orbital descriptors correlate with DFT redox energies, motivating inexpensive surrogate models.*

The calculated IE and EA are vertical electronic-energy descriptors, **not experimental or electrode-referenced redox potentials**.

[DFT methodology](docs/dft_reference.md) · [DFT-complete features](data/processed/phase8_features_v1.csv) · [DFT exclusion policy](configs/phase8_dft_exclusions_v1.yaml)

## Machine learning benchmark

Six regressors were evaluated using eight RDKit/xTB molecular descriptors to predict DFT ionization energy (IE) and electron affinity (EA).

<table>
<tr>
<td width="50%"><img src="docs/assets/phase9/model_comparison_ionization_energy.png" alt="Six-model IE comparison" width="100%"></td>
<td width="50%"><img src="docs/assets/phase9/model_comparison_electron_affinity.png" alt="Six-model EA comparison" width="100%"></td>
</tr>
</table>

**Held-out mean absolute error (MAE), eV — lower is better**

| Model | IE: repeated CV | IE: family holdout | EA: repeated CV | EA: family holdout |
|---|---:|---:|---:|---:|
| Mean | 0.974 | 1.117 | 0.412 | 0.439 |
| Linear orbital baseline | 0.228 | **0.236** | 0.259 | **0.277** |
| Ridge | 0.195 | 0.270 | 0.202 | 0.409 |
| Extra Trees | 0.147 | 0.337 | **0.135** | 0.342 |
| Gaussian Process | **0.129** | 0.359 | 0.147 | 0.332 |
| Neural Network (MLP) | 0.165 | 0.280 | 0.154 | 0.384 |

*Nonlinear models perform well within represented families. The simple orbital baselines transfer better to held-out chemical families.*

### DFT versus ML predictions

<table>
<tr>
<td><img src="docs/assets/phase9/gpr_parity_ionization_energy.png" alt="GPR ionization-energy parity plot" width="100%"></td>
<td><img src="docs/assets/phase9/gpr_parity_electron_affinity.png" alt="GPR electron-affinity parity plot" width="100%"></td>
</tr>
</table>

*Each molecule is represented by the mean of five held-out GPR predictions.*

<details>
<summary><strong>Model errors by chemical family</strong></summary>

<table>
<tr>
<td><img src="docs/assets/phase9/family_error_ionization_energy.png" alt="Ionization energy family error heatmap" width="100%"></td>
<td><img src="docs/assets/phase9/family_error_electron_affinity.png" alt="Electron affinity family error heatmap" width="100%"></td>
</tr>
</table>

The family-held-out benchmark tests prediction on a chemical family absent from model training.

</details>

## Candidate prioritization

![Ionization energy versus electron affinity with top ten ranked candidates](docs/assets/phase9/redox_priority_map.png)

*The exploratory rank combines low vertical IE and high vertical EA using equal-weight ordinal positions. Li+ association is reported separately, not included in the score.*

| Rank | Candidate | IE (eV) | EA (eV) |
|---:|---|---:|---:|
| 1 | CEI-0054 · Rhodanine † | 5.878 | 1.752 |
| 2 | CEI-0037 · Diphenyl sulfide | 6.141 | 0.739 |
| 3 | CEI-0035 · Thioanisole | 6.065 | 0.633 |
| 4 | CEI-0057 · Glycyl-L-methionine | 6.363 | 0.430 |
| 5 | CEI-0048 · p-Toluenesulfonamide | 7.329 | 1.160 |
| 6 | CEI-0044 · N-Methyl-p-toluenesulfonamide | 7.350 | 1.169 |
| 7 | CEI-0050 · N-Ethyl-p-toluenesulfonamide | 7.390 | 1.231 |
| 8 | CEI-0047 · Benzenesulfonamide | 7.680 | 1.290 |
| 9 | CEI-0056 · L-Ethionine | 6.351 | 0.283 |
| 10 | CEI-0032 · Dipropyl disulfide | 6.766 | 0.588 |

† Rhodanine retains a DFT QC review flag. The four DFT-incomplete candidates remain in the original 60-molecule pool but are not assigned DFT ranks.

[Full 60-candidate ranking](results/phase9_v1/species_ranking_v1.csv)

## Uncertainty-aware ranking

![GPR ranking stability for top ten candidates](docs/assets/phase9/ranking_stability_top10.png)

*Five cross-validation repetitions reveal which molecular priorities remain stable as the training set changes.*

| Candidate | DFT rank | GPR median rank | Predicted top-10 frequency |
|---|---:|---:|---:|
| Rhodanine | 1 | 3 | 60% |
| Diphenyl sulfide | 2 | 2 | 100% |
| Thioanisole | 3 | 1 | 100% |
| Glycyl-L-methionine | 4 | 16 | 0% |
| N-Methyl-p-toluenesulfonamide | 6 | 4 | 100% |
| Benzenesulfonamide | 8 | 5 | 100% |

GPR nominal 95% interval coverage was 88.9% for IE and 90.4% for EA under repeated CV. The intervals are not calibrated, and top-10 frequencies are stability diagnostics rather than probabilities.

[Full ranking-stability results](results/phase9_v1/gpr_ranking_stability_v1.csv) · [Uncertainty diagnostics](results/phase9_v1/gpr_uncertainty_diagnostics_v1.csv)

## Retrospective active learning

<table>
<tr>
<td width="50%"><img src="docs/assets/phase9/active_learning_ionization_energy.png" alt="Ionization energy active-learning curve" width="100%"></td>
<td width="50%"><img src="docs/assets/phase9/active_learning_electron_affinity.png" alt="Electron affinity active-learning curve" width="100%"></td>
</tr>
</table>

*Five paired simulations compared GPR uncertainty sampling against random acquisition, using previously calculated DFT labels.*

| Target | Random MAE | Uncertainty MAE | Final improvement |
|---|---:|---:|---:|
| Ionization energy | 0.2226 eV | **0.1586 eV** | 28.8% |
| Electron affinity | 0.1305 eV | **0.1222 eV** | 6.3% |

Each simulation started with 12 labeled molecules, reserved 12 for testing, and acquired 20 additional molecules over ten rounds.

The experiment reuses existing DFT labels. **No prospective active-learning DFT campaign was performed.**

[Learning curves](results/phase9_v1/al_summary_v1.csv) · [Selected molecules](results/phase9_v1/al_acquisitions_v1.csv) · [Split manifest](results/phase9_v1/al_split_manifest_v1.csv)

---

## Methods & reproducibility

| Stage | Methods |
|---|---|
| Molecular preparation | RDKit, ETKDGv3, MMFF94s / UFF |
| Semiempirical screening | GFN2-xTB; neutral and Li+ association calculations |
| DFT | PySCF, omegaB97X-V / def2-SVPD, IEF-PCM |
| ML descriptors | Eight selected RDKit and xTB descriptors |
| Regressors | Mean, linear, Ridge, Extra Trees, GPR, MLP |
| Validation | Repeated 5-fold CV and leave-one-chemical-family-out |
| Active learning | Retrospective joint GPR uncertainty sampling versus random acquisition |

All ML splits are reproducible and preserve candidate identity, molecular families, QC information, and exclusion decisions.

To regenerate the portfolio figures from the existing result tables, use the project's configured Python environment:

```bash
python scripts/plot_molecule_galleries.py
python scripts/plot_phase9_portfolio.py
python -m pytest -q
```

[Phase 9 ML configuration](configs/phase9_ml_v1.yaml) · [Model outputs](results/phase9_v1/) · [Processed features](data/processed/) · [Source code](src/cei_scout/ml/)

### Scope

This is a screening-level research and ML portfolio project. It does not establish real CEI formation, interfacial decomposition mechanisms, transport behavior, cell lifetime, or experimentally validated additive performance.

The candidate additives are non-fluorinated; the motivating LiPF6-based electrolyte is not fluorine-free.
