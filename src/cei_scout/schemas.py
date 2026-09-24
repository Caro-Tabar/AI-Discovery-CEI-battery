"""Validated data contracts for CEI-Scout scientific records."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CANONICAL_PROPERTY_UNITS",
    "AcquisitionRecord",
    "ActiveLearningRoundRecord",
    "CalculationRecord",
    "CandidateFamily",
    "CandidateRecord",
    "CanonicalUnit",
    "MDObservableRecord",
    "MDReplicaRecord",
    "MDSystemRecord",
    "ProvenanceRecord",
    "RejectionReason",
    "SchemaModel",
]


class CandidateFamily(StrEnum):
    """Registered sulfur-containing candidate families."""

    SULFITE = "sulfite"
    SULFATE_OR_SULTONE = "sulfate_or_sultone"
    SULFONE_OR_SULFOXIDE = "sulfone_or_sulfoxide"
    THIOETHER_OR_DISULFIDE = "thioether_or_disulfide"
    SULFONATE_OR_SULFONAMIDE = "sulfonate_or_sulfonamide"
    MIXED_S_O_N = "mixed_s_o_n"


class RejectionReason(StrEnum):
    """Controlled reasons for excluding a proposed candidate."""

    INVALID_VALENCE = "invalid_valence"
    MIXTURE = "mixture"
    DUPLICATE = "duplicate"
    NON_NEUTRAL = "non_neutral"
    DISALLOWED_ELEMENT = "disallowed_element"
    RADICAL = "radical"
    MOLECULAR_WEIGHT_OUT_OF_RANGE = "molecular_weight_out_of_range"
    TOO_MANY_ROTATABLE_BONDS = "too_many_rotatable_bonds"
    PEROXIDE = "peroxide"
    AZIDE = "azide"
    ENERGETIC_MOTIF = "energetic_motif"
    STANDARDIZATION_FAILED = "standardization_failed"
    MISSING_IDENTITY = "missing_identity"


class CanonicalUnit(StrEnum):
    """Canonical serialized units used by CEI-Scout records."""

    HARTREE = "hartree"
    ELECTRONVOLT = "eV"
    GRAM_PER_MOLE = "g/mol"
    KELVIN = "K"
    BAR = "bar"
    NANOSECOND = "ns"
    ANGSTROM = "angstrom"
    GRAM_PER_MILLILITER = "g/mL"
    SQUARE_CENTIMETER_PER_SECOND = "cm^2/s"
    DIMENSIONLESS = "1"


CANONICAL_PROPERTY_UNITS: dict[str, CanonicalUnit] = {
    "molecular_weight_g_mol": CanonicalUnit.GRAM_PER_MOLE,
    "electronic_energy_hartree": CanonicalUnit.HARTREE,
    "vertical_ionization_energy_ev": CanonicalUnit.ELECTRONVOLT,
    "oxidation_selectivity_ev": CanonicalUnit.ELECTRONVOLT,
    "association_energy_ev": CanonicalUnit.ELECTRONVOLT,
    "li_coordination_objective_ev": CanonicalUnit.ELECTRONVOLT,
    "temperature_k": CanonicalUnit.KELVIN,
    "pressure_bar": CanonicalUnit.BAR,
    "additive_mass_fraction": CanonicalUnit.DIMENSIONLESS,
    "ion_charge_scale": CanonicalUnit.DIMENSIONLESS,
    "equilibration_time_ns": CanonicalUnit.NANOSECOND,
    "production_time_ns": CanonicalUnit.NANOSECOND,
    "distance": CanonicalUnit.ANGSTROM,
    "density": CanonicalUnit.GRAM_PER_MILLILITER,
    "li_self_diffusion": CanonicalUnit.SQUARE_CENTIMETER_PER_SECOND,
    "li_diffusion_ratio": CanonicalUnit.DIMENSIONLESS,
    "radial_distribution_function": CanonicalUnit.DIMENSIONLESS,
    "coordination_number": CanonicalUnit.DIMENSIONLESS,
    "ec_coordination_number": CanonicalUnit.DIMENSIONLESS,
    "emc_coordination_number": CanonicalUnit.DIMENSIONLESS,
    "additive_coordination_number": CanonicalUnit.DIMENSIONLESS,
    "carbonate_coordination_number": CanonicalUnit.DIMENSIONLESS,
    "carbonate_displacement": CanonicalUnit.DIMENSIONLESS,
    "solvent_shell_fraction": CanonicalUnit.DIMENSIONLESS,
    "residence_time": CanonicalUnit.NANOSECOND,
    "pf6_contact_pair_fraction": CanonicalUnit.DIMENSIONLESS,
    "additive_cluster_size": CanonicalUnit.DIMENSIONLESS,
}


class SchemaModel(BaseModel):
    """Shared validation policy for persistent CEI-Scout records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ProvenanceRecord(SchemaModel):
    """Source identity and transformation history for scientific data."""

    provenance_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_doi: str | None = None
    source_url: str | None = None
    license_id: str = Field(min_length=1)
    retrieval_date: date
    source_version: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformation: str = Field(min_length=1)
    parent_provenance_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_source_locator(self) -> Self:
        """Require at least one persistent source locator."""
        if not self.source_doi and not self.source_url:
            raise ValueError("provenance requires a source DOI or URL")
        return self


class CandidateRecord(SchemaModel):
    """Identity and eligibility state for one candidate molecule."""

    provenance_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    canonical_smiles: str | None = None
    inchi: str | None = None
    inchikey: str | None = None
    pubchem_cid: int | None = Field(default=None, ge=1)
    molecular_formula: str | None = None
    molecular_weight_g_mol: float | None = Field(default=None, gt=0)
    formal_charge: int | None = None
    spin_multiplicity: int | None = Field(default=None, ge=1)
    candidate_family: CandidateFamily | None = None
    eligibility_status: Literal["pending", "accepted", "rejected"] = "pending"
    rejection_reasons: tuple[RejectionReason, ...] = ()

    @model_validator(mode="after")
    def validate_eligibility(self) -> Self:
        """Require rejection reasons only for rejected candidates."""
        if self.eligibility_status == "accepted" and self.rejection_reasons:
            raise ValueError("accepted candidates cannot have rejection reasons")
        if self.eligibility_status == "rejected" and not self.rejection_reasons:
            raise ValueError("rejected candidates require at least one reason")
        return self


class CalculationRecord(SchemaModel):
    """One auditable xTB or DFT calculation attempt."""

    provenance_id: str = Field(min_length=1)
    calculation_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    engine: Literal["xtb", "pyscf"]
    job_type: Literal[
        "geometry_optimization",
        "neutral_single_point",
        "cation_single_point",
        "li_complex_single_point",
    ]
    method: str = Field(min_length=1)
    basis_set: str | None = None
    solvent_model: str | None = None
    dielectric: float | None = Field(default=None, gt=0)
    charge: int
    spin_multiplicity: int = Field(ge=1)
    input_geometry_path: str = Field(min_length=1)
    attempt: int = Field(default=1, ge=1)
    status: Literal["pending", "running", "succeeded", "failed"] = "pending"
    electronic_energy_hartree: float | None = None
    output_geometry_path: str | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Require the result associated with a terminal status."""
        if self.status == "succeeded" and self.electronic_energy_hartree is None:
            raise ValueError("successful calculations require an electronic energy")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed calculations require a failure reason")
        return self


class ActiveLearningRoundRecord(SchemaModel):
    """State frozen at the beginning of one active-learning round."""

    provenance_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    observed_candidate_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    batch_size: int = Field(ge=1)
    random_seed: int = Field(ge=0)
    reference_point: tuple[float, float]
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        """Require the frozen batch to contain only unobserved candidates."""
        if len(self.selected_candidate_ids) != self.batch_size:
            raise ValueError("selected candidate count must equal batch_size")
        if set(self.observed_candidate_ids) & set(self.selected_candidate_ids):
            raise ValueError("selected candidates must be unobserved")
        return self


class AcquisitionRecord(SchemaModel):
    """Prediction and acquisition score recorded before label observation."""

    provenance_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    candidate_id: str = Field(min_length=1)
    oxidation_selectivity_mean_ev: float
    oxidation_selectivity_std_ev: float = Field(ge=0)
    li_coordination_objective_mean_ev: float
    li_coordination_objective_std_ev: float = Field(ge=0)
    acquisition_value: float
    acquisition_rank: int = Field(ge=1)
    selected: bool


class MDSystemRecord(SchemaModel):
    """Composition and force-field identity for one MD system."""

    provenance_id: str = Field(min_length=1)
    system_id: str = Field(min_length=1)
    system_role: Literal["baseline", "finalist"]
    additive_candidate_id: str | None = None
    additive_mass_fraction: float = Field(default=0.0, ge=0, lt=1)
    temperature_k: float = Field(gt=0)
    pressure_bar: float = Field(gt=0)
    force_field_family: Literal["openff-sage", "gaff2"]
    ion_charge_scale: float = Field(default=0.8, gt=0, le=1)
    status: Literal[
        "planned",
        "parameterized",
        "running",
        "completed",
        "failed",
    ] = "planned"

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        """Keep baseline and additive-system compositions distinct."""
        if self.system_role == "baseline":
            if (
                self.additive_candidate_id is not None
                or self.additive_mass_fraction != 0
            ):
                raise ValueError("baseline systems cannot contain an additive")
        elif self.additive_candidate_id is None or self.additive_mass_fraction <= 0:
            raise ValueError(
                "finalist systems require an additive and positive fraction"
            )
        return self


class MDReplicaRecord(SchemaModel):
    """Execution state for one statistically independent MD replica."""

    provenance_id: str = Field(min_length=1)
    replica_id: str = Field(min_length=1)
    system_id: str = Field(min_length=1)
    replica_index: int = Field(ge=1)
    random_seed: int = Field(ge=0)
    equilibration_time_ns: float = Field(ge=0)
    production_time_ns: float = Field(ge=0)
    status: Literal["planned", "running", "completed", "failed"] = "planned"
    trajectory_path: str | None = None
    checkpoint_path: str | None = None
    failure_reason: str | None = None


class MDObservableRecord(SchemaModel):
    """One MD observable with uncertainty and analysis provenance."""

    provenance_id: str = Field(min_length=1)
    observable_id: str = Field(min_length=1)
    system_id: str = Field(min_length=1)
    replica_id: str | None = None
    observable_name: str = Field(min_length=1)
    value: float
    unit: CanonicalUnit
    uncertainty: float | None = Field(default=None, ge=0)
    uncertainty_method: str | None = None
    analysis_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_canonical_unit(self) -> Self:
        """Require every registered MD property to use its canonical unit."""
        expected = CANONICAL_PROPERTY_UNITS.get(self.observable_name)
        if expected is None:
            raise ValueError("observable_name has no registered canonical unit")
        if self.unit != expected:
            raise ValueError(
                f"{self.observable_name} must use canonical unit {expected.value}"
            )
        return self
