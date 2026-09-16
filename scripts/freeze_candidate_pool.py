"""Freeze and summarize the CEI-Scout 60-candidate search space."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/interim/candidate_registry_resolved_v0.csv")
OUTPUT_PATH = Path("data/processed/candidates_v1.parquet")
SUMMARY_PATH = Path("results/tables/candidate_pool_summary.csv")

POOL_SIZE = 60
ORIGINAL_MAX_ROTATABLE_BONDS = 8
RELAXED_MAX_ROTATABLE_BONDS = 10
SELECTION_POLICY_VERSION = "cei-scout-family-balance-v1"

FAMILY_ORDER = [
    "sulfite",
    "sulfate_or_sultone",
    "sulfone_or_sulfoxide",
    "thioether_or_disulfide",
    "sulfonate_or_sulfonamide",
    "mixed_s_o_n",
]

BENCHMARK_CIDS = {
    14075: "Ethylene sulfate",
    31347: "Sulfolane",
    77342: "Ethylene sulfite",
    14264: "1,3-Propane sultone",
    77839: "Trimethylene sulfite",
    69223: "Dimethyl sulfite",
    12197: "Diethyl sulfite",
}

SAFETY_ALERT = "consult_current_sds_before_any_handling"
SAFETY_NOTE = (
    "Computational screening entry only; not an experimental-use recommendation. "
    "Review the current supplier SDS and institutional safety requirements before "
    "any acquisition or handling."
)


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejection_reasons(value: str) -> set[str]:
    """Parse the semicolon-delimited rejection-reason field."""
    return {reason for reason in value.split(";") if reason}


def prepare_selection_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Apply the sole registered fallback only when fewer than 60 are eligible."""
    prepared = frame.copy()
    prepared["molecular_weight_g_mol"] = pd.to_numeric(
        prepared["molecular_weight_g_mol"],
        errors="raise",
    )
    prepared["rotatable_bonds"] = pd.to_numeric(
        prepared["rotatable_bonds"],
        errors="raise",
    ).astype("int64")
    prepared["candidate_relaxed_for_selection"] = False
    prepared["selection_eligible"] = prepared["filter_status"].eq("eligible_prebalance")

    eligible_count = int(prepared["selection_eligible"].sum())
    relaxation_used = False

    if eligible_count < POOL_SIZE:
        relaxation_used = True
        needed = POOL_SIZE - eligible_count

        relaxable_indices: list[int] = []
        for index, row in prepared.iterrows():
            reasons = rejection_reasons(str(row["rejection_reasons"]))
            rotatable_bonds = int(row["rotatable_bonds"])

            if (
                reasons == {f"rotatable_bonds_above_{ORIGINAL_MAX_ROTATABLE_BONDS}"}
                and rotatable_bonds <= RELAXED_MAX_ROTATABLE_BONDS
            ):
                relaxable_indices.append(index)

        relaxable_indices.sort(
            key=lambda index: (
                str(prepared.at[index, "candidate_family"]),
                str(prepared.at[index, "source_entry_id"]),
            )
        )

        for index in relaxable_indices[:needed]:
            prepared.at[index, "selection_eligible"] = True
            prepared.at[index, "candidate_relaxed_for_selection"] = True

    final_eligible_count = int(prepared["selection_eligible"].sum())

    if final_eligible_count < POOL_SIZE:
        raise RuntimeError(
            "The registered rotatable-bond fallback is insufficient: "
            f"only {final_eligible_count} candidates are eligible, but "
            f"{POOL_SIZE} are required. Do not relax another filter."
        )

    return prepared, relaxation_used


def select_balanced_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """Select 60 candidates by deterministic round-robin family balancing."""
    unknown_families = set(frame["candidate_family"]) - set(FAMILY_ORDER)

    if unknown_families:
        raise ValueError(f"Unknown candidate families: {sorted(unknown_families)}")

    family_rank = {family: rank for rank, family in enumerate(FAMILY_ORDER)}
    buckets: dict[str, list[int]] = {}

    for family in FAMILY_ORDER:
        family_rows = frame.loc[
            frame["selection_eligible"] & frame["candidate_family"].eq(family)
        ].sort_values(
            ["source_entry_id", "standardized_inchikey"],
            kind="stable",
        )
        buckets[family] = list(family_rows.index)

    selected_indices: list[int] = []
    selected_counts: Counter[str] = Counter()

    while len(selected_indices) < POOL_SIZE:
        available = [family for family in FAMILY_ORDER if buckets[family]]

        if not available:
            raise RuntimeError("Eligible candidate buckets were exhausted early")

        family = min(
            available,
            key=lambda name: (selected_counts[name], family_rank[name]),
        )
        selected_indices.append(buckets[family].pop(0))
        selected_counts[family] += 1

    selected = frame.loc[selected_indices].copy()
    selected["family_rank"] = selected["candidate_family"].map(family_rank)
    selected = selected.sort_values(
        ["family_rank", "standardized_inchikey", "source_entry_id"],
        kind="stable",
    ).reset_index(drop=True)

    selected["candidate_id"] = [
        f"CEI-{number:04d}" for number in range(1, POOL_SIZE + 1)
    ]

    return selected.drop(columns=["family_rank", "selection_eligible"])


def add_selection_and_safety_metadata(
    frame: pd.DataFrame,
    *,
    input_hash: str,
    relaxation_used: bool,
) -> pd.DataFrame:
    """Add provenance and non-prescriptive safety-review metadata."""
    enriched = frame.copy()
    enriched["registry_version"] = "candidates_v1"
    enriched["selection_policy_version"] = SELECTION_POLICY_VERSION
    enriched["selection_input_sha256"] = input_hash
    enriched["selection_status"] = "selected"
    pubchem_cids = pd.to_numeric(
        enriched["pubchem_cid"],
        errors="raise",
    ).astype("int64")
    enriched["registry_role"] = pubchem_cids.map(
        lambda cid: "both" if cid in BENCHMARK_CIDS else "candidate"
    )
    enriched["effective_filter_status"] = enriched[
        "candidate_relaxed_for_selection"
    ].map(
        {
            False: "eligible_under_primary_filters",
            True: "eligible_via_registered_rotatable_bond_relaxation",
        }
    )
    enriched["selection_basis"] = (
        "eligibility_then_family_balance_then_stable_source_identity"
    )
    enriched["pool_relaxation_used"] = relaxation_used
    enriched["relaxation_policy"] = (
        f"rotatable_bonds_max_{RELAXED_MAX_ROTATABLE_BONDS}"
        if relaxation_used
        else "not_used"
    )

    enriched["safety_alert"] = SAFETY_ALERT
    enriched["safety_review_required"] = True
    enriched["experimental_use_recommendation"] = False
    enriched["safety_metadata_scope"] = "metadata_only_not_selection_feature"
    enriched["safety_information_url"] = enriched["source_url"].map(
        lambda url: f"{url}#section=Safety-and-Hazards"
    )
    enriched["safety_note"] = SAFETY_NOTE

    return enriched


def summary_row(
    frame: pd.DataFrame,
    scope: str,
    family: str,
) -> dict[str, object]:
    """Calculate one overall or family-level summary row."""
    weights = pd.to_numeric(frame["molecular_weight_g_mol"], errors="raise")
    rotatable = pd.to_numeric(frame["rotatable_bonds"], errors="raise")

    return {
        "scope": scope,
        "candidate_family": family,
        "candidate_count": len(frame),
        "unique_inchikey_count": frame["standardized_inchikey"].nunique(),
        "molecular_weight_min_g_mol": round(float(weights.min()), 3),
        "molecular_weight_median_g_mol": round(float(weights.median()), 3),
        "molecular_weight_max_g_mol": round(float(weights.max()), 3),
        "rotatable_bonds_min": int(rotatable.min()),
        "rotatable_bonds_median": float(rotatable.median()),
        "rotatable_bonds_max": int(rotatable.max()),
        "relaxed_candidate_count": int(frame["candidate_relaxed_for_selection"].sum()),
    }


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the tracked overall and per-family chemical-space summary."""
    rows = [summary_row(frame, "overall", "all")]
    rows.extend(
        summary_row(
            frame.loc[frame["candidate_family"].eq(family)],
            "family",
            family,
        )
        for family in FAMILY_ORDER
    )

    return pd.DataFrame(rows)


def write_outputs(
    pool: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write both outputs atomically."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_parquet = OUTPUT_PATH.with_suffix(".tmp.parquet")
    temporary_summary = SUMMARY_PATH.with_suffix(".tmp.csv")

    pool.to_parquet(temporary_parquet, index=False, engine="pyarrow")
    summary.to_csv(temporary_summary, index=False, lineterminator="\n")

    temporary_parquet.replace(OUTPUT_PATH)
    temporary_summary.replace(SUMMARY_PATH)


def main() -> None:
    """Freeze and summarize the candidate pool."""
    frame = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)

    if len(frame) != 64:
        raise ValueError(f"Expected 64 interim rows; found {len(frame)}")

    prepared, relaxation_used = prepare_selection_frame(frame)
    pool = select_balanced_pool(prepared)
    pool = add_selection_and_safety_metadata(
        pool,
        input_hash=sha256(INPUT_PATH),
        relaxation_used=relaxation_used,
    )
    summary = build_summary(pool)
    write_outputs(pool, summary)

    print(f"frozen_candidates: {len(pool)}")
    print(f"family_counts: {dict(Counter(pool['candidate_family']))}")
    print(f"relaxation_used: {relaxation_used}")
    print(f"parquet: {OUTPUT_PATH}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
