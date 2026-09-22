"""Tests for the frozen CEI-Scout candidate registry."""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data/processed/candidates_v1.parquet"
REVIEW_PATH = ROOT / "data/processed/candidate_review_v1.csv"
BENCHMARK_PATH = ROOT / "data/processed/benchmark_registry_v1.csv"


def load_pool() -> pd.DataFrame:
    """Load the frozen candidate pool."""
    return pd.read_parquet(POOL_PATH)


def test_candidate_registry_has_exact_unique_identity() -> None:
    pool = load_pool()

    assert len(pool) == 60
    assert pool["candidate_id"].tolist() == [
        f"CEI-{number:04d}" for number in range(1, 61)
    ]
    assert pool["candidate_id"].is_unique
    assert pool["standardized_inchikey"].is_unique


def test_candidate_registry_obeys_chemical_scope() -> None:
    pool = load_pool()
    allowed_elements = {"C", "H", "N", "O", "S"}

    for value in pool["elements"]:
        observed = set(re.findall(r"[A-Z][a-z]?", str(value)))
        assert observed
        assert observed <= allowed_elements

    charges = pd.to_numeric(pool["formal_charge"], errors="raise")
    weights = pd.to_numeric(
        pool["molecular_weight_g_mol"],
        errors="raise",
    )

    assert charges.eq(0).all()
    assert weights.between(70, 220, inclusive="both").all()


def test_candidate_registry_roles_and_safety_metadata() -> None:
    pool = load_pool()

    assert set(pool["registry_role"]) <= {"candidate", "both"}
    assert pool["safety_review_required"].eq(True).all()
    assert pool["experimental_use_recommendation"].eq(False).all()


def test_manual_review_is_complete() -> None:
    review = pd.read_csv(
        REVIEW_PATH,
        dtype=str,
        keep_default_na=False,
    )

    assert len(review) == 60
    assert review["candidate_id"].is_unique
    assert review["standardized_inchikey"].is_unique
    assert review["review_status"].eq("passed").all()
    assert review["reviewer"].str.len().gt(0).all()
    assert review["review_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()


def test_all_preregistered_benchmarks_have_roles() -> None:
    benchmarks = pd.read_csv(BENCHMARK_PATH)
    expected_cids = {
        14075,
        31347,
        77342,
        14264,
        77839,
        69223,
        12197,
    }

    assert len(benchmarks) == 7
    assert set(benchmarks["pubchem_cid"]) == expected_cids
    assert benchmarks["standardized_inchikey"].is_unique
    assert set(benchmarks["registry_role"]) <= {
        "benchmark_only",
        "both",
    }
