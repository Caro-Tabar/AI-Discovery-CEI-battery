"""Tests for GPR ranking stability and uncertainty diagnostics."""

from copy import deepcopy
from pathlib import Path

import pytest

from cei_scout.ml.uncertainty_ranking import (
    analyze,
    composite_ranks,
    read_csv,
)

RESULTS = Path(__file__).resolve().parents[1] / "results/phase9_v1"


@pytest.fixture(scope="module")
def source_data():
    ranking = read_csv(RESULTS / "species_ranking_v1.csv")
    predictions = read_csv(RESULTS / "extended_predictions_v1.csv")
    return ranking, predictions


def test_composite_rank_direction_and_ties():
    ie = {"CEI-A": 1.0, "CEI-B": 2.0, "CEI-C": 3.0}
    ea = {"CEI-A": 0.0, "CEI-B": 1.0, "CEI-C": 2.0}

    assert composite_ranks(ie, ea) == {
        "CEI-A": 1,
        "CEI-B": 2,
        "CEI-C": 3,
    }


def test_population_and_rank_stability(source_data):
    stability, diagnostics = analyze(*source_data)

    assert len(stability) == 60
    assert len(diagnostics) == 28

    complete = [row for row in stability if row["dft_status"] == "complete"]
    incomplete = [row for row in stability if row["dft_status"] == "incomplete"]

    assert len(complete) == 56
    assert len(incomplete) == 4

    assert sorted(int(row["dft_dual_redox_rank"]) for row in complete) == list(
        range(1, 57)
    )

    assert all(row["gpr_median_rank"] == "" for row in incomplete)

    for row in complete:
        assert 1 <= row["gpr_median_rank"] <= 56
        assert 0 <= row["gpr_top10_frequency"] <= 1
        assert row["gpr_rank_range"] >= 0
        assert row["gpr_mean_ie_std_ev"] >= 0
        assert row["gpr_mean_ea_std_ev"] >= 0

    for row in diagnostics:
        assert 0 <= row["empirical_95pct_coverage"] <= 1
        assert row["mean_predictive_std_ev"] >= 0


def test_rhodanine_retains_qc_flag(source_data):
    stability, _ = analyze(*source_data)
    indexed = {row["candidate_id"]: row for row in stability}

    rhodanine = indexed["CEI-0054"]

    assert int(rhodanine["dft_dual_redox_rank"]) == 1
    assert rhodanine["manual_review_flag"] == "True"


def test_mismatched_dft_reference_rejected(source_data):
    ranking, predictions = source_data
    modified = deepcopy(predictions)

    target = next(row for row in modified if row["model"] == "gaussian_process")
    target["dft_value_ev"] = "9999.0"

    with pytest.raises(ValueError, match="reference mismatch"):
        analyze(ranking, modified)


def test_duplicate_gpr_prediction_rejected(source_data):
    ranking, predictions = source_data
    modified = list(predictions)

    target = next(row for row in predictions if row["model"] == "gaussian_process")
    modified.append(target.copy())

    with pytest.raises(ValueError, match="Duplicate held-out"):
        analyze(ranking, modified)


def test_missing_gpr_predictions_rejected(source_data):
    ranking, predictions = source_data
    modified = [
        row
        for row in predictions
        if not (
            row["model"] == "gaussian_process"
            and row["validation"] == "primary"
            and row["target"] == "ionization_energy"
            and row["repeat"] == "1"
        )
    ]

    with pytest.raises(ValueError, match="Incomplete"):
        analyze(ranking, modified)
