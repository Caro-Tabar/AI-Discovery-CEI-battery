"""Scientific integrity tests for Phase 9 candidate ranking."""

from pathlib import Path

import pytest

from cei_scout.ml.ranking import build_ranking, ordinal_ranks

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9_ranking_v1.yaml"


@pytest.fixture(scope="module")
def ranking():
    return build_ranking(CONFIG)


def test_frozen_pool_and_exclusions(ranking):
    rows, metadata = ranking
    complete = [r for r in rows if r["dft_status"] == "complete"]
    incomplete = [r for r in rows if r["dft_status"] == "incomplete"]

    assert len(rows) == 60
    assert len(complete) == 56
    assert {r["candidate_id"] for r in incomplete} == {
        "CEI-0021",
        "CEI-0051",
        "CEI-0059",
        "CEI-0060",
    }
    assert all(r["dual_redox_priority_rank"] == "" for r in incomplete)
    assert all(r["unconverged_state"] for r in incomplete)
    assert metadata["population"]["dft_complete"] == 56


def test_ranks_are_complete_and_directionally_correct(ranking):
    rows, _ = ranking
    complete = [r for r in rows if r["dft_status"] == "complete"]

    for key in (
        "oxidation_propensity_rank",
        "electron_attachment_rank",
        "oxidation_resistance_rank",
        "dual_redox_priority_rank",
    ):
        assert sorted(r[key] for r in complete) == list(range(1, 57))

    easiest_oxidation = min(complete, key=lambda r: r["vertical_ionization_energy_ev"])
    strongest_attachment = max(
        complete, key=lambda r: r["vertical_electron_affinity_ev"]
    )

    assert easiest_oxidation["oxidation_propensity_rank"] == 1
    assert strongest_attachment["electron_attachment_rank"] == 1

    ordered = sorted(
        complete,
        key=lambda r: r["dual_redox_priority_rank"],
    )
    assert [
        (r["dual_redox_rank_sum"], r["oxidation_propensity_rank"]) for r in ordered
    ] == sorted(
        (r["dual_redox_rank_sum"], r["oxidation_propensity_rank"]) for r in complete
    )


def test_qc_flags_are_preserved(ranking):
    rows, _ = ranking
    indexed = {r["candidate_id"]: r for r in rows}

    rhodanine = indexed["CEI-0054"]
    assert rhodanine["dual_redox_priority_rank"] == 1
    assert rhodanine["manual_review_flag"] == "True"

    captopril = indexed["CEI-0053"]
    assert int(captopril["qc_li_connectivity_change_count"]) >= 1


def test_ties_use_candidate_id():
    sample = [
        {"candidate_id": "CEI-B", "value": "2.0"},
        {"candidate_id": "CEI-A", "value": "2.0"},
    ]
    result = ordinal_ranks(sample, "value")
    assert result == {"CEI-A": 1, "CEI-B": 2}
