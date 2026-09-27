"""Tests for full candidate conformer preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cei_scout.preprocessing.candidate_conformers import (
    CandidateInput,
    load_candidate_pool,
    preprocess_candidate,
)
from cei_scout.preprocessing.config import (
    load_conformer_pipeline_config,
)


def test_load_candidate_pool_preserves_frozen_identity(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "candidates.parquet"

    pd.DataFrame(
        [
            {
                "candidate_id": "CEI-0002",
                "canonical_smiles": "CCSC",
                "candidate_family": "thioether_or_disulfide",
                "source_entry_id": "source-002",
                "registry_version": "candidates_v1",
                "selection_status": "selected",
            },
            {
                "candidate_id": "CEI-0001",
                "canonical_smiles": "CS(=O)C",
                "candidate_family": "sulfone_or_sulfoxide",
                "source_entry_id": "source-001",
                "registry_version": "candidates_v1",
                "selection_status": "selected",
            },
        ]
    ).to_parquet(
        table_path,
        index=False,
    )

    candidates = load_candidate_pool(
        table_path,
        expected_count=2,
    )

    assert tuple(candidate.candidate_id for candidate in candidates) == (
        "CEI-0001",
        "CEI-0002",
    )


def test_candidate_preprocessing_retains_at_most_three(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "conformers.yaml"

    config_path.write_text(
        """
schema_version: "1.0"

embedding:
  method: ETKDGv3
  random_seed: 20260925
  num_conformers: 20
  prune_rms_threshold_angstrom: 0.5
  num_threads: 1
  enforce_chirality: true
  use_small_ring_torsions: true
  use_macrocycle_torsions: true
  use_macrocycle_14config: true
  only_heavy_atoms_for_pruning: true

prescreen:
  primary_force_field: MMFF94s
  fallback_force_field: UFF
  fallback_if_mmff_unavailable: true
  max_iterations: 500

selection:
  max_selected_conformers: 3
  min_heavy_atom_rmsd_angstrom: 0.5
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_conformer_pipeline_config(config_path)

    candidate = CandidateInput(
        candidate_id="CEI-TEST",
        canonical_smiles="CCOC(=O)OC",
        candidate_family="test",
        source_entry_id="test-source",
        registry_version="test",
    )

    artifact = preprocess_candidate(
        candidate,
        config,
    )

    assert artifact.result.status == "ok"

    selected = artifact.result.selected

    assert 1 <= len(selected) <= 3

    assert tuple(conformer.selection_rank for conformer in selected) == tuple(
        range(1, len(selected) + 1)
    )

    for conformer in selected[1:]:
        assert conformer.min_rmsd_to_selected_angstrom is not None

        assert conformer.min_rmsd_to_selected_angstrom >= 0.5
