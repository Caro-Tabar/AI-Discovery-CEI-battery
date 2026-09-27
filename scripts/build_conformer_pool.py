"""Build the Phase 7 conformer pool for all frozen candidates."""

from pathlib import Path

from cei_scout.preprocessing.candidate_conformers import (
    load_candidate_pool,
    preprocess_candidate_pool,
    write_conformer_pool_outputs,
)
from cei_scout.preprocessing.config import (
    load_conformer_pipeline_config,
)

ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_PATH = ROOT / "data/processed/candidates_v1.parquet"
CONFIG_PATH = ROOT / "configs/conformers.yaml"
OUTPUT_DIRECTORY = ROOT / "results/conformers_v1"


def main() -> None:
    config = load_conformer_pipeline_config(CONFIG_PATH)

    candidates = load_candidate_pool(
        CANDIDATE_PATH,
        expected_count=60,
    )

    artifacts = preprocess_candidate_pool(
        candidates,
        config,
    )

    write_conformer_pool_outputs(
        artifacts,
        OUTPUT_DIRECTORY,
    )

    failed = tuple(
        artifact.result for artifact in artifacts if artifact.result.status == "failed"
    )

    selected_count = sum(
        artifact.result.selected_conformer_count for artifact in artifacts
    )

    print(f"Candidates processed: {len(artifacts)}")
    print(f"Selected conformers: {selected_count}")
    print(f"Failed candidates: {len(failed)}")

    for result in failed:
        print(f"FAILED {result.candidate_id}: {result.error}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
