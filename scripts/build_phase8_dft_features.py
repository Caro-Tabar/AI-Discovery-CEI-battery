"""Build complete Phase 8 DFT features from screening calculations."""

from pathlib import Path

import pandas as pd
import yaml
from pyscf.data.nist import HARTREE2EV

RESULTS_PATH = Path("results/phase8_v1/dft_state_results.csv")
EXCLUSIONS_PATH = Path("configs/phase8_dft_exclusions_v1.yaml")
OUTPUT_PATH = Path("results/phase8_v1/dft_features_v1.csv")

REQUIRED_STATES = {"neutral", "oxidized", "reduced"}


def load_exclusions(path: Path) -> set[str]:
    """Load candidate IDs intentionally excluded from DFT feature construction."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return set(config["candidates"])


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)

    # Keep only the latest recorded attempt for every candidate/state.
    latest = (
        df.sort_values("attempt")
        .groupby(["candidate_id", "state"], as_index=False)
        .tail(1)
        .copy()
    )

    exclusions = load_exclusions(EXCLUSIONS_PATH)

    eligible = latest[~latest["candidate_id"].isin(exclusions)].copy()

    # Every remaining candidate must have all three required states.
    state_sets = eligible.groupby("candidate_id")["state"].agg(set)

    incomplete_states = state_sets[state_sets != REQUIRED_STATES]

    if not incomplete_states.empty:
        raise RuntimeError(
            "Unexpected candidates with incomplete state coverage:\n"
            f"{incomplete_states}"
        )

    # Every remaining state must actually be converged.
    unsuccessful = eligible[eligible["status"] != "success"]

    if not unsuccessful.empty:
        raise RuntimeError(
            "Unexpected non-successful DFT states remain outside the "
            "configured exclusion list:\n"
            f"{unsuccessful[['candidate_id', 'state', 'status']]}"
        )

    # Ensure each candidate used one geometry for all electronic states.
    hash_counts = eligible.groupby("candidate_id")["xyz_sha256"].nunique()

    if (hash_counts != 1).any():
        raise RuntimeError(
            "At least one candidate used inconsistent geometries across states."
        )

    energy = eligible.pivot(
        index="candidate_id",
        columns="state",
        values="energy_eh",
    )

    s2 = eligible.pivot(
        index="candidate_id",
        columns="state",
        values="s2",
    )

    runtime = eligible.pivot(
        index="candidate_id",
        columns="state",
        values="runtime_seconds",
    )

    features = pd.DataFrame(index=energy.index)

    features["energy_neutral_eh"] = energy["neutral"]
    features["energy_oxidized_eh"] = energy["oxidized"]
    features["energy_reduced_eh"] = energy["reduced"]

    # Vertical ionization: M -> M+ + e-
    features["vertical_ionization_energy_ev"] = (
        energy["oxidized"] - energy["neutral"]
    ) * HARTREE2EV

    # Raw energy change for electron attachment: M + e- -> M-
    features["vertical_electron_attachment_delta_e_ev"] = (
        energy["reduced"] - energy["neutral"]
    ) * HARTREE2EV

    # Conventional positive electron-affinity sign:
    # EA = E(neutral) - E(anion)
    features["vertical_electron_affinity_ev"] = (
        energy["neutral"] - energy["reduced"]
    ) * HARTREE2EV

    features["oxidized_s2"] = s2["oxidized"]
    features["reduced_s2"] = s2["reduced"]

    features["runtime_neutral_s"] = runtime["neutral"]
    features["runtime_oxidized_s"] = runtime["oxidized"]
    features["runtime_reduced_s"] = runtime["reduced"]

    provenance = eligible.groupby("candidate_id").agg(
        xyz_sha256=("xyz_sha256", "first"),
        functional=("functional", "first"),
        basis=("basis", "first"),
        pcm_model=("pcm_model", "first"),
        dielectric=("dielectric", "first"),
        grid_level=("grid_level", "first"),
    )

    features = features.join(provenance).reset_index()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print("Original candidates: 60")
    print(f"Configured DFT exclusions: {len(exclusions)}")
    print(f"DFT-complete candidates: {len(features)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nExcluded candidates:")
    for candidate_id in sorted(exclusions):
        print(f"  {candidate_id}")


if __name__ == "__main__":
    main()
