"""Deterministic ordinary and chemical-family-held-out CV splits."""

from dataclasses import dataclass
from typing import Any, Literal

from sklearn.model_selection import LeaveOneGroupOut, RepeatedKFold

from cei_scout.ml.dataset import MLDataset


@dataclass(frozen=True, slots=True)
class CVFold:
    """Candidate row positions for one evaluation fold."""

    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    held_out_family: str | None


def make_cv_folds(
    dataset: MLDataset,
    config: dict[str, Any],
    scheme: Literal["primary", "family"],
) -> tuple[CVFold, ...]:
    """Generate splits without preprocessing or using DFT targets."""
    settings = config["validation"]

    if scheme == "primary":
        spec = settings["primary"]

        cv = RepeatedKFold(
            n_splits=int(spec["n_splits"]),
            n_repeats=int(spec["n_repeats"]),
            random_state=int(settings["random_seed"]),
        )
        splits = cv.split(dataset.X)

    elif scheme == "family":
        cv = LeaveOneGroupOut()
        splits = cv.split(dataset.X, groups=dataset.families)

    else:
        raise ValueError(f"Unknown CV scheme: {scheme}")

    folds: list[CVFold] = []

    for train, test in splits:
        family = None

        if scheme == "family":
            held_out = {dataset.families[int(i)] for i in test}
            if len(held_out) != 1:
                raise ValueError("Family-held-out fold spans multiple families")
            family = held_out.pop()

        folds.append(
            CVFold(
                train_indices=tuple(int(i) for i in train),
                test_indices=tuple(int(i) for i in test),
                held_out_family=family,
            )
        )

    return tuple(folds)
