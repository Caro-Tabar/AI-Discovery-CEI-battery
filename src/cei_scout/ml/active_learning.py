"""Retrospective, two-target GPR active-learning benchmark."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from cei_scout.ml.baselines import calculate_metrics
from cei_scout.ml.dataset import MLDataset, load_ml_dataset
from cei_scout.ml.extended_models import predict_outer_fold

TARGETS = ("ionization_energy", "electron_affinity")


def make_partition(
    dataset: MLDataset,
    seed: int,
    test_size: int,
    initial_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Stratify twice without consulting DFT target values."""
    families = np.asarray(dataset.families)

    outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    available, test = next(outer.split(dataset.X, families))

    inner = StratifiedShuffleSplit(
        n_splits=1,
        train_size=initial_size,
        random_state=seed + 10000,
    )
    initial_local, pool_local = next(
        inner.split(dataset.X[available], families[available])
    )

    initial = tuple(sorted(int(i) for i in available[initial_local]))
    pool = tuple(sorted(int(i) for i in available[pool_local]))
    test_ids = tuple(sorted(int(i) for i in test))

    a, b, c = set(initial), set(pool), set(test_ids)
    if a & b or a & c or b & c or a | b | c != set(range(len(dataset.candidate_ids))):
        raise ValueError("Partition overlap or missing candidates")

    return initial, pool, test_ids


def select_uncertain(
    dataset: MLDataset,
    labeled: tuple[int, ...],
    pool: tuple[int, ...],
    uncertainties: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[tuple[int, ...], dict[int, float]]:
    """Select molecules by combined target-normalized GPR uncertainty."""
    scores = np.zeros(len(pool), dtype=float)

    for target in TARGETS:
        sigma = np.asarray(uncertainties[target], dtype=float)

        if sigma.shape != (len(pool),) or not np.isfinite(sigma).all():
            raise ValueError("Invalid pool uncertainty")

        if np.any(sigma < 0):
            raise ValueError("Negative pool uncertainty")

        target_scale = max(
            float(np.std(dataset.targets[target][list(labeled)])),
            1e-8,
        )
        scores += (sigma / target_scale) ** 2

    scores = np.sqrt(scores)

    ordered = sorted(
        range(len(pool)),
        key=lambda j: (
            -float(scores[j]),
            dataset.candidate_ids[pool[j]],
        ),
    )

    chosen = tuple(pool[j] for j in ordered[:batch_size])
    score_map = {pool[j]: float(scores[j]) for j in ordered[:batch_size]}
    return chosen, score_map


def run_experiment(
    dataset: MLDataset,
    extended: dict,
    config: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Simulate acquisition without exposing pool/test DFT labels."""
    settings = config["experiment"]
    seeds = settings["seeds"]
    rounds = int(settings["acquisition_rounds"])
    batch_size = int(settings["acquisition_batch_size"])
    strategies = settings["strategies"]

    if (
        len(set(seeds)) != len(seeds)
        or strategies != ["uncertainty", "random"]
        or settings["acquisition"] != "normalized_joint_gpr_std"
        or settings["oracle"] != "existing_dft_labels_only"
        or settings["additional_dft"] is not False
        or batch_size < 1
        or rounds < 1
    ):
        raise ValueError("Unsupported active-learning configuration")

    curves: list[dict] = []
    acquisitions: list[dict] = []
    manifest: list[dict] = []

    for seed in seeds:
        initial, original_pool, test = make_partition(
            dataset,
            int(seed),
            int(settings["test_size"]),
            int(settings["initial_labeled_size"]),
        )

        roles = {
            **{i: "initial_labeled" for i in initial},
            **{i: "acquisition_pool" for i in original_pool},
            **{i: "held_out_test" for i in test},
        }

        for i in sorted(roles):
            manifest.append(
                {
                    "seed": seed,
                    "candidate_id": dataset.candidate_ids[i],
                    "candidate_family": dataset.families[i],
                    "initial_role": roles[i],
                }
            )

        if rounds * batch_size > len(original_pool):
            raise ValueError("Acquisition budget exceeds pool")

        for strategy in strategies:
            labeled = list(initial)
            pool = list(original_pool)
            rng = np.random.default_rng(int(seed) + 20000)

            for round_number in range(rounds + 1):
                # All queries are unlabeled for fitting purposes.
                # Test entries come first solely for slicing predictions.
                queries = (*test, *pool)
                n_test = len(test)
                pool_std: dict[str, np.ndarray] = {}

                for target in TARGETS:
                    predicted, sigma, warnings_seen = predict_outer_fold(
                        dataset,
                        extended,
                        target,
                        "gaussian_process",
                        tuple(labeled),
                        queries,
                    )

                    if sigma is None:
                        raise ValueError("GPR returned no uncertainty")

                    expected = dataset.targets[target][list(test)]
                    scores = calculate_metrics(expected, predicted[:n_test])

                    pool_std[target] = sigma[n_test:]
                    curves.append(
                        {
                            "seed": seed,
                            "strategy": strategy,
                            "round": round_number,
                            "n_labeled": len(labeled),
                            "target": target,
                            "n_test": n_test,
                            **scores,
                            "n_model_warnings": len(warnings_seen),
                            "model_warnings_json": json.dumps(warnings_seen),
                        }
                    )

                if round_number == rounds:
                    continue

                if strategy == "uncertainty":
                    chosen, score_map = select_uncertain(
                        dataset,
                        tuple(labeled),
                        tuple(pool),
                        pool_std,
                        batch_size,
                    )
                else:
                    chosen = tuple(
                        int(i) for i in rng.choice(pool, size=batch_size, replace=False)
                    )
                    score_map = {}

                if (
                    len(set(chosen)) != batch_size
                    or not set(chosen).issubset(pool)
                    or set(chosen) & set(test)
                ):
                    raise ValueError("Invalid acquisition")

                for index in chosen:
                    acquisitions.append(
                        {
                            "seed": seed,
                            "strategy": strategy,
                            "round": round_number + 1,
                            "candidate_id": dataset.candidate_ids[index],
                            "candidate_family": dataset.families[index],
                            "acquisition_score": score_map.get(index, ""),
                        }
                    )

                # This is the only point where selected labels
                # become available for the NEXT model fit.
                labeled.extend(chosen)
                selected = set(chosen)
                pool = [i for i in pool if i not in selected]

    grouped = defaultdict(list)
    for row in curves:
        key = (
            row["strategy"],
            row["target"],
            row["n_labeled"],
        )
        grouped[key].append(row)

    summary = []
    for (strategy, target, count), records in sorted(grouped.items()):
        mae = [r["mae_ev"] for r in records]
        rmse = [r["rmse_ev"] for r in records]

        if len(records) != len(seeds):
            raise ValueError("Incomplete learning-curve repetitions")

        summary.append(
            {
                "strategy": strategy,
                "target": target,
                "n_labeled": count,
                "n_repetitions": len(records),
                "mean_mae_ev": float(np.mean(mae)),
                "std_mae_ev": float(np.std(mae, ddof=1)),
                "mean_rmse_ev": float(np.mean(rmse)),
                "std_rmse_ev": float(np.std(rmse, ddof=1)),
                "n_warning_runs": sum(r["n_model_warnings"] > 0 for r in records),
            }
        )

    return curves, acquisitions, manifest, summary


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"No records for {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase9_active_learning_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase9_v1"),
    )
    args = parser.parse_args()

    path = args.config.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent

    base_path = root / config["base_ml_config"]
    extended_path = root / config["extended_ml_config"]

    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    extended = yaml.safe_load(extended_path.read_text(encoding="utf-8"))

    if int(extended["random_seed"]) != int(base["validation"]["random_seed"]):
        raise ValueError("Inconsistent ML random seeds")

    dataset = load_ml_dataset(base_path)
    curves, acquisitions, manifest, summary = run_experiment(dataset, extended, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("al_learning_curves_v1.csv", curves),
        ("al_acquisitions_v1.csv", acquisitions),
        ("al_split_manifest_v1.csv", manifest),
        ("al_summary_v1.csv", summary),
    ):
        write_csv(args.output_dir / filename, rows)

    input_paths = {
        "active_learning_config": path,
        "base_ml_config": base_path,
        "extended_ml_config": extended_path,
    }
    for name in ("labeled", "prospective"):
        source = Path(base["datasets"][name])
        input_paths[name] = source if source.is_absolute() else root / source

    metadata = {
        "phase": 9,
        "version": 1,
        "experiment": config["experiment"],
        "scikit_learn_version": sklearn.__version__,
        "input_sha256": {
            name: hashlib.sha256(source.read_bytes()).hexdigest()
            for name, source in input_paths.items()
        },
        "interpretation": (
            "Retrospective active-learning simulation using existing "
            "DFT labels. Test labels are evaluation-only. Pool labels "
            "are revealed only following selection. Comparisons are "
            "paired within seeds; repeated test sets are not fully "
            "independent. Predictive uncertainties are uncalibrated."
        ),
    }

    (args.output_dir / "al_metadata_v1.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("RETROSPECTIVE ACTIVE LEARNING COMPLETE")
    print(f"Learning-curve rows: {len(curves)}")
    print(f"Acquisition rows: {len(acquisitions)}")
    print(f"Split manifest rows: {len(manifest)}")
    print(f"Summary rows: {len(summary)}")
    print("\nFINAL ACQUISITION BUDGET — MEAN TEST MAE")

    final_count = (
        config["experiment"]["initial_labeled_size"]
        + config["experiment"]["acquisition_batch_size"]
        * config["experiment"]["acquisition_rounds"]
    )

    for row in summary:
        if row["n_labeled"] == final_count:
            print(
                f"{row['target']:20} "
                f"{row['strategy']:12} "
                f"MAE={row['mean_mae_ev']:.4f} eV "
                f"SD={row['std_mae_ev']:.4f} eV"
            )


if __name__ == "__main__":
    main()
