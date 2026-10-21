"""Nested-CV Ridge and fixed-config Extra Trees DFT surrogates."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cei_scout.ml.baselines import calculate_metrics
from cei_scout.ml.dataset import MLDataset, load_ml_dataset
from cei_scout.ml.validation import make_cv_folds


def make_inner_splits(
    families: tuple[str, ...],
    config: dict[str, Any],
    validation: str,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Make inner folds from outer-training molecules only."""
    settings = config["validation"]
    indices = np.arange(len(families))

    if validation == "primary":
        spec = settings["inner_model_selection"]
        splitter = KFold(
            n_splits=int(spec["n_splits"]),
            shuffle=bool(spec["shuffle"]),
            random_state=int(settings["random_seed"]),
        )
        raw_splits = splitter.split(indices)

    elif validation == "family":
        splitter = LeaveOneGroupOut()
        raw_splits = splitter.split(
            indices,
            groups=np.asarray(families),
        )

    else:
        raise ValueError(f"Unknown validation scheme: {validation}")

    folds = tuple(
        (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
        for train, test in raw_splits
    )

    if not folds:
        raise ValueError("No inner validation folds")

    for train, test in folds:
        if set(train) & set(test):
            raise ValueError("Inner training/validation overlap")

        if validation == "family":
            train_families = {families[i] for i in train}
            test_families = {families[i] for i in test}

            if train_families & test_families:
                raise ValueError("Inner chemical-family leakage")

    return folds


def predict_outer_fold(
    dataset: MLDataset,
    config: dict[str, Any],
    target: str,
    model_name: str,
    validation: str,
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit using outer-training data and predict outer-test data."""
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)

    if (
        not len(train)
        or not len(test)
        or set(train) & set(test)
        or np.any(train < 0)
        or np.any(test < 0)
        or np.any(train >= len(dataset.candidate_ids))
        or np.any(test >= len(dataset.candidate_ids))
    ):
        raise ValueError("Invalid outer train/test indices")

    X_train = dataset.X[train]
    X_test = dataset.X[test]
    y_train = dataset.targets[target][train]

    details: dict[str, Any] = {
        "selected_alpha": "",
        "inner_cv_mae_ev": "",
        "n_inner_folds": 0,
    }

    if model_name == "ridge":
        alphas = [float(value) for value in config["models"]["ridge"]["alpha_grid"]]

        if not alphas or not all(np.isfinite(alpha) and alpha > 0 for alpha in alphas):
            raise ValueError("Ridge alpha values must be positive and finite")

        training_families = tuple(dataset.families[i] for i in train)
        inner_folds = make_inner_splits(
            training_families,
            config,
            validation,
        )

        search = GridSearchCV(
            estimator=make_pipeline(StandardScaler(), Ridge()),
            param_grid={"ridge__alpha": alphas},
            scoring="neg_mean_absolute_error",
            cv=inner_folds,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )

        search.fit(X_train, y_train)
        prediction = search.predict(X_test)

        details = {
            "selected_alpha": float(search.best_params_["ridge__alpha"]),
            "inner_cv_mae_ev": float(-search.best_score_),
            "n_inner_folds": len(inner_folds),
        }

    elif model_name == "extra_trees":
        settings = config["models"]["extra_trees"]

        model = ExtraTreesRegressor(
            n_estimators=int(settings["n_estimators"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            max_features=float(settings["max_features"]),
            random_state=int(config["validation"]["random_seed"]),
            n_jobs=1,
        )

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

    else:
        raise ValueError(f"Unknown ML model: {model_name}")

    prediction = np.asarray(prediction, dtype=float)

    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise ValueError("Invalid model predictions")

    return prediction, details


def evaluate_surrogates(
    dataset: MLDataset,
    config: dict[str, Any],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Evaluate held-out predictions for both targets and models."""
    predictions: list[dict] = []
    fold_metrics: list[dict] = []
    summary: list[dict] = []

    for validation in ("primary", "family"):
        outer_folds = make_cv_folds(dataset, config, validation)

        for target in config["targets"]:
            y = dataset.targets[target]

            for model_name in config["models"]["candidates"]:
                all_true: list[float] = []
                all_pred: list[float] = []

                for fold_number, fold in enumerate(outer_folds, start=1):
                    test = np.asarray(fold.test_indices, dtype=int)

                    estimate, details = predict_outer_fold(
                        dataset,
                        config,
                        target,
                        model_name,
                        validation,
                        fold.train_indices,
                        fold.test_indices,
                    )

                    truth = y[test]
                    metrics = calculate_metrics(truth, estimate)

                    repeat = (
                        (fold_number - 1) // config["validation"]["primary"]["n_splits"]
                        + 1
                        if validation == "primary"
                        else ""
                    )

                    fold_metrics.append(
                        {
                            "validation": validation,
                            "target": target,
                            "model": model_name,
                            "fold": fold_number,
                            "repeat": repeat,
                            "held_out_family": fold.held_out_family or "",
                            "n_train": len(fold.train_indices),
                            "n_test": len(test),
                            **details,
                            **metrics,
                        }
                    )

                    for index, observed, predicted in zip(
                        test, truth, estimate, strict=True
                    ):
                        error = float(predicted - observed)

                        predictions.append(
                            {
                                "validation": validation,
                                "target": target,
                                "model": model_name,
                                "fold": fold_number,
                                "repeat": repeat,
                                "held_out_family": (fold.held_out_family or ""),
                                "candidate_id": dataset.candidate_ids[index],
                                "candidate_family": dataset.families[index],
                                "dft_value_ev": float(observed),
                                "predicted_value_ev": float(predicted),
                                "signed_error_ev": error,
                                "absolute_error_ev": abs(error),
                            }
                        )

                    all_true.extend(float(value) for value in truth)
                    all_pred.extend(float(value) for value in estimate)

                summary.append(
                    {
                        "validation": validation,
                        "target": target,
                        "model": model_name,
                        "n_folds": len(outer_folds),
                        "n_predictions": len(all_pred),
                        "n_unique_candidates": len(dataset.candidate_ids),
                        **calculate_metrics(
                            np.asarray(all_true),
                            np.asarray(all_pred),
                        ),
                    }
                )

    return predictions, fold_metrics, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No evaluation results for {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_surrogate_results(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    predictions: list[dict],
    folds: list[dict],
    summary: list[dict],
) -> None:
    """Write reproducible model results and input provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "surrogate_predictions_v1.csv", predictions)
    _write_csv(output_dir / "surrogate_fold_metrics_v1.csv", folds)
    _write_csv(output_dir / "surrogate_summary_v1.csv", summary)

    root = config_path.resolve().parent.parent
    hashes = {"configuration": _sha256(config_path)}

    for name in ("labeled", "prospective"):
        source = Path(config["datasets"][name])
        if not source.is_absolute():
            source = root / source
        hashes[name] = _sha256(source)

    metadata = {
        "phase": 9,
        "output_version": 1,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "input_sha256": hashes,
        "targets": config["targets"],
        "features": config["features"]["primary"],
        "model_settings": {
            name: config["models"][name] for name in config["models"]["candidates"]
        },
        "validation": config["validation"],
        "ridge_selection_metric": "inner_cv_mae_ev",
        "extra_trees_tuned": False,
        "prediction_policy": "outer_fold_held_out_only",
        "scientific_scope": (
            "Vertical DFT electronic-energy descriptors; "
            "not experimental electrochemical potentials."
        ),
    }

    (output_dir / "surrogate_metadata_v1.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 9 DFT surrogate models."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase9_ml_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase9_v1"),
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = load_ml_dataset(config_path)

    predictions, folds, summary = evaluate_surrogates(dataset, config)

    write_surrogate_results(
        args.output_dir,
        config_path,
        config,
        predictions,
        folds,
        summary,
    )

    print("PHASE 9 SURROGATE EVALUATION COMPLETE")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Fold metrics rows: {len(folds)}")
    print(f"Summary rows: {len(summary)}")

    for row in summary:
        print(
            f"{row['validation']:8s} "
            f"{row['target']:20s} "
            f"{row['model']:12s} "
            f"MAE={row['mae_ev']:.4f} eV "
            f"RMSE={row['rmse_ev']:.4f} eV"
        )


if __name__ == "__main__":
    main()
