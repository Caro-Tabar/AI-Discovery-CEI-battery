"""Cross-validated physical baselines for Phase 9 DFT surrogates."""

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
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from cei_scout.ml.dataset import MLDataset, load_ml_dataset
from cei_scout.ml.validation import make_cv_folds


def predict_fold(
    dataset: MLDataset,
    config: dict[str, Any],
    target: str,
    model_name: str,
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
) -> np.ndarray:
    """Fit on training molecules and predict held-out molecules."""
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)

    if not len(train) or not len(test) or not set(train).isdisjoint(set(test)):
        raise ValueError("Invalid or overlapping train/test indices")

    if model_name == "mean":
        estimator = DummyRegressor(strategy="mean")
        X = dataset.X[:, :1]

    elif model_name == "linear_regression":
        selected = config["features"]["simple_baselines"][target]

        if len(selected) != 1:
            raise ValueError("Linear baseline requires one predictor")

        feature_index = dataset.feature_names.index(selected[0])
        X = dataset.X[:, [feature_index]]
        estimator = LinearRegression()

    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

    y = dataset.targets[target]

    # Held-out target values are never supplied to the estimator.
    estimator.fit(X[train], y[train])
    predictions = np.asarray(estimator.predict(X[test]), dtype=float)

    if predictions.shape != (len(test),):
        raise ValueError("Unexpected prediction shape")

    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite model predictions")

    return predictions


def calculate_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate prediction errors in electronvolts."""
    return {
        "mae_ev": float(mean_absolute_error(observed, predicted)),
        "rmse_ev": float(root_mean_squared_error(observed, predicted)),
    }


def evaluate_baselines(
    dataset: MLDataset,
    config: dict[str, Any],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Evaluate both targets with both cross-validation schemes."""
    predictions_rows: list[dict] = []
    fold_rows: list[dict] = []
    summary_rows: list[dict] = []

    for validation in ("primary", "family"):
        folds = make_cv_folds(dataset, config, validation)

        for target in config["targets"]:
            observed = dataset.targets[target]

            for model_name in config["models"]["baselines"]:
                all_true: list[float] = []
                all_pred: list[float] = []

                for fold_number, fold in enumerate(folds, start=1):
                    test = np.asarray(fold.test_indices, dtype=int)
                    predicted = predict_fold(
                        dataset,
                        config,
                        target,
                        model_name,
                        fold.train_indices,
                        fold.test_indices,
                    )

                    true_values = observed[test]
                    metrics = calculate_metrics(true_values, predicted)

                    repeat = (
                        (fold_number - 1) // config["validation"]["primary"]["n_splits"]
                        + 1
                        if validation == "primary"
                        else ""
                    )

                    fold_rows.append(
                        {
                            "validation": validation,
                            "target": target,
                            "model": model_name,
                            "fold": fold_number,
                            "repeat": repeat,
                            "held_out_family": fold.held_out_family or "",
                            "n_train": len(fold.train_indices),
                            "n_test": len(fold.test_indices),
                            **metrics,
                        }
                    )

                    for index, truth, estimate in zip(
                        test, true_values, predicted, strict=True
                    ):
                        error = float(estimate - truth)

                        predictions_rows.append(
                            {
                                "validation": validation,
                                "target": target,
                                "model": model_name,
                                "fold": fold_number,
                                "repeat": repeat,
                                "held_out_family": (fold.held_out_family or ""),
                                "candidate_id": dataset.candidate_ids[index],
                                "candidate_family": dataset.families[index],
                                "dft_value_ev": float(truth),
                                "predicted_value_ev": float(estimate),
                                "signed_error_ev": error,
                                "absolute_error_ev": abs(error),
                            }
                        )

                    all_true.extend(float(x) for x in true_values)
                    all_pred.extend(float(x) for x in predicted)

                # Pooled held-out errors; repeated CV records are correlated.
                summary_rows.append(
                    {
                        "validation": validation,
                        "target": target,
                        "model": model_name,
                        "n_folds": len(folds),
                        "n_predictions": len(all_pred),
                        "n_unique_candidates": len(dataset.candidate_ids),
                        **calculate_metrics(
                            np.asarray(all_true),
                            np.asarray(all_pred),
                        ),
                    }
                )

    return predictions_rows, fold_rows, summary_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_baseline_results(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    predictions: list[dict],
    folds: list[dict],
    summary: list[dict],
) -> None:
    """Write versioned evaluation results with input provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        output_dir / "baseline_predictions_v1.csv",
        predictions,
    )
    _write_csv(
        output_dir / "baseline_fold_metrics_v1.csv",
        folds,
    )
    _write_csv(
        output_dir / "baseline_summary_v1.csv",
        summary,
    )

    root = config_path.resolve().parent.parent
    checksums = {"configuration": _checksum(config_path)}

    for name in ("labeled", "prospective"):
        path = Path(config["datasets"][name])

        if not path.is_absolute():
            path = root / path

        checksums[name] = _checksum(path)

    metadata = {
        "phase": 9,
        "output_version": 1,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "input_sha256": checksums,
        "targets": config["targets"],
        "baseline_models": config["models"]["baselines"],
        "simple_baseline_features": config["features"]["simple_baselines"],
        "validation": config["validation"],
        "metrics": ["mae_ev", "rmse_ev"],
        "prediction_policy": "held_out_only",
        "interpretation": (
            "Repeated-CV predictions are not independent observations. "
            "Reported metrics describe prediction of vertical DFT "
            "electronic-energy descriptors, not experimental potentials."
        ),
    }

    path = output_dir / "baseline_metadata_v1.json"
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 9 held-out baseline evaluations."
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

    predictions, folds, summary = evaluate_baselines(dataset, config)

    write_baseline_results(
        args.output_dir,
        config_path,
        config,
        predictions,
        folds,
        summary,
    )

    print("PHASE 9 BASELINE EVALUATION COMPLETE")
    print(f"Candidate count: {len(dataset.candidate_ids)}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Fold metrics rows: {len(folds)}")
    print(f"Summary rows: {len(summary)}")

    for row in summary:
        print(
            f"{row['validation']:8s} "
            f"{row['target']:20s} "
            f"{row['model']:18s} "
            f"MAE={row['mae_ev']:.4f} eV "
            f"RMSE={row['rmse_ev']:.4f} eV"
        )


if __name__ == "__main__":
    main()
