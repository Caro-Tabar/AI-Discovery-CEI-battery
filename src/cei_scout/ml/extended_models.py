"""Fold-safe Gaussian-process and neural-network CEI-Scout surrogates."""

import argparse
import csv
import hashlib
import json
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import yaml
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cei_scout.ml.baselines import calculate_metrics
from cei_scout.ml.dataset import MLDataset, load_ml_dataset
from cei_scout.ml.validation import make_cv_folds


def predict_outer_fold(
    dataset: MLDataset,
    settings: dict[str, Any],
    target: str,
    model_name: str,
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...]]:
    """Fit solely on outer-training rows, then predict outer-test rows."""
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    n = len(dataset.candidate_ids)

    if (
        not len(train)
        or not len(test)
        or len(set(train)) != len(train)
        or len(set(test)) != len(test)
        or not set(train).isdisjoint(test)
        or np.any(train < 0)
        or np.any(test < 0)
        or np.any(train >= n)
        or np.any(test >= n)
    ):
        raise ValueError("Invalid or overlapping outer train/test indices")

    X_train, X_test = dataset.X[train], dataset.X[test]
    y_train = dataset.targets[target][train]
    seed = int(settings["random_seed"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        if model_name == "gaussian_process":
            params = settings["models"]["gaussian_process"]

            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(X_train)
            scaled_test = scaler.transform(X_test)

            kernel = ConstantKernel(
                float(params["signal_variance"]),
                tuple(params["signal_bounds"]),
            ) * RBF(
                length_scale=float(params["length_scale"]),
                length_scale_bounds=tuple(params["length_bounds"]),
            ) + WhiteKernel(
                noise_level=float(params["noise_level"]),
                noise_level_bounds=tuple(params["noise_bounds"]),
            )

            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=float(params["numerical_alpha"]),
                normalize_y=True,
                n_restarts_optimizer=int(params["n_restarts_optimizer"]),
                random_state=seed,
            )

            model.fit(scaled_train, y_train)
            predicted, predictive_std = model.predict(scaled_test, return_std=True)

        elif model_name == "mlp":
            params = settings["models"]["mlp"]

            model = make_pipeline(
                StandardScaler(),
                TransformedTargetRegressor(
                    regressor=MLPRegressor(
                        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
                        activation=str(params["activation"]),
                        solver=str(params["solver"]),
                        alpha=float(params["alpha"]),
                        max_iter=int(params["max_iter"]),
                        max_fun=int(params["max_fun"]),
                        tol=float(params["tol"]),
                        random_state=seed,
                    ),
                    transformer=StandardScaler(),
                ),
            )

            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            predictive_std = None

        else:
            raise ValueError(f"Unknown model: {model_name}")

    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    if predicted.shape != (len(test),) or not np.isfinite(predicted).all():
        raise ValueError("Invalid model prediction values")

    if predictive_std is not None:
        predictive_std = np.asarray(predictive_std, dtype=float).reshape(-1)

        if (
            predictive_std.shape != predicted.shape
            or not np.isfinite(predictive_std).all()
            or np.any(predictive_std < 0)
        ):
            raise ValueError("Invalid Gaussian-process predictive uncertainty")

    diagnostics = tuple(f"{item.category.__name__}: {item.message}" for item in caught)
    return predicted, predictive_std, diagnostics


def evaluate_models(
    dataset: MLDataset,
    base_config: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Evaluate both models on the existing Phase 9 validation folds."""
    predictions: list[dict] = []
    fold_metrics: list[dict] = []
    summary: list[dict] = []

    for validation in ("primary", "family"):
        folds = make_cv_folds(dataset, base_config, validation)

        for target in base_config["targets"]:
            y = dataset.targets[target]

            for model_name in ("gaussian_process", "mlp"):
                truths: list[float] = []
                estimates: list[float] = []
                covered: list[int] = []
                std_values: list[float] = []
                warning_folds = 0

                for fold_number, fold in enumerate(folds, start=1):
                    test = np.asarray(fold.test_indices, dtype=int)

                    estimate, uncertainty, diagnostics = predict_outer_fold(
                        dataset,
                        settings,
                        target,
                        model_name,
                        fold.train_indices,
                        fold.test_indices,
                    )

                    truth = y[test]
                    warning_folds += bool(diagnostics)
                    scores = calculate_metrics(truth, estimate)

                    repeat = (
                        (fold_number - 1)
                        // base_config["validation"]["primary"]["n_splits"]
                        + 1
                        if validation == "primary"
                        else ""
                    )

                    fold_coverage: float | str = ""
                    fold_std: float | str = ""
                    if uncertainty is not None:
                        in_interval = np.abs(truth - estimate) <= 1.96 * uncertainty
                        covered.extend(int(v) for v in in_interval)
                        std_values.extend(float(v) for v in uncertainty)
                        fold_coverage = float(np.mean(in_interval))
                        fold_std = float(np.mean(uncertainty))

                    fold_metrics.append(
                        {
                            "validation": validation,
                            "target": target,
                            "model": model_name,
                            "fold": fold_number,
                            "repeat": repeat,
                            "held_out_family": (fold.held_out_family or ""),
                            "n_train": len(fold.train_indices),
                            "n_test": len(test),
                            **scores,
                            "gp_95pct_coverage": fold_coverage,
                            "gp_mean_predictive_std_ev": fold_std,
                            "warnings_json": json.dumps(diagnostics),
                        }
                    )

                    for offset, (pos, actual, estimate_value) in enumerate(
                        zip(test, truth, estimate, strict=True)
                    ):
                        error = float(estimate_value - actual)
                        sigma_value: float | None = (
                            float(uncertainty[offset])
                            if uncertainty is not None
                            else None
                        )

                        predictions.append(
                            {
                                "validation": validation,
                                "target": target,
                                "model": model_name,
                                "fold": fold_number,
                                "repeat": repeat,
                                "held_out_family": (fold.held_out_family or ""),
                                "candidate_id": dataset.candidate_ids[int(pos)],
                                "candidate_family": dataset.families[int(pos)],
                                "dft_value_ev": float(actual),
                                "predicted_value_ev": float(estimate_value),
                                "signed_error_ev": error,
                                "absolute_error_ev": abs(error),
                                "predictive_std_ev": (
                                    sigma_value if sigma_value is not None else ""
                                ),
                                "within_gp_95pct_interval": (
                                    int(abs(error) <= 1.96 * sigma_value)
                                    if sigma_value is not None
                                    else ""
                                ),
                            }
                        )

                    truths.extend(float(value) for value in truth)
                    estimates.extend(float(value) for value in estimate)

                summary.append(
                    {
                        "validation": validation,
                        "target": target,
                        "model": model_name,
                        "n_folds": len(folds),
                        "n_predictions": len(estimates),
                        "n_unique_candidates": len(dataset.candidate_ids),
                        **calculate_metrics(
                            np.asarray(truths),
                            np.asarray(estimates),
                        ),
                        "gp_95pct_coverage": (
                            float(np.mean(covered)) if covered else ""
                        ),
                        "gp_mean_predictive_std_ev": (
                            float(np.mean(std_values)) if std_values else ""
                        ),
                        "n_warning_folds": warning_folds,
                    }
                )

    return predictions, fold_metrics, summary


def write_results(
    output_dir: Path,
    extended_config_path: Path,
    base_config_path: Path,
    settings: dict[str, Any],
    base_config: dict[str, Any],
    predictions: list[dict],
    fold_metrics: list[dict],
    summary: list[dict],
) -> None:
    """Save versioned outputs and input SHA-256 checksums."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, rows in (
        ("extended_predictions_v1.csv", predictions),
        ("extended_fold_metrics_v1.csv", fold_metrics),
        ("extended_summary_v1.csv", summary),
    ):
        if not rows:
            raise ValueError(f"No rows to save for {filename}")

        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    root = base_config_path.resolve().parent.parent
    sources = {
        "extended_config": extended_config_path,
        "base_config": base_config_path,
    }

    for name in ("labeled", "prospective"):
        path = Path(base_config["datasets"][name])
        sources[name] = path if path.is_absolute() else root / path

    metadata = {
        "phase": 9,
        "version": 1,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "models": settings["models"],
        "features": base_config["features"]["primary"],
        "validation": base_config["validation"],
        "seed": settings["random_seed"],
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sources.items()
        },
        "note": (
            "All predictions are outer-fold held out. "
            "GPR intervals are model-based and not guaranteed "
            "calibrated. DFT targets are vertical electronic energies."
        ),
    }

    (output_dir / "extended_metadata_v1.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 GPR and MLP comparisons")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase9_extended_models_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase9_v1"),
    )
    args = parser.parse_args()

    extended_path = args.config.resolve()
    settings = yaml.safe_load(extended_path.read_text(encoding="utf-8"))
    base_path = (extended_path.parent.parent / settings["base_ml_config"]).resolve()
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    if int(settings["random_seed"]) != int(base_config["validation"]["random_seed"]):
        raise ValueError("Extended and base ML seeds do not match")

    dataset = load_ml_dataset(base_path)
    predictions, folds, summary = evaluate_models(dataset, base_config, settings)

    write_results(
        args.output_dir,
        extended_path,
        base_path,
        settings,
        base_config,
        predictions,
        folds,
        summary,
    )

    print("PHASE 9 EXTENDED MODELS COMPLETE")
    print(
        f"Predictions: {len(predictions)} | "
        f"Fold results: {len(folds)} | "
        f"Summaries: {len(summary)}"
    )

    for row in summary:
        print(
            f"{row['validation']:8} "
            f"{row['target']:20} "
            f"{row['model']:18} "
            f"MAE={row['mae_ev']:.4f} eV "
            f"RMSE={row['rmse_ev']:.4f} eV "
            f"Warnings={row['n_warning_folds']}"
        )


if __name__ == "__main__":
    main()
