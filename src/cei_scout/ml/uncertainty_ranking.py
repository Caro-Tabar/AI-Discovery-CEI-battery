"""Out-of-fold GPR ranking stability and interval diagnostics."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

TARGETS = {
    "ionization_energy": "vertical_ionization_energy_ev",
    "electron_affinity": "vertical_electron_affinity_ev",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite prediction or uncertainty")
    return result


def composite_ranks(
    ie: dict[str, float],
    ea: dict[str, float],
) -> dict[str, int]:
    """Match the deterministic Phase 9 dual-redox ranking policy."""
    if set(ie) != set(ea):
        raise ValueError("IE/EA candidate mismatch")

    ids = set(ie)
    ie_order = sorted(ids, key=lambda cid: (ie[cid], cid))
    ea_order = sorted(ids, key=lambda cid: (-ea[cid], cid))

    ie_rank = {cid: i for i, cid in enumerate(ie_order, 1)}
    ea_rank = {cid: i for i, cid in enumerate(ea_order, 1)}

    ordered = sorted(
        ids,
        key=lambda cid: (
            ie_rank[cid] + ea_rank[cid],
            ie_rank[cid],
            cid,
        ),
    )
    return {cid: i for i, cid in enumerate(ordered, 1)}


def analyze(
    ranking: list[dict[str, str]],
    predictions: list[dict[str, str]],
    repeats: int = 5,
    top_k: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Compare cross-validated surrogate ranks with observed DFT ranks."""
    complete = {
        row["candidate_id"]: row for row in ranking if row["dft_status"] == "complete"
    }

    if (
        len(ranking) != 60
        or len(complete) != 56
        or len({r["candidate_id"] for r in ranking}) != 60
    ):
        raise ValueError("Unexpected frozen-pool population")

    gp_rows = [row for row in predictions if row["model"] == "gaussian_process"]

    lookup = {}
    for row in gp_rows:
        cid = row["candidate_id"]
        target = row["target"]
        validation = row["validation"]

        if cid not in complete or target not in TARGETS:
            raise ValueError("Unknown candidate or GPR target")

        if row["candidate_family"] != complete[cid]["candidate_family"]:
            raise ValueError("Candidate family mismatch")

        observed = finite(row["dft_value_ev"])
        expected = finite(complete[cid][TARGETS[target]])

        if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Prediction/DFT reference mismatch")

        std = finite(row["predictive_std_ev"])
        if std < 0:
            raise ValueError("Negative predictive standard deviation")

        key = (
            validation,
            target,
            row["fold"],
            cid,
        )
        if key in lookup:
            raise ValueError("Duplicate held-out prediction")

        lookup[key] = row

    ranks = defaultdict(list)
    ie_std = defaultdict(list)
    ea_std = defaultdict(list)

    for repeat in range(1, repeats + 1):
        ie: dict[str, float] = {}
        ea: dict[str, float] = {}

        for row in gp_rows:
            if row["validation"] != "primary":
                continue
            if int(row["repeat"]) != repeat:
                continue

            cid = row["candidate_id"]
            target = row["target"]
            destination = ie if target == "ionization_energy" else ea

            if cid in destination:
                raise ValueError("Duplicate prediction within repeat")

            destination[cid] = finite(row["predicted_value_ev"])

            if target == "ionization_energy":
                ie_std[cid].append(finite(row["predictive_std_ev"]))
            else:
                ea_std[cid].append(finite(row["predictive_std_ev"]))

        if set(ie) != set(complete) or set(ea) != set(complete):
            raise ValueError("Incomplete cross-validation repetition")

        predicted_ranks = composite_ranks(ie, ea)

        for cid, rank in predicted_ranks.items():
            ranks[cid].append(rank)

    output = []

    for row in ranking:
        cid = row["candidate_id"]
        record: dict[str, str | int | float] = {
            "candidate_id": cid,
            "preferred_name": row["preferred_name"],
            "candidate_family": row["candidate_family"],
            "dft_status": row["dft_status"],
            "dft_dual_redox_rank": row["dual_redox_priority_rank"],
            "gpr_median_rank": "",
            "gpr_min_rank": "",
            "gpr_max_rank": "",
            "gpr_rank_range": "",
            "gpr_top10_frequency": "",
            "gpr_mean_ie_std_ev": "",
            "gpr_mean_ea_std_ev": "",
            "manual_review_flag": row["manual_review_flag"],
        }

        if cid in complete:
            values = ranks[cid]
            record.update(
                {
                    "gpr_median_rank": median(values),
                    "gpr_min_rank": min(values),
                    "gpr_max_rank": max(values),
                    "gpr_rank_range": max(values) - min(values),
                    "gpr_top10_frequency": sum(value <= top_k for value in values)
                    / repeats,
                    "gpr_mean_ie_std_ev": mean(ie_std[cid]),
                    "gpr_mean_ea_std_ev": mean(ea_std[cid]),
                }
            )

        output.append(record)

    output.sort(
        key=lambda row: (
            row["dft_status"] != "complete",
            int(row["dft_dual_redox_rank"]) if row["dft_dual_redox_rank"] else 999,
            row["candidate_id"],
        )
    )

    diagnostics = []

    for validation in ("primary", "family"):
        for target in TARGETS:
            subset = [
                r
                for r in gp_rows
                if r["validation"] == validation and r["target"] == target
            ]

            expected_count = (
                len(complete) * repeats if validation == "primary" else len(complete)
            )
            if len(subset) != expected_count:
                raise ValueError("Unexpected held-out GPR row count")

            groups = ["ALL", *sorted({r["candidate_family"] for r in subset})]
            for group in groups:
                selected = [
                    r
                    for r in subset
                    if group == "ALL" or r["candidate_family"] == group
                ]

                coverage = [
                    abs(finite(r["dft_value_ev"]) - finite(r["predicted_value_ev"]))
                    <= 1.96 * finite(r["predictive_std_ev"])
                    for r in selected
                ]

                diagnostics.append(
                    {
                        "validation": validation,
                        "target": target,
                        "family": group,
                        "n_predictions": len(selected),
                        "empirical_95pct_coverage": mean(coverage),
                        "mean_predictive_std_ev": mean(
                            finite(r["predictive_std_ev"]) for r in selected
                        ),
                        "mae_ev": mean(
                            abs(
                                finite(r["dft_value_ev"])
                                - finite(r["predicted_value_ev"])
                            )
                            for r in selected
                        ),
                    }
                )

    return output, diagnostics


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No output rows")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/phase9_v1"),
    )
    args = parser.parse_args()

    folder = args.input_dir
    sources = {
        "ranking": folder / "species_ranking_v1.csv",
        "gpr_predictions": folder / "extended_predictions_v1.csv",
    }

    ranking = read_csv(sources["ranking"])
    predictions = read_csv(sources["gpr_predictions"])
    stability, diagnostics = analyze(ranking, predictions)

    write_csv(folder / "gpr_ranking_stability_v1.csv", stability)
    write_csv(
        folder / "gpr_uncertainty_diagnostics_v1.csv",
        diagnostics,
    )

    metadata = {
        "version": 1,
        "method": "repeated_CV_out_of_fold_GPR",
        "repetitions": 5,
        "top_k": 10,
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sources.items()
        },
        "interpretation": (
            "Top-10 frequency is a cross-validation stability "
            "diagnostic, not a calibrated probability. Predictive "
            "standard deviations describe surrogate uncertainty, "
            "not DFT calculation uncertainty. The four DFT-incomplete "
            "candidates have no observed DFT ranking."
        ),
    }

    (folder / "gpr_ranking_stability_metadata_v1.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("GPR RANKING STABILITY COMPLETE")
    print(f"Candidates: {len(stability)}")
    print("\nTop 10 by observed DFT ranking:")
    print("Candidate   DFT rank  GP median  GP range  Top-10 freq.")

    for row in stability[:10]:
        print(
            f"{row['candidate_id']:<11} "
            f"{int(row['dft_dual_redox_rank']):>5} "
            f"{row['gpr_median_rank']:>10.1f} "
            f"{row['gpr_rank_range']:>9} "
            f"{row['gpr_top10_frequency']:>12.0%}"
        )

    print("\nOverall GPR predictive interval diagnostics:")
    for row in diagnostics:
        if row["family"] == "ALL":
            print(
                f"{row['validation']:8} "
                f"{row['target']:20} "
                f"Coverage={row['empirical_95pct_coverage']:.1%} "
                f"MAE={row['mae_ev']:.3f} eV"
            )


if __name__ == "__main__":
    main()
