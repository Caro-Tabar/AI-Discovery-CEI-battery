"""Generate GitHub-ready Phase 9 scientific figures."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

RESULTS = Path("results/phase9_v1")
OUTPUT = Path("docs/assets/phase9")
OUTPUT.mkdir(parents=True, exist_ok=True)

NAVY = "#24334B"
TEAL = "#237D79"
CORAL = "#C76C55"
GREY = "#A2AAB5"
MODEL_NAMES = {
    "mean": "Mean",
    "linear_regression": "Linear",
    "ridge": "Ridge",
    "extra_trees": "Extra Trees",
    "gaussian_process": "Gaussian Process",
    "mlp": "Neural Network",
}
FAMILY_NAMES = {
    "sulfite": "Sulfite",
    "sulfate_or_sultone": "Sulfate / sultone",
    "sulfone_or_sulfoxide": "Sulfone / sulfoxide",
    "thioether_or_disulfide": "Thioether / disulfide",
    "sulfonate_or_sulfonamide": "Sulfonate / sulfonamide",
    "mixed_s_o_n": "Mixed S/O/N",
}
MODELS = list(MODEL_NAMES)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def read_csv(filename):
    with (RESULTS / filename).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty input: {filename}")
    return rows


def save(fig, filename):
    fig.savefig(
        OUTPUT / f"{filename}.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / f"{filename}.svg",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Created: {filename}.png / .svg")


def model_comparison(target, title):
    rows = []
    for filename in (
        "baseline_summary_v1.csv",
        "surrogate_summary_v1.csv",
        "extended_summary_v1.csv",
    ):
        rows.extend(read_csv(filename))

    lookup = {
        (r["validation"], r["target"], r["model"]): float(r["mae_ev"]) for r in rows
    }
    if len(lookup) != len(rows):
        raise ValueError("Duplicate model-summary records")

    primary = np.array([lookup[("primary", target, model)] for model in MODELS])
    family = np.array([lookup[("family", target, model)] for model in MODELS])

    y = np.arange(len(MODELS))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(y - 0.2, primary, height=0.38, color=TEAL, label="Repeated 5-fold CV")
    ax.barh(y + 0.2, family, height=0.38, color=CORAL, label="Leave-one-family-out")

    ax.set_yticks(y, [MODEL_NAMES[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error (eV)  ·  Lower is better")
    ax.set_title(title, loc="left", pad=16)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    for i, (p, f) in enumerate(zip(primary, family, strict=True)):
        ax.text(p + 0.007, i - 0.2, f"{p:.3f}", va="center", fontsize=8)
        ax.text(f + 0.007, i + 0.2, f"{f:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, max(primary.max(), family.max()) * 1.20)
    save(fig, f"model_comparison_{target}")


def gp_parity(target, title):
    rows = read_csv("extended_predictions_v1.csv")
    grouped = defaultdict(list)
    observed = {}

    for row in rows:
        if (
            row["validation"] != "primary"
            or row["model"] != "gaussian_process"
            or row["target"] != target
        ):
            continue

        cid = row["candidate_id"]
        truth = float(row["dft_value_ev"])
        if cid in observed and not np.isclose(observed[cid], truth):
            raise ValueError(f"Inconsistent DFT reference: {cid}")

        observed[cid] = truth
        grouped[cid].append(float(row["predicted_value_ev"]))

    if len(grouped) != 56 or any(len(values) != 5 for values in grouped.values()):
        raise ValueError("Expected five held-out predictions per molecule")

    ids = sorted(grouped)
    x = np.array([observed[cid] for cid in ids])
    y = np.array([np.mean(grouped[cid]) for cid in ids])

    lower = min(x.min(), y.min())
    upper = max(x.max(), y.max())
    margin = (upper - lower) * 0.07

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(x, y, s=38, alpha=0.83, color=TEAL, edgecolor="white", linewidth=0.5)
    ax.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        "--",
        color=GREY,
        linewidth=1.5,
    )

    ax.set_xlim(lower - margin, upper + margin)
    ax.set_ylim(lower - margin, upper + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("DFT reference (eV)")
    ax.set_ylabel("Mean held-out GPR prediction (eV)")
    ax.set_title(title, loc="left", pad=16)
    ax.grid(alpha=0.12)

    fig.text(
        0.5,
        0.01,
        "56 molecules · each prediction averaged over 5 held-out repeats",
        ha="center",
        fontsize=8,
        color="#667085",
    )
    save(fig, f"gpr_parity_{target}")


def active_learning_curve(target, title):
    rows = read_csv("al_summary_v1.csv")
    fig, ax = plt.subplots(figsize=(7.8, 5))

    for strategy, color, label in (
        ("random", GREY, "Random sampling"),
        ("uncertainty", TEAL, "GPR uncertainty sampling"),
    ):
        selected = sorted(
            (
                row
                for row in rows
                if row["target"] == target and row["strategy"] == strategy
            ),
            key=lambda r: int(r["n_labeled"]),
        )

        if len(selected) != 11:
            raise ValueError("Expected 11 acquisition-budget points")

        x = np.array([int(r["n_labeled"]) for r in selected])
        y = np.array([float(r["mean_mae_ev"]) for r in selected])
        sd = np.array([float(r["std_mae_ev"]) for r in selected])

        ax.plot(x, y, "-o", markersize=4, linewidth=2, color=color, label=label)
        ax.fill_between(x, np.maximum(0, y - sd), y + sd, color=color, alpha=0.15)

    ax.set_xlabel("Number of DFT-labeled molecules")
    ax.set_ylabel("Held-out test MAE (eV)")
    ax.set_title(title, loc="left", pad=16)
    ax.legend(frameon=False)
    ax.grid(alpha=0.16)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        "Mean ± 1 SD across 5 paired seeds · retrospective simulation",
        ha="center",
        fontsize=8,
        color="#667085",
    )
    save(fig, f"active_learning_{target}")


def family_heatmap(target, title):
    rows = []
    for filename in (
        "baseline_predictions_v1.csv",
        "surrogate_predictions_v1.csv",
        "extended_predictions_v1.csv",
    ):
        rows.extend(read_csv(filename))

    families = sorted(
        {r["candidate_family"] for r in rows if r["validation"] == "family"}
    )

    groups = defaultdict(list)
    for row in rows:
        if row["validation"] == "family" and row["target"] == target:
            groups[(row["candidate_family"], row["model"])].append(
                float(row["absolute_error_ev"])
            )

    matrix = np.array(
        [[np.mean(groups[(family, model)]) for model in MODELS] for family in families]
    )

    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise ValueError("Incomplete chemical-family error matrix")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(
        range(len(MODELS)), [MODEL_NAMES[m] for m in MODELS], rotation=25, ha="right"
    )

    ax.set_yticks(
        range(len(families)),
        [FAMILY_NAMES[family] for family in families],
    )

    ax.set_title(title, loc="left", pad=16)

    for i in range(6):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[i, j] > matrix.max() * 0.60 else NAVY,
            )

    fig.colorbar(im, ax=ax, label="Held-out MAE (eV)", shrink=0.85)
    save(fig, f"family_error_{target}")


def redox_priority():
    rows = [
        r for r in read_csv("species_ranking_v1.csv") if r["dft_status"] == "complete"
    ]

    fig, ax = plt.subplots(figsize=(8.5, 6.3))
    other = [r for r in rows if int(r["dual_redox_priority_rank"]) > 10]
    top = [r for r in rows if int(r["dual_redox_priority_rank"]) <= 10]

    def coordinates(selection, field):
        return [float(row[field]) for row in selection]

    ax.scatter(
        coordinates(other, "vertical_ionization_energy_ev"),
        coordinates(other, "vertical_electron_affinity_ev"),
        color=GREY,
        alpha=0.65,
        s=38,
        label="Other DFT-complete",
    )
    ax.scatter(
        coordinates(top, "vertical_ionization_energy_ev"),
        coordinates(top, "vertical_electron_affinity_ev"),
        color=CORAL,
        s=70,
        label="Top 10 dual-redox rank",
    )

    for i, row in enumerate(
        sorted(top, key=lambda r: int(r["dual_redox_priority_rank"]))
    ):
        x = float(row["vertical_ionization_energy_ev"])
        y = float(row["vertical_electron_affinity_ev"])
        ax.annotate(
            str(row["dual_redox_priority_rank"]),
            (x, y),
            xytext=(7, 8 if i % 2 == 0 else -11),
            textcoords="offset points",
            fontsize=9,
            color=NAVY,
        )

    ax.set_xlabel("Vertical ionization energy (eV)")
    ax.set_ylabel("Vertical electron affinity (eV)")
    ax.set_title("DFT redox descriptor landscape", loc="left", pad=16)
    ax.legend(frameon=False)
    ax.grid(alpha=0.13)
    fig.text(
        0.5,
        0.01,
        "Numbers indicate exploratory dual-redox rank; not CEI performance",
        ha="center",
        fontsize=8,
        color="#667085",
    )
    save(fig, "redox_priority_map")


def ranking_stability():
    rows = sorted(
        (
            r
            for r in read_csv("gpr_ranking_stability_v1.csv")
            if r["dft_status"] == "complete"
        ),
        key=lambda r: int(r["dft_dual_redox_rank"]),
    )[:10]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(rows))
    labels = []

    for i, row in enumerate(rows):
        low = float(row["gpr_min_rank"])
        high = float(row["gpr_max_rank"])
        mid = float(row["gpr_median_rank"])
        truth = int(row["dft_dual_redox_rank"])

        ax.plot([low, high], [i, i], color=TEAL, linewidth=3, alpha=0.70)
        ax.scatter(mid, i, s=75, color=TEAL, zorder=3)
        ax.scatter(truth, i, marker="D", s=42, color=CORAL, zorder=4)

        frequency = float(row["gpr_top10_frequency"])
        labels.append(f"{row['candidate_id']}  ·  {frequency:.0%} top-10")

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Redox-priority rank  ·  Lower is higher priority")
    ax.set_title(
        "Top-ranked species: DFT vs GPR ranking stability",
        loc="left",
        pad=16,
    )
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    ax.scatter([], [], marker="D", color=CORAL, label="DFT rank")
    ax.scatter([], [], color=TEAL, label="GPR median rank")
    ax.plot([], [], color=TEAL, linewidth=3, label="GPR min-max rank")
    ax.legend(frameon=False, loc="lower right")

    save(fig, "ranking_stability_top10")


def main():
    model_comparison("ionization_energy", "Ionization energy · Model comparison")
    model_comparison("electron_affinity", "Electron affinity · Model comparison")
    gp_parity("ionization_energy", "GPR · Ionization energy")
    gp_parity("electron_affinity", "GPR · Electron affinity")
    active_learning_curve("ionization_energy", "Active learning · Ionization energy")
    active_learning_curve("electron_affinity", "Active learning · Electron affinity")
    family_heatmap("ionization_energy", "Family generalization · IE")
    family_heatmap("electron_affinity", "Family generalization · EA")
    redox_priority()
    ranking_stability()
    print("PHASE 9 FIGURES COMPLETE")


if __name__ == "__main__":
    main()
