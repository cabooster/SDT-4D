"""Select an epsilon value from uncertainty maps without ground truth."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import tifffile


CONFIDENCE_INTERVALS = (
    ("[0, 0.15)", 0.0, 0.15, True, False),
    ("[0.15, 0.6)", 0.15, 0.6, True, False),
    ("[0.6, 0.9]", 0.6, 0.9, True, True),
    ("(0.9, 0.95]", 0.9, 0.95, False, True),
    ("(0.95, 1.0]", 0.95, 1.0, False, True),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aleatoric",
        required=True,
        help="Path to an aleatoric TIFF or a directory containing *_aleatoric.tif files",
    )
    parser.add_argument(
        "--epistemic",
        default=None,
        help="Optional epistemic standard-deviation TIFF",
    )
    parser.add_argument(
        "--use_total",
        action="store_true",
        help="Use sqrt(aleatoric^2 + epistemic^2)",
    )
    parser.add_argument(
        "--epsilons",
        default="2,5,15,25,30,40",
        help="Comma-separated candidate epsilon values",
    )
    parser.add_argument("--bins", type=int, default=30, help="Confidence histogram bin count")
    parser.add_argument(
        "--target_mean_confidence",
        type=float,
        default=0.78,
        help="Preferred mean confidence used by the selection score",
    )
    parser.add_argument(
        "--out_dir",
        default="./epsilon_selection",
        help="Directory for metrics and figures",
    )
    return parser.parse_args()


def resolve_map(path_text, suffix):
    path = Path(path_text).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Uncertainty path does not exist: {path}")
    candidates = list(path.glob(f"*{suffix}.tif")) + list(path.glob(f"*{suffix}.tiff"))
    if not candidates:
        raise FileNotFoundError(f"No *{suffix}.tif files found in: {path}")
    selected = max(candidates, key=lambda item: item.stat().st_mtime)
    print(f"Found {len(candidates)} {suffix} maps. Using latest: {selected}")
    return selected.resolve()


def parse_epsilon_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            value = float(item)
            if value <= 0:
                raise ValueError("Every epsilon must be greater than zero")
            values.append(value)
    values = sorted(set(values))
    if not values:
        raise ValueError("The epsilon list is empty")
    return values


def load_sigma(aleatoric_path, epistemic_path=None, use_total=False):
    aleatoric = tifffile.imread(aleatoric_path).astype(np.float32)
    sigma = np.maximum(aleatoric, 0.0)
    if use_total:
        if epistemic_path is None:
            raise ValueError("--use_total requires --epistemic")
        epistemic = tifffile.imread(epistemic_path).astype(np.float32)
        if epistemic.shape != sigma.shape:
            raise ValueError(
                f"Uncertainty shape mismatch: {sigma.shape} versus {epistemic.shape}"
            )
        sigma = np.sqrt(sigma**2 + np.maximum(epistemic, 0.0) ** 2)
    sigma = sigma.reshape(-1)
    sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    if sigma.size == 0:
        raise ValueError("No finite positive uncertainty values were found")
    return sigma


def interval_mass(
    centers, probabilities, lower, upper, include_lower, include_upper
):
    lower_mask = centers >= lower if include_lower else centers > lower
    upper_mask = centers <= upper if include_upper else centers < upper
    return float(probabilities[lower_mask & upper_mask].sum())


def evaluate_epsilon(sigmas, epsilon, bins, target_mean_confidence):
    confidence = erf(epsilon / (np.sqrt(2.0) * np.maximum(sigmas, 1e-12)))
    confidence = np.clip(confidence, 0.0, 1.0)
    counts, edges = np.histogram(confidence, bins=bins, range=(0.0, 1.0))
    probabilities = counts.astype(np.float64) / max(int(counts.sum()), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean_confidence = float(np.sum(probabilities * centers))
    informative_mass = float(
        probabilities[(centers >= 0.6) & (centers <= 0.9)].sum()
    )
    low_mass = float(probabilities[centers < 0.15].sum())
    saturated_mass = float(probabilities[centers > 0.95].sum())
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    normalized_entropy = entropy / np.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    center_penalty = abs(mean_confidence - target_mean_confidence)
    score = (
        2.0 * informative_mass
        + 0.6 * normalized_entropy
        - 1.2 * low_mass
        - 1.0 * saturated_mass
        - 0.8 * center_penalty
    )
    masses = {
        name: interval_mass(
            centers,
            probabilities,
            lower,
            upper,
            include_lower,
            include_upper,
        )
        for name, lower, upper, include_lower, include_upper in CONFIDENCE_INTERVALS
    }
    metrics = {
        "epsilon": epsilon,
        "mean_confidence": mean_confidence,
        "informative_mass": informative_mass,
        "low_mass": low_mass,
        "saturated_mass": saturated_mass,
        "normalized_entropy": normalized_entropy,
        "center_penalty": center_penalty,
        "score": score,
    }
    return centers, probabilities, metrics, masses


def save_csv(path, rows, columns):
    matrix = np.asarray([[row[column] for column in columns] for row in rows], dtype=np.float64)
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt="%.10g",
    )


def plot_score(metrics, best_epsilon, output_path):
    epsilons = [row["epsilon"] for row in metrics]
    scores = [row["score"] for row in metrics]
    best_index = epsilons.index(best_epsilon)
    plt.figure(figsize=(6.2, 4.0))
    plt.plot(epsilons, scores, marker="o", color="#845ec2", label="Selection score")
    plt.scatter(
        [best_epsilon],
        [scores[best_index]],
        marker="*",
        s=180,
        color="#d62728",
        edgecolors="black",
        zorder=5,
        label=f"Selected epsilon={best_epsilon:g}",
    )
    plt.axvline(best_epsilon, linestyle="--", color="#d62728", alpha=0.85)
    plt.xlabel("Epsilon")
    plt.ylabel("Selection score")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_interval_masses(mass_rows, best_epsilon, output_path):
    epsilons = [row["epsilon"] for row in mass_rows]
    positions = np.arange(len(epsilons))
    bottoms = np.zeros(len(epsilons), dtype=np.float64)
    colors = ("#fbeaff", "#b39cd0", "#845ec2", "#00c9a7", "#c4fcef")
    plt.figure(figsize=(7.0, 4.4))
    for (name, _, _, _, _), color in zip(CONFIDENCE_INTERVALS, colors):
        heights = np.asarray([row[name] for row in mass_rows])
        plt.bar(
            positions,
            heights,
            bottom=bottoms,
            width=0.72,
            label=name,
            color=color,
            edgecolor="white",
        )
        bottoms += heights
    selected = epsilons.index(best_epsilon)
    plt.axvline(selected, linestyle="--", color="#d62728", alpha=0.85)
    plt.scatter(
        [selected],
        [1.02],
        marker="*",
        s=180,
        color="#d62728",
        edgecolors="black",
        clip_on=False,
        zorder=5,
    )
    plt.xticks(positions, [f"{value:g}" for value in epsilons])
    plt.ylim(0.0, 1.08)
    plt.xlabel("Epsilon")
    plt.ylabel("Fraction of pixels")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(title="Confidence interval", frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least two")
    aleatoric_path = resolve_map(args.aleatoric, "_aleatoric")
    epistemic_path = (
        resolve_map(args.epistemic, "_epistemic") if args.epistemic else None
    )
    epsilons = parse_epsilon_list(args.epsilons)
    sigmas = load_sigma(aleatoric_path, epistemic_path, args.use_total)
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution_rows = []
    metrics = []
    mass_rows = []
    for epsilon in epsilons:
        centers, probabilities, metric, masses = evaluate_epsilon(
            sigmas, epsilon, args.bins, args.target_mean_confidence
        )
        distribution_rows.extend(
            {
                "epsilon": epsilon,
                "confidence_bin_center": center,
                "fraction_of_pixels": probability,
            }
            for center, probability in zip(centers, probabilities)
        )
        metrics.append(metric)
        mass_rows.append({"epsilon": epsilon, **masses})

    ranking = sorted(
        metrics,
        key=lambda row: (
            -row["score"],
            -row["informative_mass"],
            row["saturated_mass"],
            row["low_mass"],
            row["epsilon"],
        ),
    )
    best = ranking[0]
    best_epsilon = best["epsilon"]
    save_csv(
        output_dir / "epsilon_confidence_distribution.csv",
        distribution_rows,
        ("epsilon", "confidence_bin_center", "fraction_of_pixels"),
    )
    save_csv(
        output_dir / "epsilon_selection_metrics.csv",
        metrics,
        (
            "epsilon",
            "mean_confidence",
            "informative_mass",
            "low_mass",
            "saturated_mass",
            "normalized_entropy",
            "center_penalty",
            "score",
        ),
    )
    plot_score(metrics, best_epsilon, output_dir / "epsilon_selection_score.png")
    plot_interval_masses(
        mass_rows, best_epsilon, output_dir / "confidence_mass_by_interval.png"
    )
    summary = {
        "selected_epsilon": best_epsilon,
        "selection_metrics": best,
        "aleatoric_path": str(aleatoric_path),
        "epistemic_path": str(epistemic_path) if epistemic_path else None,
        "used_total_uncertainty": args.use_total,
        "pixel_count": int(sigmas.size),
    }
    with (output_dir / "selected_epsilon.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(
        "Selected epsilon by unsupervised confidence-distribution score: "
        f"{best_epsilon:g} (score={best['score']:.6f})"
    )
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
