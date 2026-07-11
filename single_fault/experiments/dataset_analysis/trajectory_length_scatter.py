from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from single_fault.utils.experiment_paths import DATASET_ANALYSIS_OUTPUT_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "single_fault" / "json"
OUTPUT_DIR = DATASET_ANALYSIS_OUTPUT_DIR
OUTPUT_PATH = OUTPUT_DIR / "trajectory_length_scatter.png"

DATASET_DIRS = {
    "Algorithm-Generated": DATA_DIR / "who_and_when__algorithm-generated",
    "Hand-Crafted": DATA_DIR / "who_and_when__hand-crafted",
}

COLORS = {
    "Algorithm-Generated": "#4C78A8",
    "Hand-Crafted": "#F58518",
}

X_OFFSETS = {
    "Algorithm-Generated": -0.18,
    "Hand-Crafted": 0.18,
}


def numeric_path_order(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), path.name)
    except ValueError:
        return (10**9, path.name)


def load_trajectory_lengths(dataset_dir: Path) -> list[int]:
    json_paths = sorted(dataset_dir.glob("*.json"), key=numeric_path_order)
    lengths: list[int] = []
    for path in json_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        lengths.append(len(row.get("trajectory", [])))
    return lengths


def build_stacked_scatter_points(lengths: list[int], x_offset: float) -> tuple[list[float], list[int]]:
    counts_by_length = Counter(lengths)
    x_values: list[float] = []
    y_values: list[int] = []

    for length in sorted(counts_by_length):
        count = counts_by_length[length]
        for stack_index in range(1, count + 1):
            x_values.append(length + x_offset)
            y_values.append(stack_index)

    return x_values, y_values


def main() -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to run this script. "
            "Install it in the active Python environment before executing."
        ) from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_lengths = {
        dataset_name: load_trajectory_lengths(dataset_dir)
        for dataset_name, dataset_dir in DATASET_DIRS.items()
    }

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    max_stack_height = 0
    all_unique_lengths: set[int] = set()
    for dataset_name, lengths in dataset_lengths.items():
        x_values, y_values = build_stacked_scatter_points(lengths, X_OFFSETS[dataset_name])
        counts = Counter(lengths)
        max_stack_height = max(max_stack_height, max(counts.values(), default=0))
        all_unique_lengths.update(counts.keys())

        ax.scatter(
            x_values,
            y_values,
            s=42,
            alpha=0.8,
            color=COLORS[dataset_name],
            edgecolors="black",
            linewidths=0.4,
            label=f"{dataset_name} (n={len(lengths)})",
        )

    sorted_unique_lengths = sorted(all_unique_lengths)
    if len(sorted_unique_lengths) <= 12:
        tick_values = sorted_unique_lengths
    else:
        step = max(1, len(sorted_unique_lengths) // 10)
        tick_values = sorted_unique_lengths[::step]
        if tick_values[-1] != sorted_unique_lengths[-1]:
            tick_values.append(sorted_unique_lengths[-1])

    ax.set_xticks(tick_values)
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_ylim(0, max_stack_height + 1)
    ax.set_xlabel("Trajectory length")
    ax.set_ylabel("Stacked count at the same trajectory length")
    ax.set_title("Trajectory Length Scatter by Dataset", fontsize=16, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    ax.text(
        0.01,
        0.99,
        "Points with the same trajectory length are stacked vertically.\n"
        "Datasets are shifted slightly on the x-axis for readability.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
    )

    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
