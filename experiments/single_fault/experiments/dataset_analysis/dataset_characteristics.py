from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable

from experiments.single_fault.utils.experiment_paths import DATASET_ANALYSIS_OUTPUT_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "single_fault" / "json"
OUTPUT_DIR = DATASET_ANALYSIS_OUTPUT_DIR

DATASET_DIRS = {
    "Algorithm-Generated": DATA_DIR / "who_and_when__algorithm-generated",
    "Hand-Crafted": DATA_DIR / "who_and_when__hand-crafted",
}

COLORS = {
    "Algorithm-Generated": "#4C78A8",
    "Hand-Crafted": "#F58518",
}

GPT_4O_MINI_MAX_TOKENS = 128_000


@dataclass
class DatasetMetrics:
    name: str
    num_samples: int
    mistake_steps: list[int]
    trajectory_lengths: list[int]
    behavior_content_tokens: list[int]
    trajectory_total_tokens: list[int]


def numeric_path_order(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), path.name)
    except ValueError:
        return (10**9, path.name)


def approximate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def build_token_counter() -> tuple[Callable[[str], int], str]:
    try:
        import tiktoken  # type: ignore
    except Exception as exc:
        note = f"approximate: len(text)/4 fallback because tiktoken import failed ({type(exc).__name__})"
        return approximate_token_count, note

    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        return lambda text: len(encoding.encode(text or "")), "exact:gpt-4o-mini"
    except Exception as exc:
        note = (
            "approximate: len(text)/4 fallback because gpt-4o-mini encoding "
            f"is unavailable in this environment ({type(exc).__name__})"
        )
        return approximate_token_count, note


def load_json_rows(dataset_dir: Path) -> list[dict]:
    json_paths = sorted(dataset_dir.glob("*.json"), key=numeric_path_order)
    return [json.loads(path.read_text(encoding="utf-8")) for path in json_paths]


def collect_metrics(
    dataset_name: str, dataset_dir: Path, count_tokens: Callable[[str], int]
) -> DatasetMetrics:
    rows = load_json_rows(dataset_dir)

    mistake_steps: list[int] = []
    trajectory_lengths: list[int] = []
    behavior_content_tokens: list[int] = []
    trajectory_total_tokens: list[int] = []

    for row in rows:
        trajectory = row.get("trajectory", [])
        mistake_steps.append(int(row["mistake_step"]))
        trajectory_lengths.append(len(trajectory))

        token_counts = [count_tokens(item.get("content", "")) for item in trajectory]
        behavior_content_tokens.extend(token_counts)
        trajectory_total_tokens.append(sum(token_counts))

    return DatasetMetrics(
        name=dataset_name,
        num_samples=len(rows),
        mistake_steps=mistake_steps,
        trajectory_lengths=trajectory_lengths,
        behavior_content_tokens=behavior_content_tokens,
        trajectory_total_tokens=trajectory_total_tokens,
    )


def summarize(values: list[int]) -> dict[str, float | int | list[int]]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "most_common": [],
            "least_common": [],
        }

    counter = Counter(values)
    max_count = max(counter.values())
    min_count = min(counter.values())
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "most_common": sorted(
            [value for value, count in counter.items() if count == max_count]
        ),
        "least_common": sorted(
            [value for value, count in counter.items() if count == min_count]
        ),
    }


def discrete_bins(
    datasets: list[DatasetMetrics], extractor: Callable[[DatasetMetrics], list[int]]
) -> list[int]:
    unique_values = sorted(
        {value for dataset in datasets for value in extractor(dataset)}
    )
    if not unique_values:
        return [0]
    return unique_values


def build_hist_bins(
    datasets: list[DatasetMetrics],
    extractor: Callable[[DatasetMetrics], list[int]],
    num_bins: int,
) -> list[float]:
    all_values = [value for dataset in datasets for value in extractor(dataset)]
    if not all_values:
        return [0.0, 1.0]

    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        return [min_value - 0.5, max_value + 0.5]

    step = (max_value - min_value) / num_bins
    return [min_value + index * step for index in range(num_bins + 1)]


def format_summary_lines(
    stats: dict[str, float | int | list[int]], top_k: int = 5
) -> list[str]:
    most_common = list(stats["most_common"])[:top_k]
    least_common = list(stats["least_common"])[:top_k]
    return [
        f"n={stats['count']} | min={stats['min']} | max={stats['max']}",
        f"mean={stats['mean']:.2f} | median={stats['median']:.2f}",
        f"most common={most_common}",
        f"least common={least_common}",
    ]


def annotate_axis(
    ax, dataset: DatasetMetrics, values: list[int], color: str, top_k: int = 5
) -> None:
    stats = summarize(values)
    lines = format_summary_lines(stats, top_k=top_k)
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": color,
            "alpha": 0.9,
        },
    )
    ax.set_title(
        f"{dataset.name} (n={dataset.num_samples})", fontsize=13, fontweight="bold"
    )


def configure_discrete_x_ticks(ax, values: list[int], max_ticks: int = 8) -> None:
    unique_values = sorted(set(values))
    if not unique_values:
        return

    if len(unique_values) <= max_ticks:
        tick_values = unique_values
    else:
        step = math.ceil(len(unique_values) / max_ticks)
        tick_values = unique_values[::step]
        if tick_values[-1] != unique_values[-1]:
            tick_values.append(unique_values[-1])

    ax.set_xticks(tick_values)
    ax.tick_params(axis="x", labelrotation=30)


def configure_hist_x_ticks(ax, values: list[int], max_ticks: int = 7) -> None:
    if not values:
        return

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        tick_values = [min_value]
    else:
        tick_values = [
            round(min_value + (max_value - min_value) * index / (max_ticks - 1))
            for index in range(max_ticks)
        ]
        tick_values = sorted(set(tick_values))

    ax.set_xticks(tick_values)
    ax.tick_params(axis="x", labelrotation=30)


def configure_axis_number_format(ax) -> None:
    from matplotlib.ticker import FuncFormatter  # type: ignore

    def formatter(value, _pos):
        abs_value = abs(value)
        if abs_value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if abs_value >= 1_000:
            return f"{value / 1_000:.1f}k"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.1f}"

    ax.xaxis.set_major_formatter(FuncFormatter(formatter))


def plot_discrete_bar_distribution(
    output_path_base: Path,
    datasets: list[DatasetMetrics],
    extractor: Callable[[DatasetMetrics], list[int]],
    figure_title: str,
    x_label: str,
    color_map: dict[str, str],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to run this script. "
            "Install it in the active Python environment before executing."
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)

    for ax, dataset in zip(axes, datasets):
        values = extractor(dataset)
        color = color_map[dataset.name]
        counts = Counter(values)
        x_values = sorted(counts.keys())
        y_values = [counts[value] for value in x_values]
        ax.bar(
            x_values, y_values, color=color, edgecolor="black", alpha=0.85, width=0.8
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        configure_discrete_x_ticks(ax, values)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        annotate_axis(ax, dataset, values, color)

    fig.suptitle(figure_title, fontsize=16, fontweight="bold")
    fig.savefig(output_path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_histogram_with_boxplot(
    output_path_base: Path,
    datasets: list[DatasetMetrics],
    extractor: Callable[[DatasetMetrics], list[int]],
    figure_title: str,
    x_label: str,
    color_map: dict[str, str],
    num_bins: int,
    reference_line_x: int | None = None,
    reference_line_label: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to run this script. "
            "Install it in the active Python environment before executing."
        ) from exc

    from matplotlib.ticker import MaxNLocator  # type: ignore

    bins = build_hist_bins(datasets, extractor, num_bins=num_bins)
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, height_ratios=[5, 1.6], hspace=0.12, wspace=0.08)

    for index, dataset in enumerate(datasets):
        values = extractor(dataset)
        color = color_map[dataset.name]
        hist_ax = fig.add_subplot(outer[0, index])

        hist_ax.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.85)
        hist_ax.set_xlabel(x_label)
        hist_ax.set_ylabel("Count")
        hist_ax.grid(axis="y", linestyle="--", alpha=0.35)
        hist_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        configure_hist_x_ticks(hist_ax, values)
        configure_axis_number_format(hist_ax)
        annotate_axis(hist_ax, dataset, values, color)
        if reference_line_x is not None:
            hist_ax.axvline(
                reference_line_x, color="red", linestyle="--", linewidth=2, alpha=0.85
            )
            if reference_line_label:
                ymax = hist_ax.get_ylim()[1]
                hist_ax.text(
                    reference_line_x,
                    ymax * 0.96,
                    reference_line_label,
                    color="red",
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=9,
                    backgroundcolor="white",
                )

    compare_ax = fig.add_subplot(outer[1, :])
    compare_values = [extractor(dataset) for dataset in datasets]
    compare_colors = [color_map[dataset.name] for dataset in datasets]
    box = compare_ax.boxplot(
        compare_values,
        vert=False,
        patch_artist=True,
        labels=[dataset.name for dataset in datasets],
        widths=0.55,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#666666",
            "markeredgecolor": "#333333",
            "alpha": 0.25,
        },
    )
    for patch, color in zip(box["boxes"], compare_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    compare_ax.set_xlabel(x_label)
    compare_ax.set_title(
        "Side-by-Side Boxplot Comparison", fontsize=13, fontweight="bold"
    )
    compare_ax.grid(axis="x", linestyle="--", alpha=0.35)
    configure_axis_number_format(compare_ax)
    if reference_line_x is not None:
        compare_ax.axvline(
            reference_line_x, color="red", linestyle="--", linewidth=2, alpha=0.85
        )
        if reference_line_label:
            ymax = compare_ax.get_ylim()[1]
            compare_ax.text(
                reference_line_x,
                ymax - 0.02,
                reference_line_label,
                color="red",
                rotation=90,
                va="top",
                ha="right",
                fontsize=9,
                backgroundcolor="white",
            )

    fig.suptitle(figure_title, fontsize=16, fontweight="bold")
    fig.savefig(output_path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_report(
    output_path: Path, datasets: list[DatasetMetrics], tokenizer_note: str
) -> None:
    lines = [
        "# Single-Fault Dataset Characteristics",
        "",
        f"Tokenizer mode for token-based plots: `{tokenizer_note}`",
        "",
    ]

    for dataset in datasets:
        lines.append(f"## {dataset.name}")
        for label, values in [
            ("mistake_step", dataset.mistake_steps),
            ("trajectory_length", dataset.trajectory_lengths),
            ("behavior_content_tokens", dataset.behavior_content_tokens),
            ("trajectory_total_tokens", dataset.trajectory_total_tokens),
        ]:
            stats = summarize(values)
            lines.append(
                f"- `{label}`: count={stats['count']}, min={stats['min']}, max={stats['max']}, "
                f"mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
                f"most_common={list(stats['most_common'])[:10]}, least_common={list(stats['least_common'])[:10]}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count_tokens, tokenizer_note = build_token_counter()
    datasets = [
        collect_metrics(dataset_name, dataset_dir, count_tokens)
        for dataset_name, dataset_dir in DATASET_DIRS.items()
    ]

    plot_discrete_bar_distribution(
        output_path_base=OUTPUT_DIR / "mistake_step_distribution",
        datasets=datasets,
        extractor=lambda dataset: dataset.mistake_steps,
        figure_title="Mistake Step Distribution",
        x_label="Mistake step",
        color_map=COLORS,
    )
    plot_histogram_with_boxplot(
        output_path_base=OUTPUT_DIR / "trajectory_length_distribution",
        datasets=datasets,
        extractor=lambda dataset: dataset.trajectory_lengths,
        figure_title="Trajectory Length Distribution",
        x_label="Number of agent behaviors in one trajectory",
        color_map=COLORS,
        num_bins=24,
    )
    plot_histogram_with_boxplot(
        output_path_base=OUTPUT_DIR / "behavior_content_token_distribution",
        datasets=datasets,
        extractor=lambda dataset: dataset.behavior_content_tokens,
        figure_title="Agent-Behavior Content Token Distribution",
        x_label="Tokens per agent-behavior content",
        color_map=COLORS,
        num_bins=30,
    )
    plot_histogram_with_boxplot(
        output_path_base=OUTPUT_DIR / "trajectory_total_token_distribution",
        datasets=datasets,
        extractor=lambda dataset: dataset.trajectory_total_tokens,
        figure_title="Trajectory Total Token Distribution",
        x_label="Tokens per full trajectory",
        color_map=COLORS,
        num_bins=30,
        reference_line_x=GPT_4O_MINI_MAX_TOKENS,
        reference_line_label="GPT-4o mini max context (128k)",
    )

    write_summary_report(
        output_path=OUTPUT_DIR / "summary.md",
        datasets=datasets,
        tokenizer_note=tokenizer_note,
    )

    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Tokenizer mode: {tokenizer_note}")


if __name__ == "__main__":
    main()
