from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from single_fault.utils.datasets import DATASET_DIRS, OUTPUT_DIR
else:
    from .utils.datasets import DATASET_DIRS, OUTPUT_DIR


LENGTH_PLOT_DIR = OUTPUT_DIR / "length_analysis"
BASE_COLUMNS = ["file", "gt_agent", "gt_step"]
METHODS = [
    ("all_at_once", "All at once", "#6c757d"),
    ("step_by_step", "Step by step", "#2b8a3e"),
    ("step_based_multi_step_w5", "Step-based w=5", "#e76f51"),
]
COMMON_BUCKET_BINS = [0, 10, 20, 40, 60, 10_000]
COMMON_BUCKET_LABELS = ["<10", "10-19", "20-39", "40-59", "60+"]


def load_dataset_costs(dataset_key: str) -> pd.DataFrame:
    baseline_path = OUTPUT_DIR / f"{dataset_key}_cost.csv"
    step_based_path = OUTPUT_DIR / f"{dataset_key}_step_based_multi_step_cost.csv"

    baseline_df = pd.read_csv(baseline_path)
    step_based_df = pd.read_csv(step_based_path)
    return baseline_df.merge(step_based_df, on=BASE_COLUMNS, how="inner")


def load_lengths(dataset_dir: Path, files: pd.Series) -> list[int]:
    lengths: list[int] = []
    for file_name in files:
        row = json.loads((dataset_dir / file_name).read_text(encoding="utf-8"))
        lengths.append(len(row.get("trajectory", [])))
    return lengths


def add_total_token_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for method_key, _, _ in METHODS:
        result[f"{method_key}_total_tokens"] = (
            pd.to_numeric(result[f"{method_key}_input_tokens"], errors="coerce").fillna(0)
            + pd.to_numeric(result[f"{method_key}_output_tokens"], errors="coerce").fillna(0)
        )
    return result


def build_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    bucket_df = add_total_token_columns(df)
    bucket_df["bucket"] = pd.cut(
        bucket_df["length"],
        bins=COMMON_BUCKET_BINS,
        labels=COMMON_BUCKET_LABELS,
        right=False,
    )

    metric_columns: list[str] = []
    for method_key, _, _ in METHODS:
        metric_columns.extend([f"{method_key}_latency", f"{method_key}_total_tokens"])

    grouped = bucket_df.groupby("bucket", observed=False)
    summary = grouped[metric_columns].mean().reset_index()
    summary["n"] = grouped.size().values
    return summary[summary["n"] > 0].reset_index(drop=True)


def annotate_points(ax: plt.Axes, x_positions: list[int], y_values: list[float], color: str, integer: bool = False) -> None:
    for x, y in zip(x_positions, y_values):
        if pd.isna(y):
            continue
        label = f"{y:.0f}" if integer else f"{y:.1f}"
        offset = max(y_values) * 0.02 if max(y_values) > 0 else 0.02
        ax.text(x, y + offset, label, color=color, fontsize=8, ha="center", va="bottom")


def plot_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric_suffix: str,
    title: str,
    y_label: str,
    integer: bool = False,
) -> None:
    x_positions = list(range(len(summary)))

    for method_key, label, color in METHODS:
        y_values = summary[f"{method_key}_{metric_suffix}"].tolist()
        ax.plot(
            x_positions,
            y_values,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=color,
            label=label,
        )
        annotate_points(ax, x_positions, y_values, color, integer=integer)

    bucket_labels = [f"{bucket}\n(n={n})" for bucket, n in zip(summary["bucket"].astype(str), summary["n"])]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Trajectory length bucket")
    ax.set_title(f"{title}\nHigher is worse", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)


def main() -> None:
    LENGTH_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_frames: list[pd.DataFrame] = []
    for dataset_key, dataset_dir in DATASET_DIRS.items():
        df = load_dataset_costs(dataset_key)
        df["length"] = load_lengths(dataset_dir, df["file"])
        df["dataset"] = dataset_key
        dataset_frames.append(df)

    merged_df = pd.concat(dataset_frames, ignore_index=True)
    summary = build_bucket_summary(merged_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    plot_metric(
        ax=axes[0],
        summary=summary,
        metric_suffix="latency",
        title="All datasets - Latency",
        y_label="Latency (seconds)",
        integer=False,
    )
    plot_metric(
        ax=axes[1],
        summary=summary,
        metric_suffix="total_tokens",
        title="All datasets - Token Cost Proxy",
        y_label="Total tokens (input + output)",
        integer=True,
    )

    fig.savefig(LENGTH_PLOT_DIR / "all_datasets_cost_line_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cost comparison plot to: {LENGTH_PLOT_DIR}")


if __name__ == "__main__":
    main()
