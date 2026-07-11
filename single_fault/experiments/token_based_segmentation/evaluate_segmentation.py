from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from single_fault.utils.datasets import DATASET_DIRS
from single_fault.utils.experiment_paths import (
    BASELINE_OUTPUT_DIR,
    STEP_BASED_SEGMENTATION_OUTPUT_DIR,
    TOKEN_BASED_SEGMENTATION_OUTPUT_DIR,
    baseline_accuracy_path,
    step_based_accuracy_path,
    token_based_accuracy_path,
    TOKEN_BASED_EVALUATION_DIR,
)


EVAL_OUTPUT_DIR = TOKEN_BASED_EVALUATION_DIR
BASELINE_METHODS = ["all_at_once", "step_by_step"]
BASE_COLUMNS = {"file", "gt_agent", "gt_step"}
BASELINE_STYLES = {
    "all_at_once": {"color": "#6c757d", "linestyle": "--", "label": "All at once"},
    "step_by_step": {"color": "#2b8a3e", "linestyle": ":", "label": "Step by step"},
}
FAMILY_COLORS = {
    "step_based_segmentation": "#e76f51",
    "token_based_segmentation": "#2a9d8f",
    "other": "#8d99ae",
}


def extract_methods(columns: list[str], suffix: str) -> list[str]:
    methods: list[str] = []
    for column in columns:
        if column.endswith(suffix):
            methods.append(column[: -len(suffix)])
    return methods


def load_dataset_frames(dataset_key: str) -> list[pd.DataFrame]:
    import pandas as pd

    frames: list[pd.DataFrame] = []

    baseline_path = baseline_accuracy_path(dataset_key)
    if baseline_path.exists():
        frames.append(pd.read_csv(baseline_path))

    step_based_path = step_based_accuracy_path(dataset_key)
    if step_based_path.exists():
        frames.append(pd.read_csv(step_based_path))

    token_based_path = token_based_accuracy_path(dataset_key)
    if token_based_path.exists():
        frames.append(pd.read_csv(token_based_path))

    experiment_output_dirs = [
        BASELINE_OUTPUT_DIR,
        STEP_BASED_SEGMENTATION_OUTPUT_DIR,
        TOKEN_BASED_SEGMENTATION_OUTPUT_DIR,
    ]
    known_paths = {baseline_path, step_based_path, token_based_path}
    extra_paths: list[Path] = []
    for output_dir in experiment_output_dirs:
        extra_paths.extend(
            path
            for path in sorted(output_dir.glob(f"{dataset_key}*.csv"))
            if not path.name.endswith("_cost.csv") and path not in known_paths
        )

    for path in extra_paths:
        frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError(f"No result CSVs found for dataset '{dataset_key}'.")
    return frames


def merge_result_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0].copy()
    for frame in frames[1:]:
        extra_columns = [column for column in frame.columns if column not in BASE_COLUMNS and column not in merged.columns]
        merged = merged.merge(frame[list(BASE_COLUMNS) + extra_columns], on=["file", "gt_agent", "gt_step"], how="outer")
    return merged


def summarize_dataset(dataset_key: str, df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    agent_methods = set(extract_methods(list(df.columns), "_agent_acc"))
    step_methods = set(extract_methods(list(df.columns), "_step_acc"))
    methods = sorted(agent_methods & step_methods)

    records: list[dict[str, object]] = []
    for method in methods:
        agent_col = f"{method}_agent_acc"
        step_col = f"{method}_step_acc"
        records.append(
            {
                "dataset": dataset_key,
                "method": method,
                "method_family": infer_method_family(method),
                "agent_accuracy": pd.to_numeric(df[agent_col], errors="coerce").mean(),
                "step_accuracy": pd.to_numeric(df[step_col], errors="coerce").mean(),
                "num_samples": int(df["file"].nunique()),
            }
        )

    summary = pd.DataFrame(records)
    if summary.empty:
        raise ValueError(f"No accuracy columns found for dataset '{dataset_key}'.")
    return summary.sort_values(["method_family", "method"]).reset_index(drop=True)


def infer_method_family(method: str) -> str:
    if method in BASELINE_METHODS:
        return "baseline"
    if method.startswith("step_based_multi_step"):
        return "step_based_segmentation"
    if method.startswith("token_based_multi_step"):
        return "token_based_segmentation"
    return "other"


def build_label(method: str) -> str:
    if method == "all_at_once":
        return "All at once"
    if method == "step_by_step":
        return "Step by step"

    step_match = re.fullmatch(r"step_based_multi_step_prev_w(\d+)", method)
    if step_match:
        window = step_match.group(1)
        return f"Step-based\nprev, w={window}"

    step_match = re.fullmatch(r"step_based_multi_step_next_w(\d+)", method)
    if step_match:
        window = step_match.group(1)
        return f"Step-based\nnext, w={window}"

    step_match = re.fullmatch(r"step_based_multi_step_w(\d+)_o(\d+)", method)
    if step_match:
        window, overlap = step_match.groups()
        return f"Step-based\nw={window}, o={overlap}"

    step_match = re.fullmatch(r"step_based_multi_step_w(\d+)", method)
    if step_match:
        window = step_match.group(1)
        return f"Step-based\nw={window}"

    token_match = re.fullmatch(r"token_based_multi_step_(\d+)pct_ov(\d+)", method)
    if token_match:
        ratio, overlap = token_match.groups()
        return f"Token-based\n{ratio}%, ov={overlap}%"

    token_match = re.fullmatch(r"token_based_multi_step_(\d+)pct", method)
    if token_match:
        ratio = token_match.group(1)
        return f"Token-based\n{ratio}%"

    return method.replace("_", "\n")


def sort_methods(methods: list[str]) -> list[str]:
    def segmentation_key(method: str) -> tuple[int, int, str]:
        step_match = re.fullmatch(r"step_based_multi_step_prev_w(\d+)", method)
        if step_match:
            return (0, int(step_match.group(1)), method)
        step_match = re.fullmatch(r"step_based_multi_step_w(\d+)_o(\d+)", method)
        if step_match:
            return (1, int(step_match.group(1)), method)
        step_match = re.fullmatch(r"step_based_multi_step_w(\d+)", method)
        if step_match:
            return (2, int(step_match.group(1)), method)
        step_match = re.fullmatch(r"step_based_multi_step_next_w(\d+)", method)
        if step_match:
            return (3, int(step_match.group(1)), method)
        token_match = re.fullmatch(r"token_based_multi_step_(\d+)pct_ov(\d+)", method)
        if token_match:
            return (4, int(token_match.group(1)), method)
        token_match = re.fullmatch(r"token_based_multi_step_(\d+)pct", method)
        if token_match:
            return (4, int(token_match.group(1)), method)
        return (5, 0, method)

    return sorted([method for method in methods if method not in BASELINE_METHODS], key=segmentation_key)


def style_bar(value: float, low_baseline: float, high_baseline: float) -> tuple[float, float]:
    if value > high_baseline:
        return high_baseline, value - high_baseline
    if value < low_baseline:
        return value, low_baseline - value

    nearest_baseline = high_baseline if abs(value - high_baseline) <= abs(value - low_baseline) else low_baseline
    bottom = min(value, nearest_baseline)
    height = abs(value - nearest_baseline)
    return bottom, height


def annotate_bar(ax: Any, x: float, value: float) -> None:
    ax.text(
        x,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


def plot_metric(summary: pd.DataFrame, dataset_key: str, metric: str, output_path: Path) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ordered_methods = sort_methods(summary["method"].tolist())
    plot_df = summary.set_index("method").loc[ordered_methods].reset_index()

    baseline_values = {
        method: float(plot_df.loc[plot_df["method"] == method, metric].iloc[0])
        for method in BASELINE_METHODS
        if method in plot_df["method"].values
    }
    if len(baseline_values) != 2:
        baseline_df = summary[summary["method"].isin(BASELINE_METHODS)]
        baseline_values = {row["method"]: float(row[metric]) for _, row in baseline_df.iterrows()}
    if len(baseline_values) != 2:
        raise ValueError(f"Expected both baseline methods in summary for dataset '{dataset_key}'.")

    plot_df = plot_df[~plot_df["method"].isin(BASELINE_METHODS)].reset_index(drop=True)
    if plot_df.empty:
        raise ValueError(f"No segmentation methods to plot for dataset '{dataset_key}'.")

    low_baseline = min(baseline_values.values())
    high_baseline = max(baseline_values.values())
    best_value = float(plot_df[metric].max())
    best_methods = set(plot_df.loc[plot_df[metric] == best_value, "method"])

    fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 1.2), 6))
    x_positions = list(range(len(plot_df)))

    for idx, row in plot_df.iterrows():
        method = row["method"]
        value = float(row[metric])
        family = row["method_family"]
        color = FAMILY_COLORS.get(family, FAMILY_COLORS["other"])
        bottom, height = style_bar(value, low_baseline, high_baseline)
        ax.bar(idx, height, bottom=bottom, width=0.72, color=color, alpha=0.9, zorder=3)
        ax.scatter(idx, value, color=color, edgecolors="white", linewidth=0.8, s=40, zorder=4)
        if method in best_methods:
            ax.scatter(idx, value + 0.012, marker="*", s=260, color="#f4b400", edgecolors="#8a6d1d", linewidth=0.8, zorder=5)
            ax.text(idx, value + 0.028, "Best", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#8a6d1d")
        annotate_bar(ax, idx, value)

    ax.axhline(0, color="#333333", linewidth=1.0, alpha=0.6, zorder=1)
    for baseline_method in BASELINE_METHODS:
        style = BASELINE_STYLES[baseline_method]
        ax.axhline(
            baseline_values[baseline_method],
            color=style["color"],
            linewidth=2.0,
            linestyle=style["linestyle"],
            zorder=2,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([build_label(method) for method in plot_df["method"]], rotation=0, ha="center")
    ax.set_ylim(0, min(1.05, max(float(plot_df[metric].max()) + 0.08, high_baseline + 0.08)))
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset_key} - {metric.replace('_', ' ').title()}")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BASELINE_STYLES["all_at_once"]["color"],
            linestyle=BASELINE_STYLES["all_at_once"]["linestyle"],
            linewidth=2.0,
            label=f"All at once = {baseline_values['all_at_once']:.3f}",
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_STYLES["step_by_step"]["color"],
            linestyle=BASELINE_STYLES["step_by_step"]["linestyle"],
            linewidth=2.0,
            label=f"Step by step = {baseline_values['step_by_step']:.3f}",
        ),
    ]

    present_families = [family for family in ["step_based_segmentation", "token_based_segmentation", "other"] if family in plot_df["method_family"].values]
    family_labels = {
        "step_based_segmentation": "Step-based segmentation",
        "token_based_segmentation": "Token-based segmentation",
        "other": "Other segmentation",
    }
    for family in present_families:
        legend_handles.append(Patch(facecolor=FAMILY_COLORS[family], edgecolor="none", label=family_labels[family]))

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#f4b400",
            markeredgecolor="#8a6d1d",
            markersize=12,
            linewidth=0,
            label="Best segmentation method",
        )
    )

    ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import pandas as pd

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_summaries: list[pd.DataFrame] = []
    dataset_merged_frames: list[pd.DataFrame] = []

    for dataset_key in DATASET_DIRS:
        frames = load_dataset_frames(dataset_key)
        merged = merge_result_frames(frames)
        summary = summarize_dataset(dataset_key, merged)

        dataset_merged_frames.append(merged)
        dataset_summaries.append(summary)

        plot_metric(
            summary=summary,
            dataset_key=dataset_key,
            metric="agent_accuracy",
            output_path=EVAL_OUTPUT_DIR / f"{dataset_key}_agent_accuracy.png",
        )
        plot_metric(
            summary=summary,
            dataset_key=dataset_key,
            metric="step_accuracy",
            output_path=EVAL_OUTPUT_DIR / f"{dataset_key}_step_accuracy.png",
        )

    overall_merged = pd.concat(dataset_merged_frames, ignore_index=True)
    overall_summary = summarize_dataset("all_datasets", overall_merged)
    dataset_summaries.append(overall_summary)

    plot_metric(
        summary=overall_summary,
        dataset_key="all_datasets",
        metric="agent_accuracy",
        output_path=EVAL_OUTPUT_DIR / "all_datasets_agent_accuracy.png",
    )
    plot_metric(
        summary=overall_summary,
        dataset_key="all_datasets",
        metric="step_accuracy",
        output_path=EVAL_OUTPUT_DIR / "all_datasets_step_accuracy.png",
    )

    final_summary = pd.concat(dataset_summaries, ignore_index=True)
    final_summary.to_csv(EVAL_OUTPUT_DIR / "segmentation_accuracy_summary.csv", index=False)

    print(f"Saved evaluation artifacts to: {EVAL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
