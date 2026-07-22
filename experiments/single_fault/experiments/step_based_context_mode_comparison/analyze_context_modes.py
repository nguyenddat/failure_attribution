from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.single_fault.experiments.step_based_context_mode_comparison.run import (
    DEFAULT_MODEL_NAME,
    FIXED_NUM_STEPS,
)
from experiments.single_fault.utils.experiment_paths import (
    STEP_BASED_CONTEXT_MODE_COMPARISON_OUTPUT_DIR,
    baseline_accuracy_path,
    step_based_accuracy_path,
)
from experiments.single_fault.utils.datasets import DATASET_DIRS


DATASET_KEY = "ww_hand_crafted"
BASE_COLUMNS = ["file", "gt_agent", "gt_step"]
BASELINE_METHODS = ["all_at_once", "step_by_step"]
CONTEXT_METHODS = [
    "step_based_multi_step_w5",
    "step_based_multi_step_prev_w5",
    "step_based_multi_step_next_w5",
]
OUTPUT_DIR = STEP_BASED_CONTEXT_MODE_COMPARISON_OUTPUT_DIR / "analysis"
SURROUNDING_METHOD = "step_based_multi_step_w5"
PREV_METHOD = "step_based_multi_step_prev_w5"
NEXT_METHOD = "step_based_multi_step_next_w5"
BASELINE_STYLES = {
    "all_at_once": {"color": "#6c757d", "linestyle": "--", "label": "All at once"},
    "step_by_step": {"color": "#2b8a3e", "linestyle": ":", "label": "Step by step"},
}
METHOD_COLORS = {
    "step_based_multi_step_w5": "#e76f51",
    "step_based_multi_step_prev_w5": "#f4a261",
    "step_based_multi_step_next_w5": "#2a9d8f",
}


def load_context_mode_frames() -> list["pd.DataFrame"]:
    import pandas as pd

    frames: list[pd.DataFrame] = []

    baseline_path = baseline_accuracy_path(DATASET_KEY)
    if baseline_path.exists():
        frames.append(pd.read_csv(baseline_path))

    comparison_path = (
        STEP_BASED_CONTEXT_MODE_COMPARISON_OUTPUT_DIR
        / f"{DATASET_KEY}_step_based_context_mode.csv"
    )
    if comparison_path.exists():
        frames.append(pd.read_csv(comparison_path))

    step_based_path = step_based_accuracy_path(DATASET_KEY)
    if step_based_path.exists():
        frames.append(pd.read_csv(step_based_path))

    if not frames:
        raise FileNotFoundError("No result CSVs found for context-mode analysis.")
    return frames


def merge_result_frames(frames: list["pd.DataFrame"]) -> "pd.DataFrame":
    merged = frames[0].copy()
    for frame in frames[1:]:
        extra_columns = [
            column
            for column in frame.columns
            if column not in BASE_COLUMNS and column not in merged.columns
        ]
        merged = merged.merge(
            frame[BASE_COLUMNS + extra_columns], on=BASE_COLUMNS, how="outer"
        )
    return merged


def summarize_context_modes(df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    records: list[dict[str, float | str]] = []
    for method in BASELINE_METHODS + CONTEXT_METHODS:
        agent_col = f"{method}_agent_acc"
        step_col = f"{method}_step_acc"
        if agent_col not in df.columns or step_col not in df.columns:
            raise ValueError(f"Missing accuracy columns for method '{method}'.")

        records.append(
            {
                "method": method,
                "agent_accuracy": float(
                    pd.to_numeric(df[agent_col], errors="coerce").mean()
                ),
                "step_accuracy": float(
                    pd.to_numeric(df[step_col], errors="coerce").mean()
                ),
            }
        )

    return pd.DataFrame(records)


def build_label(method: str) -> str:
    labels = {
        "step_based_multi_step_w5": "Surrounding\nw=5",
        "step_based_multi_step_prev_w5": "Previous only\nw=5",
        "step_based_multi_step_next_w5": "Next only\nw=5",
    }
    return labels.get(method, method)


def style_bar(
    value: float, low_baseline: float, high_baseline: float
) -> tuple[float, float]:
    if value > high_baseline:
        return high_baseline, value - high_baseline
    if value < low_baseline:
        return value, low_baseline - value

    nearest_baseline = (
        high_baseline
        if abs(value - high_baseline) <= abs(value - low_baseline)
        else low_baseline
    )
    bottom = min(value, nearest_baseline)
    height = abs(value - nearest_baseline)
    return bottom, height


def annotate_bar(ax: Any, x: float, value: float) -> None:
    ax.text(x, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)


def plot_metric(summary: "pd.DataFrame", metric: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    baseline_values = {
        method: float(summary.loc[summary["method"] == method, metric].iloc[0])
        for method in BASELINE_METHODS
    }
    plot_df = summary[summary["method"].isin(CONTEXT_METHODS)].copy()

    low_baseline = min(baseline_values.values())
    high_baseline = max(baseline_values.values())
    best_value = float(plot_df[metric].max())
    best_methods = set(plot_df.loc[plot_df[metric] == best_value, "method"])

    fig, ax = plt.subplots(figsize=(9, 6))
    x_positions = list(range(len(plot_df)))

    for idx, row in plot_df.reset_index(drop=True).iterrows():
        method = row["method"]
        value = float(row[metric])
        bottom, height = style_bar(value, low_baseline, high_baseline)
        color = METHOD_COLORS[method]
        ax.bar(idx, height, bottom=bottom, width=0.72, color=color, alpha=0.9, zorder=3)
        ax.scatter(
            idx, value, color=color, edgecolors="white", linewidth=0.8, s=46, zorder=4
        )
        if method in best_methods:
            ax.scatter(
                idx,
                value + 0.012,
                marker="*",
                s=240,
                color="#f4b400",
                edgecolors="#8a6d1d",
                linewidth=0.8,
                zorder=5,
            )
        annotate_bar(ax, idx, value)

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
    ax.set_xticklabels([build_label(method) for method in plot_df["method"]])
    ax.set_ylim(
        0, min(1.05, max(float(plot_df[metric].max()) + 0.08, high_baseline + 0.08))
    )
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{DATASET_KEY} - {metric.replace('_', ' ').title()}")
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
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_case_analysis(df: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame"]:
    import pandas as pd

    surrounding_col = f"{SURROUNDING_METHOD}_step_acc"
    prev_col = f"{PREV_METHOD}_step_acc"
    next_col = f"{NEXT_METHOD}_step_acc"
    prev_pred_agent_col = f"{PREV_METHOD}_pred_agent"
    prev_pred_step_col = f"{PREV_METHOD}_pred_step"
    next_pred_agent_col = f"{NEXT_METHOD}_pred_agent"
    next_pred_step_col = f"{NEXT_METHOD}_pred_step"

    case_df = df[
        [
            "file",
            "gt_agent",
            "gt_step",
            surrounding_col,
            prev_col,
            next_col,
            prev_pred_agent_col,
            prev_pred_step_col,
            next_pred_agent_col,
            next_pred_step_col,
        ]
    ].copy()

    case_df["surrounding_correct"] = (
        pd.to_numeric(case_df[surrounding_col], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )
    case_df["prev_correct"] = (
        pd.to_numeric(case_df[prev_col], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )
    case_df["next_correct"] = (
        pd.to_numeric(case_df[next_col], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )

    def build_case_label(row: pd.Series) -> str:
        active = []
        if row["surrounding_correct"]:
            active.append("surrounding")
        if row["prev_correct"]:
            active.append("prev")
        if row["next_correct"]:
            active.append("next")
        return "+".join(active) if active else "none_correct"

    case_df["case"] = case_df.apply(build_case_label, axis=1)

    summary_df = (
        case_df.groupby("case", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("case")
        .reset_index(drop=True)
    )
    return case_df, summary_df


def plot_correctness_venn(case_df: "pd.DataFrame", output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    surrounding_only = int(
        (
            case_df["surrounding_correct"]
            & ~case_df["prev_correct"]
            & ~case_df["next_correct"]
        ).sum()
    )
    prev_only = int(
        (
            ~case_df["surrounding_correct"]
            & case_df["prev_correct"]
            & ~case_df["next_correct"]
        ).sum()
    )
    next_only = int(
        (
            ~case_df["surrounding_correct"]
            & ~case_df["prev_correct"]
            & case_df["next_correct"]
        ).sum()
    )
    surrounding_prev = int(
        (
            case_df["surrounding_correct"]
            & case_df["prev_correct"]
            & ~case_df["next_correct"]
        ).sum()
    )
    surrounding_next = int(
        (
            case_df["surrounding_correct"]
            & ~case_df["prev_correct"]
            & case_df["next_correct"]
        ).sum()
    )
    prev_next = int(
        (
            ~case_df["surrounding_correct"]
            & case_df["prev_correct"]
            & case_df["next_correct"]
        ).sum()
    )
    all_three = int(
        (
            case_df["surrounding_correct"]
            & case_df["prev_correct"]
            & case_df["next_correct"]
        ).sum()
    )
    neither = int(
        (
            ~case_df["surrounding_correct"]
            & ~case_df["prev_correct"]
            & ~case_df["next_correct"]
        ).sum()
    )
    total = len(case_df)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    top_center = (0.5, 0.66)
    left_center = (0.40, 0.46)
    right_center = (0.60, 0.46)
    radius = 0.22

    ax.add_patch(
        Circle(
            top_center,
            radius,
            facecolor="#e76f51",
            alpha=0.48,
            edgecolor="#a13d22",
            linewidth=2,
        )
    )
    ax.add_patch(
        Circle(
            left_center,
            radius,
            facecolor="#f4a261",
            alpha=0.48,
            edgecolor="#9c5c12",
            linewidth=2,
        )
    )
    ax.add_patch(
        Circle(
            right_center,
            radius,
            facecolor="#2a9d8f",
            alpha=0.48,
            edgecolor="#1d6f65",
            linewidth=2,
        )
    )

    region_style = {"fontsize": 16, "fontweight": "bold"}
    zero_region_style = {"fontsize": 14, "fontweight": "bold", "color": "#666666"}

    def region_text(x: float, y: float, value: int) -> None:
        ax.text(
            x,
            y,
            str(value),
            ha="center",
            va="center",
            **(region_style if value else zero_region_style),
        )

    region_text(0.50, 0.80, surrounding_only)
    region_text(0.28, 0.41, prev_only)
    region_text(0.72, 0.41, next_only)
    region_text(0.43, 0.57, surrounding_prev)
    region_text(0.57, 0.57, surrounding_next)
    region_text(0.50, 0.34, prev_next)
    region_text(0.50, 0.48, all_three)

    ax.text(
        0.43, 0.63, "sur+prev", ha="center", va="center", fontsize=9, color="#444444"
    )
    ax.text(
        0.57, 0.63, "sur+next", ha="center", va="center", fontsize=9, color="#444444"
    )
    ax.text(
        0.50, 0.28, "prev+next", ha="center", va="center", fontsize=9, color="#444444"
    )
    ax.text(0.50, 0.53, "all 3", ha="center", va="center", fontsize=9, color="#444444")

    ax.text(
        top_center[0],
        0.93,
        "Surrounding\nw=5",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#a13d22",
    )
    ax.text(
        left_center[0] - 0.15,
        0.18,
        "Previous only\nw=5",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#9c5c12",
    )
    ax.text(
        right_center[0] + 0.15,
        0.18,
        "Next only\nw=5",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#1d6f65",
    )
    ax.text(
        0.50,
        0.08,
        f"None correct: {neither}\nTotal: {total}",
        ha="center",
        va="center",
        fontsize=11,
    )

    ax.set_title(
        f"{DATASET_KEY} - Step Correctness Overlap", fontsize=15, fontweight="bold"
    )
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.02, 0.98)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_step_error_analysis(
    df: "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    import pandas as pd

    analysis_df = df[
        [
            "file",
            "gt_step",
            f"{PREV_METHOD}_pred_step",
            f"{NEXT_METHOD}_pred_step",
        ]
    ].copy()

    analysis_df["prev_pred_step"] = pd.to_numeric(
        analysis_df[f"{PREV_METHOD}_pred_step"], errors="coerce"
    )
    analysis_df["next_pred_step"] = pd.to_numeric(
        analysis_df[f"{NEXT_METHOD}_pred_step"], errors="coerce"
    )
    analysis_df["gt_step"] = pd.to_numeric(analysis_df["gt_step"], errors="coerce")

    analysis_df["prev_signed_error"] = (
        analysis_df["prev_pred_step"] - analysis_df["gt_step"]
    )
    analysis_df["next_signed_error"] = (
        analysis_df["next_pred_step"] - analysis_df["gt_step"]
    )
    analysis_df["prev_abs_error"] = analysis_df["prev_signed_error"].abs()
    analysis_df["next_abs_error"] = analysis_df["next_signed_error"].abs()

    records: list[dict[str, float | str]] = []
    for label, signed_col, abs_col in [
        ("prev_w5", "prev_signed_error", "prev_abs_error"),
        ("next_w5", "next_signed_error", "next_abs_error"),
    ]:
        signed = pd.to_numeric(analysis_df[signed_col], errors="coerce")
        abs_error = pd.to_numeric(analysis_df[abs_col], errors="coerce")
        valid = signed.dropna()

        records.append(
            {
                "method": label,
                "mean_signed_error": float(valid.mean()),
                "median_signed_error": float(valid.median()),
                "mean_abs_error": float(abs_error.dropna().mean()),
                "median_abs_error": float(abs_error.dropna().median()),
                "early_rate": float((valid < 0).mean()),
                "exact_rate": float((valid == 0).mean()),
                "late_rate": float((valid > 0).mean()),
            }
        )

    summary_df = pd.DataFrame(records)
    return analysis_df, summary_df


def plot_step_error_distributions(
    analysis_df: "pd.DataFrame", output_path: Path
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    prev_signed = analysis_df["prev_signed_error"].dropna().astype(int)
    next_signed = analysis_df["next_signed_error"].dropna().astype(int)
    prev_abs = analysis_df["prev_abs_error"].dropna().astype(int)
    next_abs = analysis_df["next_abs_error"].dropna().astype(int)

    signed_min = int(min(prev_signed.min(), next_signed.min()))
    signed_max = int(max(prev_signed.max(), next_signed.max()))
    signed_bins = np.arange(signed_min - 0.5, signed_max + 1.5, 1)

    abs_max = int(max(prev_abs.max(), next_abs.max()))
    abs_bins = np.arange(-0.5, abs_max + 1.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    axes[0].hist(
        prev_signed,
        bins=signed_bins,
        alpha=0.6,
        color="#f4a261",
        edgecolor="white",
        label="prev_w5",
    )
    axes[0].hist(
        next_signed,
        bins=signed_bins,
        alpha=0.6,
        color="#2a9d8f",
        edgecolor="white",
        label="next_w5",
    )
    axes[0].axvline(0, color="#333333", linestyle="--", linewidth=1.5)
    axes[0].set_title("Signed Step Error Distribution")
    axes[0].set_xlabel("pred_step - gt_step")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].hist(
        prev_abs,
        bins=abs_bins,
        alpha=0.6,
        color="#f4a261",
        edgecolor="white",
        label="prev_w5",
    )
    axes[1].hist(
        next_abs,
        bins=abs_bins,
        alpha=0.6,
        color="#2a9d8f",
        edgecolor="white",
        label="next_w5",
    )
    axes[1].set_title("Absolute Step Error Distribution")
    axes[1].set_xlabel("|pred_step - gt_step|")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(f"{DATASET_KEY} - Step Error Analysis", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_position_bucket_analysis(
    df: "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    import pandas as pd

    dataset_dir = DATASET_DIRS[DATASET_KEY]
    analysis_df = df[
        ["file", "gt_step", f"{PREV_METHOD}_step_acc", f"{NEXT_METHOD}_step_acc"]
    ].copy()

    def trajectory_length(file_name: str) -> int:
        row = json.loads((dataset_dir / file_name).read_text(encoding="utf-8"))
        return len(row.get("trajectory", []))

    analysis_df["trajectory_length"] = analysis_df["file"].map(trajectory_length)
    analysis_df["gt_step"] = pd.to_numeric(analysis_df["gt_step"], errors="coerce")
    analysis_df["relative_position"] = analysis_df["gt_step"] / analysis_df[
        "trajectory_length"
    ].clip(lower=1)

    def bucket_position(value: float) -> str:
        if value < (1 / 3):
            return "head"
        if value < (2 / 3):
            return "middle"
        return "tail"

    analysis_df["position_bucket"] = analysis_df["relative_position"].map(
        bucket_position
    )
    analysis_df["prev_step_acc"] = pd.to_numeric(
        analysis_df[f"{PREV_METHOD}_step_acc"], errors="coerce"
    )
    analysis_df["next_step_acc"] = pd.to_numeric(
        analysis_df[f"{NEXT_METHOD}_step_acc"], errors="coerce"
    )

    summary_df = analysis_df.groupby("position_bucket", as_index=False).agg(
        count=("file", "size"),
        prev_step_accuracy=("prev_step_acc", "mean"),
        next_step_accuracy=("next_step_acc", "mean"),
        mean_relative_position=("relative_position", "mean"),
    )
    summary_df["accuracy_gap_prev_minus_next"] = (
        summary_df["prev_step_accuracy"] - summary_df["next_step_accuracy"]
    )

    bucket_order = ["head", "middle", "tail"]
    summary_df["position_bucket"] = pd.Categorical(
        summary_df["position_bucket"], categories=bucket_order, ordered=True
    )
    summary_df = summary_df.sort_values("position_bucket").reset_index(drop=True)
    analysis_df["position_bucket"] = pd.Categorical(
        analysis_df["position_bucket"], categories=bucket_order, ordered=True
    )
    analysis_df = analysis_df.sort_values(["position_bucket", "file"]).reset_index(
        drop=True
    )
    return analysis_df, summary_df


def build_qualitative_error_cases(
    df: "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    import pandas as pd

    dataset_dir = DATASET_DIRS[DATASET_KEY]
    qualitative_df = df[
        [
            "file",
            "gt_agent",
            "gt_step",
            f"{PREV_METHOD}_step_acc",
            f"{NEXT_METHOD}_step_acc",
            f"{PREV_METHOD}_pred_agent",
            f"{PREV_METHOD}_pred_step",
            f"{NEXT_METHOD}_pred_agent",
            f"{NEXT_METHOD}_pred_step",
        ]
    ].copy()

    def load_question(file_name: str) -> str:
        row = json.loads((dataset_dir / file_name).read_text(encoding="utf-8"))
        return str(row.get("question", ""))

    def load_trajectory_length(file_name: str) -> int:
        row = json.loads((dataset_dir / file_name).read_text(encoding="utf-8"))
        return len(row.get("trajectory", []))

    qualitative_df["question"] = qualitative_df["file"].map(load_question)
    qualitative_df["trajectory_length"] = qualitative_df["file"].map(
        load_trajectory_length
    )
    qualitative_df["prev_correct"] = (
        pd.to_numeric(qualitative_df[f"{PREV_METHOD}_step_acc"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )
    qualitative_df["next_correct"] = (
        pd.to_numeric(qualitative_df[f"{NEXT_METHOD}_step_acc"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )

    qualitative_df["pairwise_case"] = "both_wrong"
    qualitative_df.loc[
        qualitative_df["prev_correct"] & ~qualitative_df["next_correct"],
        "pairwise_case",
    ] = "prev_beats_next"
    qualitative_df.loc[
        ~qualitative_df["prev_correct"] & qualitative_df["next_correct"],
        "pairwise_case",
    ] = "next_beats_prev"
    qualitative_df.loc[
        qualitative_df["prev_correct"] & qualitative_df["next_correct"], "pairwise_case"
    ] = "both_correct"

    qualitative_df = qualitative_df.sort_values("file").reset_index(drop=True)

    selected_frames: list[pd.DataFrame] = []
    for case_name in ["prev_beats_next", "next_beats_prev", "both_wrong"]:
        case_rows = (
            qualitative_df[qualitative_df["pairwise_case"] == case_name].head(5).copy()
        )
        selected_frames.append(case_rows)

    selected_df = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else qualitative_df.head(0).copy()
    )
    return qualitative_df, selected_df


def write_qualitative_markdown(selected_df: "pd.DataFrame", output_path: Path) -> None:
    sections: list[str] = ["# Qualitative Error Cases", ""]

    case_titles = {
        "prev_beats_next": "Prev Beats Next",
        "next_beats_prev": "Next Beats Prev",
        "both_wrong": "Both Wrong",
    }

    for case_name in ["prev_beats_next", "next_beats_prev", "both_wrong"]:
        case_df = selected_df[selected_df["pairwise_case"] == case_name].copy()
        sections.append(f"## {case_titles[case_name]}")
        sections.append("")
        if case_df.empty:
            sections.append("_No rows_")
            sections.append("")
            continue

        display_df = case_df[
            [
                "file",
                "gt_agent",
                "gt_step",
                "trajectory_length",
                f"{PREV_METHOD}_pred_agent",
                f"{PREV_METHOD}_pred_step",
                f"{NEXT_METHOD}_pred_agent",
                f"{NEXT_METHOD}_pred_step",
                "question",
            ]
        ].copy()
        display_df = display_df.rename(
            columns={
                f"{PREV_METHOD}_pred_agent": "prev_pred_agent",
                f"{PREV_METHOD}_pred_step": "prev_pred_step",
                f"{NEXT_METHOD}_pred_agent": "next_pred_agent",
                f"{NEXT_METHOD}_pred_step": "next_pred_step",
            }
        )
        sections.append(dataframe_to_markdown(display_df))
        sections.append("")

    output_path.write_text("\n".join(sections), encoding="utf-8")


def plot_position_bucket_comparison(
    summary_df: "pd.DataFrame", output_path: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    buckets = summary_df["position_bucket"].astype(str).tolist()
    x_positions = list(range(len(buckets)))
    prev_values = summary_df["prev_step_accuracy"].tolist()
    next_values = summary_df["next_step_accuracy"].tolist()
    labels = [
        f"{bucket}\n(n={count})"
        for bucket, count in zip(buckets, summary_df["count"].tolist())
    ]

    ax.plot(
        x_positions,
        prev_values,
        marker="o",
        linewidth=2.4,
        markersize=7,
        color="#f4a261",
        label="prev_w5",
    )
    ax.plot(
        x_positions,
        next_values,
        marker="o",
        linewidth=2.4,
        markersize=7,
        color="#2a9d8f",
        label="next_w5",
    )

    for x, y in zip(x_positions, prev_values):
        ax.text(
            x,
            y + 0.02,
            f"{y:.2f}",
            color="#9c5c12",
            fontsize=9,
            ha="center",
            va="bottom",
        )
    for x, y in zip(x_positions, next_values):
        ax.text(
            x, y - 0.05, f"{y:.2f}", color="#1d6f65", fontsize=9, ha="center", va="top"
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Step Accuracy")
    ax.set_xlabel("Error position bucket")
    ax.set_title(
        f"{DATASET_KEY} - Step Accuracy by Error Position",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    accuracy_summary_df: "pd.DataFrame",
    case_summary_df: "pd.DataFrame",
    step_error_summary_df: "pd.DataFrame",
    position_bucket_summary_df: "pd.DataFrame",
    qualitative_selected_df: "pd.DataFrame",
    output_path: Path,
) -> None:
    import pandas as pd

    def accuracy_row(method: str) -> pd.Series:
        row = accuracy_summary_df.loc[accuracy_summary_df["method"] == method]
        if row.empty:
            raise ValueError(f"Missing accuracy summary row for method '{method}'.")
        return row.iloc[0]

    def case_count(case_name: str) -> int:
        row = case_summary_df.loc[case_summary_df["case"] == case_name, "count"]
        return int(row.iloc[0]) if not row.empty else 0

    surrounding = accuracy_row(SURROUNDING_METHOD)
    prev = accuracy_row(PREV_METHOD)
    next_row = accuracy_row(NEXT_METHOD)
    all_at_once = accuracy_row("all_at_once")
    step_by_step = accuracy_row("step_by_step")

    prev_error = step_error_summary_df.loc[
        step_error_summary_df["method"] == "prev_w5"
    ].iloc[0]
    next_error = step_error_summary_df.loc[
        step_error_summary_df["method"] == "next_w5"
    ].iloc[0]

    head_row = position_bucket_summary_df.loc[
        position_bucket_summary_df["position_bucket"] == "head"
    ].iloc[0]
    middle_row = position_bucket_summary_df.loc[
        position_bucket_summary_df["position_bucket"] == "middle"
    ].iloc[0]
    tail_row = position_bucket_summary_df.loc[
        position_bucket_summary_df["position_bucket"] == "tail"
    ].iloc[0]

    lines = [
        f"# Báo Cáo Experiment {DATASET_KEY} Cho So Sánh Context Mode",
        "",
        "## 1. Thiết Lập Thí Nghiệm",
        f"- Tập dữ liệu: `{DATASET_KEY}`",
        f"- Mô hình sử dụng: `{DEFAULT_MODEL_NAME}`",
        f"- Kích thước cửa sổ cố định: `{FIXED_NUM_STEPS}`",
        "- Các chế độ ngữ cảnh được so sánh:",
        "  - `surrounding_w5`",
        "  - `prev_w5`",
        "  - `next_w5`",
        "- Baseline dùng để đối chiếu:",
        "  - `all_at_once`",
        "  - `step_by_step`",
        "",
        "## 2. Kết Luận Chính",
        f"- Về `step_accuracy`, phương án tốt nhất trong ba context mode là `prev_w5` với giá trị `{prev['step_accuracy']:.3f}`.",
        f"- Về `agent_accuracy`, `surrounding_w5` là phương án tốt nhất trong ba context mode với giá trị `{surrounding['agent_accuracy']:.3f}`.",
        f"- So với baseline, `step_by_step` có `step_accuracy` bằng với `prev_w5` ở mức `{step_by_step['step_accuracy']:.3f}`, trong khi `all_at_once` thấp hơn ở mức `{all_at_once['step_accuracy']:.3f}`.",
        f"- `next_w5` là phương án yếu nhất trong ba context mode theo `step_accuracy`, chỉ đạt `{next_row['step_accuracy']:.3f}`.",
        "",
        "## 3. Bảng Tóm Tắt Accuracy",
        "",
        dataframe_to_markdown(accuracy_summary_df),
        "",
        "## 4. Phân Tích Giao Nhau Giữa Các Case Đúng",
        f"- Chỉ `surrounding` đúng: `{case_count('surrounding')}`",
        f"- Chỉ `prev` đúng: `{case_count('prev')}`",
        f"- Chỉ `next` đúng: `{case_count('next')}`",
        f"- `surrounding + prev` cùng đúng: `{case_count('surrounding+prev')}`",
        f"- `surrounding + next` cùng đúng: `{case_count('surrounding+next')}`",
        f"- `prev + next` cùng đúng: `{case_count('prev+next')}`",
        f"- Cả ba cùng đúng: `{case_count('surrounding+prev+next')}`",
        f"- Không phương án nào đúng: `{case_count('none_correct')}`",
        "- Diễn giải:",
        f"  - `prev_w5` tạo ra nhiều chiến thắng riêng hơn `next_w5` (`{case_count('prev')}` so với `{case_count('next')}`).",
        "  - Không xuất hiện case chỉ đúng theo cặp `surrounding+next` hoặc `prev+next` nếu các giá trị này vẫn bằng 0.",
        "",
        "## 5. Phân Tích Độ Lệch Step",
        "",
        dataframe_to_markdown(step_error_summary_df),
        "",
        f"- `prev_w5` có `mean signed error` bằng `{prev_error['mean_signed_error']:.3f}` và `median signed error` bằng `{prev_error['median_signed_error']:.1f}`. Điều này cho thấy phương án này không có thiên lệch mạnh theo hướng đoán sớm hay đoán muộn trên toàn bộ tập.",
        f"- `next_w5` có `mean signed error` bằng `{next_error['mean_signed_error']:.3f}` và `median signed error` bằng `{next_error['median_signed_error']:.1f}`. Dấu âm cho thấy phương án này có xu hướng đoán sớm hơn ground-truth step, chứ không phải muộn hơn.",
        f"- `mean absolute error` của hai phương án khá gần nhau: `prev_w5 = {prev_error['mean_abs_error']:.3f}`, `next_w5 = {next_error['mean_abs_error']:.3f}`.",
        "",
        "## 6. Phân Tích Theo Vị Trí Lỗi Trong Trajectory",
        "",
        dataframe_to_markdown(position_bucket_summary_df),
        "",
        f"- Ở bucket `head`: `prev_w5 = {head_row['prev_step_accuracy']:.3f}`, `next_w5 = {head_row['next_step_accuracy']:.3f}`.",
        f"- Ở bucket `middle`: `prev_w5 = {middle_row['prev_step_accuracy']:.3f}`, `next_w5 = {middle_row['next_step_accuracy']:.3f}`.",
        f"- Ở bucket `tail`: `prev_w5 = {tail_row['prev_step_accuracy']:.3f}`, `next_w5 = {tail_row['next_step_accuracy']:.3f}`.",
        "- Diễn giải:",
        f"  - Giả thuyết `prev_w5` hoạt động tốt hơn khi lỗi nằm cuối trajectory được ủng hộ, nhưng mức chênh không lớn: khoảng cách ở bucket `tail` là `{tail_row['accuracy_gap_prev_minus_next']:.3f}`.",
        f"  - Giả thuyết `next_w5` hoạt động tốt hơn khi lỗi nằm đầu trajectory không được ủng hộ trên tập này, vì ngay cả ở bucket `head`, `prev_w5` vẫn tốt hơn với khoảng cách `{head_row['accuracy_gap_prev_minus_next']:.3f}`.",
        f"  - Khoảng cách lớn nhất giữa `prev_w5` và `next_w5` xuất hiện ở bucket `middle`, với chênh lệch `{middle_row['accuracy_gap_prev_minus_next']:.3f}` nghiêng về `prev_w5`.",
        "",
        "## 7. Bảng Qualitative Error Cases",
        "- Đã sinh riêng một bảng qualitative với tối đa 5 ví dụ cho mỗi nhóm:",
        "  - `prev_beats_next`",
        "  - `next_beats_prev`",
        "  - `both_wrong`",
        f"- Tổng số dòng qualitative được chọn: `{len(qualitative_selected_df)}`.",
        "",
        "## 8. Các Artifact Được Sinh Ra",
        "- `ww_hand_crafted_agent_accuracy.png`",
        "- `ww_hand_crafted_step_accuracy.png`",
        "- `ww_hand_crafted_context_mode_step_correct_venn.png`",
        "- `ww_hand_crafted_step_error_distribution.png`",
        "- `ww_hand_crafted_position_bucket_comparison.png`",
        "- `ww_hand_crafted_context_mode_cases.csv`",
        "- `ww_hand_crafted_step_error_cases.csv`",
        "- `ww_hand_crafted_position_bucket_cases.csv`",
        "- `ww_hand_crafted_qualitative_error_cases.csv`",
        "- `qualitative_error_cases.md`",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: "pd.DataFrame") -> str:

    if df.empty:
        return "_No rows_"

    headers = [str(column) for column in df.columns]
    rows = [
        [format_markdown_value(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    separator = ["---"] * len(headers)

    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(table_lines)


def format_markdown_value(value: object) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def main() -> None:
    frames = load_context_mode_frames()
    merged = merge_result_frames(frames)
    summary = summarize_context_modes(merged)
    case_df, case_summary_df = build_case_analysis(merged)
    step_error_df, step_error_summary_df = build_step_error_analysis(merged)
    position_bucket_df, position_bucket_summary_df = build_position_bucket_analysis(
        merged
    )
    qualitative_df, qualitative_selected_df = build_qualitative_error_cases(merged)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        OUTPUT_DIR / f"{DATASET_KEY}_context_mode_accuracy_summary.csv", index=False
    )
    case_df.to_csv(OUTPUT_DIR / f"{DATASET_KEY}_context_mode_cases.csv", index=False)
    case_summary_df.to_csv(
        OUTPUT_DIR / f"{DATASET_KEY}_context_mode_case_summary.csv", index=False
    )
    step_error_df.to_csv(
        OUTPUT_DIR / f"{DATASET_KEY}_step_error_cases.csv", index=False
    )
    step_error_summary_df.to_csv(
        OUTPUT_DIR / f"{DATASET_KEY}_step_error_summary.csv", index=False
    )
    position_bucket_df.to_csv(
        OUTPUT_DIR / f"{DATASET_KEY}_position_bucket_cases.csv", index=False
    )
    position_bucket_summary_df.to_csv(
        OUTPUT_DIR / f"{DATASET_KEY}_position_bucket_summary.csv", index=False
    )
    qualitative_df.to_csv(
        OUTPUT_DIR / f"{DATASET_KEY}_qualitative_error_pool.csv", index=False
    )
    qualitative_selected_df.to_csv(
        OUTPUT_DIR / f"{DATASET_KEY}_qualitative_error_cases.csv", index=False
    )
    plot_metric(
        summary,
        metric="agent_accuracy",
        output_path=OUTPUT_DIR / f"{DATASET_KEY}_agent_accuracy.png",
    )
    plot_metric(
        summary,
        metric="step_accuracy",
        output_path=OUTPUT_DIR / f"{DATASET_KEY}_step_accuracy.png",
    )
    plot_correctness_venn(
        case_df,
        output_path=OUTPUT_DIR / f"{DATASET_KEY}_context_mode_step_correct_venn.png",
    )
    plot_step_error_distributions(
        step_error_df,
        output_path=OUTPUT_DIR / f"{DATASET_KEY}_step_error_distribution.png",
    )
    plot_position_bucket_comparison(
        position_bucket_summary_df,
        output_path=OUTPUT_DIR / f"{DATASET_KEY}_position_bucket_comparison.png",
    )
    write_markdown_report(
        accuracy_summary_df=summary,
        case_summary_df=case_summary_df,
        step_error_summary_df=step_error_summary_df,
        position_bucket_summary_df=position_bucket_summary_df,
        qualitative_selected_df=qualitative_selected_df,
        output_path=OUTPUT_DIR / "report.md",
    )
    write_qualitative_markdown(
        qualitative_selected_df,
        output_path=OUTPUT_DIR / "qualitative_error_cases.md",
    )

    print(f"Saved context-mode analysis artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
