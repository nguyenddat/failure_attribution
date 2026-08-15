"""step_accuracy vs num_steps for fixed-window-all-at-once methods.

Same x-axis idea as experiment 3's plot_baselines_vs_steps.py, but no
distribution row: single axis. Baselines (all_at_once, step_by_step) are
reused from experiment 3's accuracy.xlsx and drawn faded in the background;
the new fixed-window methods (w5/w7/w9/w11) are drawn on top, full color.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from run import ACCURACY_PATH  # noqa: E402

EXP3_ACCURACY_PATH = (
    SRC_DIR.parent.parent
    / "3.trace_length_performance_cliff"
    / "results"
    / "tables"
    / "accuracy.xlsx"
)
FIGURES_DIR = SRC_DIR.parent / "results" / "figures"
MODEL = "gpt-4o-mini"
BASELINE_METHODS = {"all_at_once": "tab:gray", "step_by_step": "tab:gray"}
WINDOW_METHODS = {
    "fixed_window_all_at_once_w5": "tab:purple",
    "fixed_window_all_at_once_w7": "tab:pink",
    "fixed_window_all_at_once_w9": "tab:brown",
    "fixed_window_all_at_once_w11": "tab:olive",
}
MAX_STEPS = 60
BIN_WIDTH = 3
MIN_STEPS_LONG = 22


def _plot(baselines: pd.DataFrame, windows: pd.DataFrame, title: str, out_name: str, bin_width: int = 1) -> None:
    baselines = baselines.copy()
    windows = windows.copy()
    if bin_width > 1:
        for d in (baselines, windows):
            d["_x"] = ((d["num_steps"] - 1) // bin_width) * bin_width + 1
        x_label = f"num_steps ({bin_width}-step bins)"
    else:
        for d in (baselines, windows):
            d["_x"] = d["num_steps"]
        x_label = "num_steps (exact trace length)"

    fig, ax = plt.subplots(figsize=(13, 6))

    for method, color in BASELINE_METHODS.items():
        by_step = (
            baselines[baselines["method"] == method]
            .groupby("_x")["step_accuracy"]
            .mean()
            .sort_index()
        )
        ax.plot(
            by_step.index,
            by_step.values,
            linewidth=1.2,
            linestyle="--",
            color=color,
            alpha=0.35,
            zorder=1,
            label=method,
        )

    for method, color in WINDOW_METHODS.items():
        by_step = (
            windows[windows["method"] == method]
            .groupby("_x")["step_accuracy"]
            .mean()
            .sort_index()
        )
        ax.plot(
            by_step.index,
            by_step.values,
            linewidth=1.8,
            linestyle="-",
            color=color,
            zorder=2,
            label=method,
        )

    ax.set_ylabel("step_accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(x_label)
    ax.legend(loc="upper right")

    bin_starts = sorted(set(baselines["_x"].unique()) | set(windows["_x"].unique()))
    if bin_width > 1:
        tick_formatter = FuncFormatter(
            lambda x, _pos: f"{int(x)}-{int(x) + bin_width - 1}"
        )
    else:
        tick_formatter = FuncFormatter(lambda x, _pos: f"{int(x)}")

    ax.xaxis.set_tick_params(labelbottom=True)
    ax.xaxis.set_major_locator(FixedLocator(bin_starts))
    ax.xaxis.set_major_formatter(tick_formatter)
    ax.tick_params(axis="x", rotation=90)

    fig.suptitle(title)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def _plot_overall_bar_ax(ax: plt.Axes, baselines: pd.DataFrame, windows: pd.DataFrame, subtitle: str) -> float:
    baseline_means = baselines.groupby("method")["step_accuracy"].mean()
    window_means = windows.groupby("method")["step_accuracy"].mean().reindex(WINDOW_METHODS.keys())
    best_baseline = baseline_means.max()

    bar_colors = [
        WINDOW_METHODS[method] if value > best_baseline else "tab:gray"
        for method, value in window_means.items()
    ]
    ax.bar(window_means.index, window_means.values, color=bar_colors, zorder=2)

    line_styles = {"all_at_once": "--", "step_by_step": ":"}
    for method, value in baseline_means.items():
        ax.axhline(
            value,
            color="black",
            linewidth=1.2,
            linestyle=line_styles.get(method, "--"),
            zorder=3,
            label=f"{method} ({value:.3f})",
        )

    ax.set_ylabel("step_accuracy (mean)")
    ax.set_title(subtitle)
    ax.set_xticks(range(len(window_means.index)))
    ax.set_xticklabels(window_means.index, rotation=20, ha="right")
    ax.legend(loc="upper right")

    return max(window_means.max(), best_baseline)


def _plot_overall_bar(
    baselines_short: pd.DataFrame,
    windows_short: pd.DataFrame,
    baselines_long: pd.DataFrame,
    windows_long: pd.DataFrame,
    title: str,
    out_name: str,
) -> None:
    fig, (ax_short, ax_long) = plt.subplots(
        1, 2, figsize=(16, 6), sharey=True, constrained_layout=True
    )

    y_max_short = _plot_overall_bar_ax(
        ax_short, baselines_short, windows_short, f"num_steps <= {MIN_STEPS_LONG}"
    )
    y_max_long = _plot_overall_bar_ax(
        ax_long, baselines_long, windows_long, f"num_steps > {MIN_STEPS_LONG}"
    )

    y_max = max(y_max_short, y_max_long) * 1.15
    ax_short.set_ylim(0, y_max)
    ax_long.set_ylim(0, y_max)

    fig.suptitle(title)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main() -> None:
    exp3 = pd.read_excel(EXP3_ACCURACY_PATH)
    exp3 = exp3[(exp3["model"] == MODEL) & (exp3["status"] == "ok")]
    baselines = exp3[
        exp3["dataset"].isin(["who_and_when__hand-crafted", "trace_elephant"])
    ]

    windows = pd.read_excel(ACCURACY_PATH)
    windows = windows[(windows["model"] == MODEL) & (windows["status"] == "ok")]

    _plot(
        baselines,
        windows,
        "step_accuracy vs trajectory length: fixed-window-all-at-once vs baselines (who_and_when + trace_elephant)",
        "fixed_window_step_accuracy_vs_steps.png",
    )

    baselines_capped = baselines[baselines["num_steps"] <= MAX_STEPS]
    windows_capped = windows[windows["num_steps"] <= MAX_STEPS]

    _plot(
        baselines_capped,
        windows_capped,
        f"step_accuracy vs trajectory length: fixed-window-all-at-once vs baselines (who_and_when + trace_elephant, <= {MAX_STEPS} steps)",
        "fixed_window_step_accuracy_vs_steps_capped.png",
        bin_width=BIN_WIDTH,
    )

    baselines_short = baselines[baselines["num_steps"] <= MIN_STEPS_LONG]
    windows_short = windows[windows["num_steps"] <= MIN_STEPS_LONG]
    baselines_long = baselines[baselines["num_steps"] > MIN_STEPS_LONG]
    windows_long = windows[windows["num_steps"] > MIN_STEPS_LONG]

    _plot_overall_bar(
        baselines_short,
        windows_short,
        baselines_long,
        windows_long,
        "Overall step_accuracy: short vs long trajectories (who_and_when + trace_elephant)",
        "fixed_window_overall_accuracy_by_length.png",
    )


if __name__ == "__main__":
    main()
