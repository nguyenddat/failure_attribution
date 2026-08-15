"""Grid: sample distribution (top) / step_accuracy (below) vs num_steps.

Four figures, since telbench is a different task from who_and_when + trace_elephant,
and each dataset group gets both a full-range and a num_steps<=MAX_STEPS variant:
  - who_and_when + trace_elephant pooled (single-fault, agent-level GT).
  - telbench (multi-fault, span-based GT).

The *_capped files drop samples with num_steps > MAX_STEPS before plotting — beyond
that point per-step sample counts are too thin (1-3 samples) and the
distribution/line are just noise. The distribution row is a short, wide bar chart of
sample count per exact num_steps. The performance row overlays the two baselines
(all_at_once vs step_by_step) as line-only (no markers). Both rows share the x-axis.
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

FIGURES_DIR = SRC_DIR.parent / "results" / "figures"
MODEL = "gpt-4o-mini"
METHODS = {"all_at_once": "tab:blue", "step_by_step": "tab:orange"}
MAX_STEPS = 60
BIN_WIDTH = 3


def _plot(df: pd.DataFrame, title: str, out_name: str, bin_width: int = 1) -> None:
    df = df.copy()
    if bin_width > 1:
        df["_x"] = ((df["num_steps"] - 1) // bin_width) * bin_width + 1
        x_label = f"num_steps ({bin_width}-step bins)"
    else:
        df["_x"] = df["num_steps"]
        x_label = "num_steps (exact trace length)"

    fig, (dist_ax, perf_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    counts = df[df["method"] == "all_at_once"].groupby("_x").size().sort_index()
    dist_ax.bar(counts.index, counts.values, width=bin_width * 0.8, color="tab:gray")
    dist_ax.set_ylabel("sample count")

    for method, color in METHODS.items():
        by_step = (
            df[df["method"] == method]
            .groupby("_x")["step_accuracy"]
            .mean()
            .sort_index()
        )
        perf_ax.plot(
            by_step.index,
            by_step.values,
            linewidth=1.5,
            linestyle="--",
            color=color,
            label=method,
        )
    perf_ax.set_ylabel("step_accuracy")
    perf_ax.set_ylim(-0.05, 1.05)
    perf_ax.legend(loc="upper right")
    perf_ax.set_xlabel(x_label)

    bin_starts = sorted(df["_x"].unique())
    if bin_width > 1:
        tick_formatter = FuncFormatter(
            lambda x, _pos: f"{int(x)}-{int(x) + bin_width - 1}"
        )
    else:
        tick_formatter = FuncFormatter(lambda x, _pos: f"{int(x)}")

    for ax in (dist_ax, perf_ax):
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


DATASETS = {
    "who_and_when__hand-crafted": ("who_and_when", "tab:green"),
    "trace_elephant": ("trace_elephant", "tab:red"),
}


def _plot_datasets_compare(df: pd.DataFrame, title: str, out_name: str, bin_width: int = 1) -> None:
    df = df.copy()
    if bin_width > 1:
        df["_x"] = ((df["num_steps"] - 1) // bin_width) * bin_width + 1
        x_label = f"num_steps ({bin_width}-step bins)"
    else:
        df["_x"] = df["num_steps"]
        x_label = "num_steps (exact trace length)"

    methods = list(METHODS.keys())
    fig, axes = plt.subplots(
        1 + len(methods),
        1,
        figsize=(13, 3 + 3.5 * len(methods)),
        sharex=True,
        gridspec_kw={"height_ratios": [1] + [1.5] * len(methods)},
    )
    dist_ax, *perf_axes = axes

    dist_df = df[df["method"] == "all_at_once"]
    counts = dist_df.groupby(["dataset", "_x"]).size().unstack("dataset").fillna(0)
    bottom = None
    for dataset_key, (label, color) in DATASETS.items():
        values = counts.get(dataset_key)
        if values is None:
            continue
        dist_ax.bar(
            values.index,
            values.values,
            width=bin_width * 0.8,
            bottom=bottom,
            color=color,
            label=label,
        )
        bottom = values if bottom is None else bottom.add(values, fill_value=0)
    dist_ax.set_ylabel("sample count")
    dist_ax.legend(loc="upper right")

    for perf_ax, method in zip(perf_axes, methods):
        for dataset_key, (label, color) in DATASETS.items():
            by_step = (
                df[(df["method"] == method) & (df["dataset"] == dataset_key)]
                .groupby("_x")["step_accuracy"]
                .mean()
                .sort_index()
            )
            perf_ax.plot(
                by_step.index,
                by_step.values,
                linewidth=1.5,
                linestyle="--",
                color=color,
                label=label,
            )
        perf_ax.set_ylabel(f"step_accuracy ({method})")
        perf_ax.set_ylim(-0.05, 1.05)
        perf_ax.legend(loc="upper right")

    perf_axes[-1].set_xlabel(x_label)

    bin_starts = sorted(df["_x"].unique())
    if bin_width > 1:
        tick_formatter = FuncFormatter(
            lambda x, _pos: f"{int(x)}-{int(x) + bin_width - 1}"
        )
    else:
        tick_formatter = FuncFormatter(lambda x, _pos: f"{int(x)}")

    for ax in (dist_ax, *perf_axes):
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


def main() -> None:
    df = pd.read_excel(ACCURACY_PATH)
    df = df[(df["model"] == MODEL) & (df["status"] == "ok")]

    pooled = df[df["dataset"].isin(["who_and_when__hand-crafted", "trace_elephant"])]
    telbench = df[df["dataset"] == "telbench"]

    _plot(
        pooled,
        "Performance and distribution on trajectory length (who_and_when + trace_elephant)",
        "baselines_step_accuracy_vs_steps_single_fault.png",
    )
    _plot(
        telbench,
        "Performance and distribution on trajectory length (telbench)",
        "baselines_step_accuracy_vs_steps_telbench.png",
    )

    pooled_capped = pooled[pooled["num_steps"] <= MAX_STEPS]
    telbench_capped = telbench[telbench["num_steps"] <= MAX_STEPS]

    _plot(
        pooled_capped,
        f"Performance and distribution on trajectory length (who_and_when + trace_elephant, <= {MAX_STEPS} steps)",
        "baselines_step_accuracy_vs_steps_single_fault_capped.png",
        bin_width=BIN_WIDTH,
    )
    _plot(
        telbench_capped,
        f"Performance and distribution on trajectory length (telbench, <= {MAX_STEPS} steps)",
        "baselines_step_accuracy_vs_steps_telbench_capped.png",
        bin_width=BIN_WIDTH,
    )

    _plot_datasets_compare(
        pooled_capped,
        f"step_accuracy by baseline: who_and_when vs trace_elephant (<= {MAX_STEPS} steps)",
        "step_accuracy_by_dataset.png",
        bin_width=BIN_WIDTH,
    )


if __name__ == "__main__":
    main()
