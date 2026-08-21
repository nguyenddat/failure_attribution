"""Cost và latency scaling theo độ dài trajectory — exp 3 + 4 + 5.

2 panel: chi phí (USD/file) và latency (giây/file) theo `num_steps`, binned.
Bổ sung cho plot_cost_accuracy.py — Pareto chỉ cho thấy trung bình pooled, hình
này cho thấy *tại sao*: baseline all_at_once nhét cả trajectory vào 1 call nên
cost tăng theo độ dài, exp 4 quét stride-1 nên tăng tuyến tính, còn exp 5 chia
đoạn cố định nên gần phẳng.

Chỉ vẽ window_size = 9 làm đại diện cho cả 4 họ fixed-window (w9 là config
Pareto-optimal ở vùng trace dài), để mỗi panel còn 6 đường thay vì 18.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from plot_cost_accuracy import _merged  # noqa: E402
from plot_pooled_all import DATASETS, FIGURES_DIR, MODEL_NAME  # noqa: E402

REPRESENTATIVE_WINDOW = 9
BINS = [5, 10, 15, 20, 25, 30, 40, 60, 130]

# (label, color, marker, method)
SERIES = [
    ("baseline all_at_once", "tab:gray", "s", "all_at_once"),
    ("baseline step_by_step", "tab:red", "s", "step_by_step"),
    ("per-step, both", "tab:blue", "o", f"fixed_window_w{REPRESENTATIVE_WINDOW}_both"),
    ("per-step, prev_only", "tab:orange", "o", f"fixed_window_w{REPRESENTATIVE_WINDOW}_prev_only"),
    ("per-step, next_only", "tab:green", "o", f"fixed_window_w{REPRESENTATIVE_WINDOW}_next_only"),
    (
        "window all_at_once",
        "tab:purple",
        "^",
        f"fixed_window_all_at_once_w{REPRESENTATIVE_WINDOW}",
    ),
]

PANELS = [
    ("cost", "chi phí trung bình mỗi file (USD)"),
    ("latency", "latency trung bình mỗi file (giây)"),
]


def _plot_ax(ax: plt.Axes, df: pd.DataFrame, value_col: str, ylabel: str) -> None:
    for label, color, marker, method in SERIES:
        subset = df[df["method"] == method]
        if subset.empty:
            continue
        means = subset.groupby("bin_center", observed=True)[value_col].mean().sort_index()
        ax.plot(
            means.index,
            means.to_numpy(),
            marker=marker,
            color=color,
            label=label,
            linewidth=1.5,
            zorder=3,
        )

    ax.set_xlabel("num_steps (bin center)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{value_col} vs độ dài trajectory")
    ax.grid(alpha=0.3, zorder=0)


def main() -> None:
    df = _merged()
    df = df[
        df["method"].isin({method for _, _, _, method in SERIES})
    ].copy()
    bins = pd.cut(df["num_steps"], BINS, include_lowest=True)
    df["bin_center"] = bins.map(lambda b: (b.left + b.right) / 2).astype(float)

    counts = (
        df.drop_duplicates(["dataset", "file"])
        .groupby("bin_center", observed=True)
        .size()
        .sort_index()
    )
    print("số file mỗi bin:")
    print(counts.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, (value_col, ylabel) in zip(axes, PANELS):
        _plot_ax(ax, df, value_col, ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=9)
    fig.suptitle(
        f"Cost & latency scaling theo độ dài trajectory — window_size={REPRESENTATIVE_WINDOW} "
        f"({' + '.join(DATASETS)} pooled, {MODEL_NAME})"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "pooled_cost_latency_scaling.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    print(
        df.pivot_table(
            index="bin_center", columns="method", values=["cost", "latency"], aggfunc="mean"
        ).round(4)
    )


if __name__ == "__main__":
    main()
