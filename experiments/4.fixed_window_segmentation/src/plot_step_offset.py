"""Xu hướng lệch của pred_step so với gt_step (2 dataset pooled).

- Figure 1: 3 pie chart (1 mode / chart) — tỉ lệ quét hết trajectory mà
  không báo lỗi (pred_step == -1, "Not Found") so với có dự đoán.
- Figure 2: phân phối delta = pred_step - gt_step dạng line cho 3 mode,
  để xem mode nào có xu hướng đoán sớm (delta < 0) hay muộn (delta > 0).
  Chỉ tính các row có dự đoán; row Not Found không có delta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from plot_pooled import DATASETS, FIGURES_DIR, MODE_COLORS  # noqa: E402
from run import ACCURACY_PATH, MODEL_NAME  # noqa: E402
from windows import CONTEXT_MODES  # noqa: E402

NOT_FOUND_STEP = -1
DELTA_LIMIT = 30
BIN_WIDTH = 1


def _load() -> pd.DataFrame:
    df = pd.read_excel(ACCURACY_PATH)
    return df[
        (df["model"] == MODEL_NAME)
        & (df["dataset"].isin(DATASETS))
        & (df["status"] == "ok")
    ]


def _plot_not_found(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(CONTEXT_MODES), figsize=(15, 5.5))

    for ax, mode in zip(axes, CONTEXT_MODES):
        subset = df[df["context_mode"] == mode.value]
        not_found = int((subset["pred_step"] == NOT_FOUND_STEP).sum())
        predicted = len(subset) - not_found
        ax.pie(
            [not_found, predicted],
            labels=["Not Found", "có dự đoán"],
            colors=["lightgrey", MODE_COLORS[mode.value]],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        ax.set_title(f"{mode.value}\n{not_found}/{len(subset)} row Not Found")

    fig.suptitle(
        "Tỉ lệ quét hết trajectory mà không báo lỗi "
        f"({' + '.join(DATASETS)} pooled, 4 window size)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = FIGURES_DIR / "pooled_not_found_rate.png"
    fig.savefig(out_path, dpi=150)
    return out_path


def _plot_delta(df: pd.DataFrame) -> Path:
    found = df[df["pred_step"] != NOT_FOUND_STEP].copy()
    found["delta"] = found["pred_step"] - found["gt_step"]

    edges = np.arange(-DELTA_LIMIT - BIN_WIDTH / 2, DELTA_LIMIT + BIN_WIDTH, BIN_WIDTH)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(16, 4.5), constrained_layout=True)

    for mode in CONTEXT_MODES:
        deltas = found.loc[found["context_mode"] == mode.value, "delta"]
        counts, _ = np.histogram(deltas, bins=edges)
        share = counts / len(deltas)
        median = deltas.median()
        ax.plot(
            centers,
            share,
            color=MODE_COLORS[mode.value],
            label=f"{mode.value} (median {median:+.0f})",
        )
        ax.axvline(
            median,
            color=MODE_COLORS[mode.value],
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            zorder=1,
        )

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", zorder=0)
    ax.set_xlabel("độ lệch (step)")
    ax.set_ylabel("tỉ lệ")
    ax.set_title("So sánh phân phối độ lệch dự đoán")
    ax.set_xticks(np.arange(-DELTA_LIMIT, DELTA_LIMIT + 1, 5))
    ax.grid(alpha=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.legend()

    out_path = FIGURES_DIR / "pooled_step_offset_distribution.png"
    fig.savefig(out_path, dpi=150)
    return out_path


def _plot_delta_stacked(df: pd.DataFrame) -> Path:
    """Cùng dữ liệu figure 2 nhưng tách 3 panel bar xếp dọc, chung trục x.

    Mỗi panel vẽ đủ 3 đường median (không chỉ của mode mình) nên 3 đường
    chạy thẳng suốt cả stack, dễ so vị trí đỉnh giữa các mode.
    """
    found = df[df["pred_step"] != NOT_FOUND_STEP].copy()
    found["delta"] = found["pred_step"] - found["gt_step"]

    edges = np.arange(-DELTA_LIMIT - BIN_WIDTH / 2, DELTA_LIMIT + BIN_WIDTH, BIN_WIDTH)
    centers = (edges[:-1] + edges[1:]) / 2
    medians = {
        mode.value: found.loc[found["context_mode"] == mode.value, "delta"].median()
        for mode in CONTEXT_MODES
    }

    fig, axes = plt.subplots(
        len(CONTEXT_MODES),
        1,
        figsize=(16, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, mode in zip(axes, CONTEXT_MODES):
        deltas = found.loc[found["context_mode"] == mode.value, "delta"]
        counts, _ = np.histogram(deltas, bins=edges)
        ax.bar(
            centers,
            counts / len(deltas),
            width=BIN_WIDTH,
            color=MODE_COLORS[mode.value],
            label=f"{mode.value} (median {medians[mode.value]:+.0f})",
            zorder=2,
        )
        for value_mode, median in medians.items():
            ax.axvline(
                median,
                color=MODE_COLORS[value_mode],
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                zorder=3,
            )
        ax.set_ylabel("tỉ lệ")
        ax.grid(alpha=0.3, zorder=0)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("độ lệch (step)")
    axes[-1].set_xticks(np.arange(-DELTA_LIMIT, DELTA_LIMIT + 1, 5))
    axes[0].set_title("So sánh phân phối độ lệch dự đoán")

    out_path = FIGURES_DIR / "pooled_step_offset_by_mode.png"
    fig.savefig(out_path, dpi=150)
    return out_path


def main() -> None:
    df = _load()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"saved {_plot_not_found(df)}")
    print(f"saved {_plot_delta(df)}")
    print(f"saved {_plot_delta_stacked(df)}")


if __name__ == "__main__":
    main()
