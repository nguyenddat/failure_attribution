"""Như plot_pooled_all.py nhưng tách riêng 2 dataset thay vì gộp chung.

Mỗi figure là lưới 2x2: hàng = dataset (who_and_when__hand-crafted,
trace_elephant), cột = short/long theo ngưỡng 30 steps. Dùng để kiểm tra kết
luận từ bản pooled có đúng trên từng dataset hay chỉ do một dataset chi phối
(trace_elephant có 219/277 file nên áp đảo bản gộp).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from plot_pooled_all import (  # noqa: E402
    ACCURACY_PATHS,
    DATASETS,
    FIGURES_DIR,
    METRICS,
    MIN_STEPS_LONG,
    _load,
    _plot_ax,
)


def main() -> None:
    baselines = _load(ACCURACY_PATHS["baselines"])
    shared_columns = [
        "model",
        "dataset",
        "method",
        "file",
        "step_accuracy",
        "agent_accuracy",
        "num_steps",
    ]
    windows = pd.concat(
        [
            _load(ACCURACY_PATHS["step"])[shared_columns],
            _load(ACCURACY_PATHS["all_at_once"])[shared_columns],
        ],
        ignore_index=True,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for value_col, name_part in METRICS:
        fig, axes = plt.subplots(
            len(DATASETS), 2, figsize=(16, 12), sharey=True, constrained_layout=True
        )

        y_max = 0.0
        for row, dataset in enumerate(DATASETS):
            b = baselines[baselines["dataset"] == dataset]
            w = windows[windows["dataset"] == dataset]
            for col, is_long in enumerate([False, True]):
                mask_b = (
                    b["num_steps"] > MIN_STEPS_LONG
                    if is_long
                    else b["num_steps"] <= MIN_STEPS_LONG
                )
                mask_w = (
                    w["num_steps"] > MIN_STEPS_LONG
                    if is_long
                    else w["num_steps"] <= MIN_STEPS_LONG
                )
                comparison = ">" if is_long else "<="
                y_max = max(
                    y_max,
                    _plot_ax(
                        axes[row][col],
                        b[mask_b],
                        w[mask_w],
                        f"{dataset} — num_steps {comparison} {MIN_STEPS_LONG}",
                        value_col,
                    ),
                )

        for row in axes:
            for ax in row:
                ax.set_ylim(0, y_max * 1.15)

        fig.suptitle("Overall performance of all segmentation settings")

        out_path = (
            FIGURES_DIR / f"by_dataset_overall_{name_part}_by_length_all_methods.png"
        )
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
