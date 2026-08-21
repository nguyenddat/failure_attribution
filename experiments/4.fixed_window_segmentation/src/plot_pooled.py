"""Overall step_accuracy: short vs long trajectories (2 dataset pooled).

Cùng dạng với experiment 5's plot_vs_steps.py::_plot_overall_bar — 2 panel
chia theo num_steps, bar là 12 config fixed-window, baseline all_at_once /
step_by_step lấy từ experiment 3 vẽ thành đường ngang.

who_and_when__hand-crafted + trace_elephant gộp chung (không tách dataset);
ngưỡng chia short/long là breaking point đo được ở experiment 3 (~30 steps,
chỗ baseline tụt xuống mức random guess), thay cho ngưỡng median 22 dùng
trước đó — median chỉ chia đôi tập mẫu, không mang ý nghĩa về độ khó.
Tên file output có hậu tố `_t{ngưỡng}` để so hai cách chia cạnh nhau.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from run import ACCURACY_PATH, MODEL_NAME  # noqa: E402
from windows import CONTEXT_MODES, WINDOW_SIZES  # noqa: E402

EXP3_ACCURACY_PATH = (
    SRC_DIR.parent.parent
    / "3.trace_length_performance_cliff"
    / "results"
    / "tables"
    / "accuracy.xlsx"
)
FIGURES_DIR = SRC_DIR.parent / "results" / "figures"
DATASETS = ["who_and_when__hand-crafted", "trace_elephant"]
MIN_STEPS_LONG = 30
MODE_COLORS = {"both": "tab:blue", "prev_only": "tab:orange", "next_only": "tab:green"}
X_LABELS = [f"fixed {w} steps" for w in WINDOW_SIZES]
BASELINE_STYLES = {"all_at_once": "--", "step_by_step": ":"}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df[
        (df["model"] == MODEL_NAME)
        & (df["dataset"].isin(DATASETS))
        & (df["status"] == "ok")
    ]


def _plot_overall_bar_ax(
    ax: plt.Axes, baselines: pd.DataFrame, windows: pd.DataFrame, subtitle: str
) -> float:
    baseline_means = baselines.groupby("method")["step_accuracy"].mean()
    means = windows.groupby(["window_size", "context_mode"])["step_accuracy"].mean()
    best_baseline = baseline_means.max()

    x = np.arange(len(WINDOW_SIZES))
    width = 0.8 / len(CONTEXT_MODES)
    y_max = best_baseline

    for i, mode in enumerate(CONTEXT_MODES):
        values = [means.get((w, mode.value), 0.0) for w in WINDOW_SIZES]
        offset = (i - (len(CONTEXT_MODES) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            color=MODE_COLORS[mode.value],
            label=mode.value,
            zorder=2,
        )
        y_max = max(y_max, *values)

    for method, value in baseline_means.items():
        ax.axhline(
            value,
            color="black",
            linewidth=1.2,
            linestyle=BASELINE_STYLES.get(method, "--"),
            zorder=3,
            label=f"{method} ({value:.3f})",
        )

    ax.set_ylabel("step_accuracy (mean)")
    ax.set_title(subtitle)
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, rotation=90)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper right")

    return y_max


def main() -> None:
    windows = _load(ACCURACY_PATH)
    baselines = _load(EXP3_ACCURACY_PATH)

    fig, (ax_short, ax_long) = plt.subplots(
        1, 2, figsize=(16, 6), sharey=True, constrained_layout=True
    )

    y_max_short = _plot_overall_bar_ax(
        ax_short,
        baselines[baselines["num_steps"] <= MIN_STEPS_LONG],
        windows[windows["num_steps"] <= MIN_STEPS_LONG],
        f"num_steps <= {MIN_STEPS_LONG}",
    )
    y_max_long = _plot_overall_bar_ax(
        ax_long,
        baselines[baselines["num_steps"] > MIN_STEPS_LONG],
        windows[windows["num_steps"] > MIN_STEPS_LONG],
        f"num_steps > {MIN_STEPS_LONG}",
    )

    y_max = max(y_max_short, y_max_long) * 1.15
    ax_short.set_ylim(0, y_max)
    ax_long.set_ylim(0, y_max)

    fig.suptitle("Overall step_accuracy: short vs long trajectories")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"pooled_overall_accuracy_by_length_t{MIN_STEPS_LONG}.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
