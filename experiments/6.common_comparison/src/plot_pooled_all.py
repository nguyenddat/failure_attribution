"""Overall accuracy (step level + agent level): short vs long trajectories, exp 4 + 5.

Cùng dạng với experiment 4's plot_pooled.py nhưng gộp cả 2 họ fixed-window:
- exp 4: chấm điểm từng step, context = cửa sổ quanh step (both/prev_only/next_only)
- exp 5: chấm điểm cả cửa sổ một lượt (all_at_once)
Baseline all_at_once / step_by_step lấy từ experiment 3 vẽ thành đường ngang.

who_and_when__hand-crafted + trace_elephant gộp chung (277 file, giống nhau ở
cả 3 experiment); ngưỡng 30 steps là breaking point đo được ở experiment 3 (chỗ
baseline all_at_once tụt xuống mức random guess), thay cho median 22 dùng trước
đó — median chỉ chia đôi tập mẫu, không mang ý nghĩa về độ khó.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

ACCURACY_PATHS = {
    "baselines": EXPERIMENTS_DIR / "3.trace_length_performance_cliff",
    "step": EXPERIMENTS_DIR / "4.fixed_window_segmentation",
    "all_at_once": EXPERIMENTS_DIR / "5.fixed_window_segmentation_all_at_once",
}

MODEL_NAME = "gpt-4o-mini"
DATASETS = ["who_and_when__hand-crafted", "trace_elephant"]
MIN_STEPS_LONG = 30
WINDOW_SIZES = [5, 7, 9, 11]

# (label, color, method name template)
SERIES = [
    ("per-step, both", "tab:blue", "fixed_window_w{w}_both"),
    ("per-step, prev_only", "tab:orange", "fixed_window_w{w}_prev_only"),
    ("per-step, next_only", "tab:green", "fixed_window_w{w}_next_only"),
    ("window all_at_once", "tab:purple", "fixed_window_all_at_once_w{w}"),
]
X_LABELS = [f"fixed {w} steps" for w in WINDOW_SIZES]
BASELINE_STYLES = {"all_at_once": "--", "step_by_step": ":"}

# (cột metric, phần tên file) — step level đòi đúng cả agent lẫn step index, agent
# level chỉ đòi đúng agent nên luôn cao hơn: chênh lệch là phần lỗi do lệch step.
METRICS = [
    ("step_accuracy", "accuracy"),
    ("agent_accuracy", "agent_accuracy"),
]


def _load(exp_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(exp_dir / "results" / "tables" / "accuracy.xlsx")
    return df[
        (df["model"] == MODEL_NAME)
        & (df["dataset"].isin(DATASETS))
        & (df["status"] == "ok")
    ]


def _plot_ax(
    ax: plt.Axes,
    baselines: pd.DataFrame,
    windows: pd.DataFrame,
    subtitle: str,
    value_col: str = "step_accuracy",
) -> float:
    baseline_means = baselines.groupby("method")[value_col].mean()
    means = windows.groupby("method")[value_col].mean()
    y_max = baseline_means.max()

    x = np.arange(len(WINDOW_SIZES))
    width = 0.8 / len(SERIES)

    for i, (label, color, template) in enumerate(SERIES):
        values = [means.get(template.format(w=w), 0.0) for w in WINDOW_SIZES]
        offset = (i - (len(SERIES) - 1) / 2) * width
        ax.bar(x + offset, values, width, color=color, label=label, zorder=2)
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

    ax.set_ylabel(f"{value_col} (mean)")
    n_files = len(windows.drop_duplicates(["dataset", "file"]))
    ax.set_title(f"{subtitle} (n={n_files} file)")
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, rotation=90)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=8)

    return y_max


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
        fig, (ax_short, ax_long) = plt.subplots(
            1, 2, figsize=(16, 6), sharey=True, constrained_layout=True
        )

        y_max_short = _plot_ax(
            ax_short,
            baselines[baselines["num_steps"] <= MIN_STEPS_LONG],
            windows[windows["num_steps"] <= MIN_STEPS_LONG],
            f"num_steps <= {MIN_STEPS_LONG}",
            value_col,
        )
        y_max_long = _plot_ax(
            ax_long,
            baselines[baselines["num_steps"] > MIN_STEPS_LONG],
            windows[windows["num_steps"] > MIN_STEPS_LONG],
            f"num_steps > {MIN_STEPS_LONG}",
            value_col,
        )

        y_max = max(y_max_short, y_max_long) * 1.15
        ax_short.set_ylim(0, y_max)
        ax_long.set_ylim(0, y_max)

        fig.suptitle(
            f"Overall {value_col}: short vs long trajectories — experiment 4 + 5 "
            f"({' + '.join(DATASETS)} pooled)"
        )

        out_path = FIGURES_DIR / f"pooled_overall_{name_part}_by_length_all_methods.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
