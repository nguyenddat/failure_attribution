"""Lỗi sai tập trung ở dataset nào — exp 3 + 4 + 5.

Phân rã mỗi row dự đoán thành 4 nhóm loại trừ nhau:
- correct: đúng cả agent lẫn step index
- agent_ok_step_wrong: đúng agent nhưng lệch step (lỗi "định vị")
- agent_wrong: sai agent (lỗi "quy trách nhiệm")
- not_found: quét hết trajectory mà không báo lỗi (pred_step == -1)

Panel trái: tỉ lệ 4 nhóm theo dataset × experiment (telbench chỉ có ở exp 3).
Panel phải: agent_accuracy / step_accuracy theo gt_agent (exp 4 + 5), để xem
lỗi agent_wrong đến từ những agent nào. Tên agent chuẩn hoá giống lúc chấm
điểm (methods.py::_normalize_agent_name) nên WebSurfer / Websurfer gộp chung.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from plot_pooled_all import ACCURACY_PATHS, FIGURES_DIR, MODEL_NAME  # noqa: E402

NOT_FOUND_STEP = -1
MIN_ROWS_PER_AGENT = 50

CLASSES = [
    ("correct", "tab:green"),
    ("agent_ok_step_wrong", "tab:orange"),
    ("agent_wrong", "tab:red"),
    ("not_found", "lightgrey"),
]
EXP_LABELS = {"baselines": "exp 3", "step": "exp 4", "all_at_once": "exp 5"}


def _load_all() -> pd.DataFrame:
    """Mọi dataset (kể cả telbench), mọi method — không lọc như plot_pooled_all._load."""
    frames = []
    for key, exp_dir in ACCURACY_PATHS.items():
        df = pd.read_excel(exp_dir / "results" / "tables" / "accuracy.xlsx")
        df = df[(df["model"] == MODEL_NAME) & (df["status"] == "ok")].copy()
        df["exp"] = EXP_LABELS[key]
        frames.append(
            df[
                [
                    "exp",
                    "dataset",
                    "method",
                    "file",
                    "gt_agent",
                    "agent_accuracy",
                    "step_accuracy",
                    "pred_step",
                    "num_steps",
                ]
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    df["error_class"] = np.select(
        [
            df["pred_step"] == NOT_FOUND_STEP,
            df["step_accuracy"] == 1,
            df["agent_accuracy"] == 1,
        ],
        ["not_found", "correct", "agent_ok_step_wrong"],
        default="agent_wrong",
    )
    return df


def _normalize_agent_name(name: str) -> str:
    return str(name).strip().lower().split(" (")[0]


def _plot_decomposition(ax: plt.Axes, df: pd.DataFrame) -> None:
    share = pd.crosstab([df["dataset"], df["exp"]], df["error_class"], normalize="index")
    counts = df.groupby(["dataset", "exp"]).size()
    share = share.reindex(counts.index)

    x = np.arange(len(share))
    bottom = np.zeros(len(share))
    for name, color in CLASSES:
        values = share.get(name, pd.Series(0.0, index=share.index)).to_numpy()
        ax.bar(x, values, 0.7, bottom=bottom, color=color, label=name, zorder=2)
        for xi, (value, base) in enumerate(zip(values, bottom)):
            if value >= 0.05:
                ax.text(
                    xi,
                    base + value / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{ds}\n{exp} (n={counts[(ds, exp)]})" for ds, exp in share.index],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("tỉ lệ row")
    ax.set_ylim(0, 1.18)
    ax.set_title("Phân rã lỗi theo dataset × experiment")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper center", ncol=4, fontsize=8, framealpha=0.9)


def _plot_per_agent(ax: plt.Axes, df: pd.DataFrame) -> None:
    windows = df[df["exp"].isin(["exp 4", "exp 5"])].copy()
    windows["gt_agent_norm"] = windows["gt_agent"].map(_normalize_agent_name)
    stats = windows.groupby(["dataset", "gt_agent_norm"]).agg(
        n=("file", "size"),
        agent_accuracy=("agent_accuracy", "mean"),
        step_accuracy=("step_accuracy", "mean"),
    )
    stats = stats[stats["n"] >= MIN_ROWS_PER_AGENT].sort_values("n", ascending=True)

    y = np.arange(len(stats))
    ax.barh(y, stats["agent_accuracy"], 0.6, color="tab:blue", label="agent_accuracy", zorder=2)
    ax.barh(y, stats["step_accuracy"], 0.3, color="tab:purple", label="step_accuracy", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{agent} — {ds.split('__')[0][:12]} (n={n})" for (ds, agent), n in zip(stats.index, stats["n"])],
        fontsize=8,
    )
    ax.set_xlabel("mean accuracy")
    ax.set_title(f"Theo gt_agent (exp 4 + 5, ≥{MIN_ROWS_PER_AGENT} row)")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.legend(loc="lower right", fontsize=8)


def main() -> None:
    df = _load_all()

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, 7), width_ratios=[1, 1.1], constrained_layout=True
    )
    _plot_decomposition(ax_left, df)
    _plot_per_agent(ax_right, df)
    fig.suptitle(f"Lỗi sai tập trung ở dataset nào — experiment 3 + 4 + 5 ({MODEL_NAME})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "error_decomposition_by_dataset.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    print("\nphân rã theo dataset (mọi exp gộp):")
    print(
        pd.crosstab(df["dataset"], df["error_class"], normalize="index")[
            [name for name, _ in CLASSES]
        ]
        .round(3)
        .to_string()
    )
    print("\nsố row lỗi tuyệt đối theo dataset:")
    print(pd.crosstab(df["dataset"], df["error_class"])[[name for name, _ in CLASSES]].to_string())


if __name__ == "__main__":
    main()
