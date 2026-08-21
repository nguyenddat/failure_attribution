"""Tỉ lệ quét hết trajectory mà không báo lỗi — tất cả phương pháp exp 3 + 4 + 5.

Cùng dạng với experiment 4's plot_step_offset.py::_plot_not_found: mỗi phương
pháp một pie chart "Not Found" (pred_step == -1) vs "có dự đoán". Ở đây gộp
4 window size của exp 4 (3 context mode) + exp 5 (all_at_once), kèm 2 baseline
all_at_once / step_by_step của exp 3 để so sánh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from plot_pooled_all import ACCURACY_PATHS, DATASETS, FIGURES_DIR, _load  # noqa: E402

NOT_FOUND_STEP = -1

# (label, color, experiment key, method prefix hoặc tên đầy đủ)
PIES = [
    ("baseline all_at_once", "tab:gray", "baselines", "all_at_once"),
    ("baseline step_by_step", "tab:red", "baselines", "step_by_step"),
    ("per-step, both", "tab:blue", "step", "both"),
    ("per-step, prev_only", "tab:orange", "step", "prev_only"),
    ("per-step, next_only", "tab:green", "step", "next_only"),
    ("window all_at_once", "tab:purple", "all_at_once", None),
]


def _subset(frames: dict[str, pd.DataFrame], exp_key: str, selector: str | None) -> pd.DataFrame:
    df = frames[exp_key]
    if selector is None:
        return df
    if exp_key == "baselines":
        return df[df["method"] == selector]
    return df[df["context_mode"] == selector]


def main() -> None:
    frames = {key: _load(path) for key, path in ACCURACY_PATHS.items()}

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))

    for ax, (label, color, exp_key, selector) in zip(axes.ravel(), PIES):
        subset = _subset(frames, exp_key, selector)
        not_found = int((subset["pred_step"] == NOT_FOUND_STEP).sum())
        predicted = len(subset) - not_found
        ax.pie(
            [not_found, predicted],
            labels=["Not Found", "có dự đoán"],
            colors=["lightgrey", color],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        ax.set_title(f"{label}\n{not_found}/{len(subset)} row Not Found")

    fig.suptitle(
        "Tỉ lệ quét hết trajectory mà không báo lỗi — experiment 3 + 4 + 5 "
        f"({' + '.join(DATASETS)} pooled, window method gộp 4 window size)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "pooled_not_found_rate_all_methods.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
