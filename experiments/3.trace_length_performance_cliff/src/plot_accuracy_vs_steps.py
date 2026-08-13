"""Line plot: step_accuracy vs exact num_steps, gpt-4o-mini only, all 3 datasets pooled.

x-axis is the exact per-file step count (no binning) — each point on the line
is the mean step_accuracy of every file (across all 3 datasets combined) that
has that exact num_steps, connected in ascending step order. Pooling (instead
of one line per dataset) is to find the overall performance-drop point.
deepseek-v4-flash is excluded (not yet run).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from run import ACCURACY_PATH  # noqa: E402

FIGURES_DIR = SRC_DIR.parent / "results" / "figures"
MODEL = "gpt-4o-mini"


def main() -> None:
    df = pd.read_excel(ACCURACY_PATH)
    df = df[(df["model"] == MODEL) & (df["status"] == "ok")]

    by_step = df.groupby("num_steps")["step_accuracy"].agg(["mean", "count"]).sort_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax_n = ax.twinx()
    ax_n.bar(by_step.index, by_step["count"], width=0.8, color="lightgray", zorder=1)
    ax_n.set_ylabel("sample count at this num_steps", color="gray")
    ax_n.tick_params(axis="y", colors="gray")

    ax.plot(
        by_step.index,
        by_step["mean"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        color="tab:blue",
        label=f"all datasets pooled (n={len(df)})",
        zorder=2,
    )
    ax.set_zorder(ax_n.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_xlabel("num_steps (exact trace length)")
    ax.set_ylabel("step_accuracy (mean per exact step count)")
    ax.set_title(f"Step accuracy vs trace length — {MODEL}, all datasets pooled")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "step_accuracy_vs_num_steps.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
