"""Grid of error-step-position distributions: topology (row) x dataset (column).

Reads results/tables/step_positions.csv (extract_step_positions.py) and
groups rows by coordination topology (framework_architecture.py, sourced
from experiments/0.framework_topology_taxonomy/finding_notes.md) instead
of exact framework name. Grouping by name alone only finds one real
cross-dataset overlap (Magentic-One, in TraceElephant + Who&When) and
drops AgentErrorBench/TELBENCH/TRAIL entirely; grouping by topology *type*
lets every dataset contribute — Hierarchical = OpenDeepResearch
(TRAIL/GAIA) alone; Centralized = Captain-Agent + Magentic-One
(TraceElephant/Who&When) + MiroFlow/OAgent (TELBENCH); Single-agent =
SWE-Agent (TraceElephant) + CodeAct (TRAIL) + AgentDebug-ReAct
(AgentErrorBench). Decentralized is empty (see framework_architecture.py
docstring).

Fixed-margin layout (not tight_layout): tight_layout doesn't account for
text placed via annotate()/text() at axes-fraction coordinates, and its
auto-computed spacing collided with a multi-line suptitle in earlier
iterations. Y is **density** (area under each line = 1), not raw count,
shared 0..global-max-density. Cell sample sizes range from a few dozen to
several thousand — sharing a raw-count axis would dwarf every other cell
to the point of making them unreadable. Density keeps every cell's *shape*
comparable regardless of n; the actual n is still printed in each
subplot's title so sample size isn't hidden, just not encoded in line
height anymore.

Usage: python experiments/2.error_step_position_by_framework/src/plot_step_position_grid_by_architecture.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from framework_architecture import ARCH_ORDER, ARCHITECTURE

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
csv_path = RESULTS_DIR / "tables" / "step_positions.csv"
fig_path = RESULTS_DIR / "figures" / "step_position_grid_architecture_x_dataset.png"

N_BINS = 12
AGGREGATE_COL = "Tổng hợp"


def line_hist(ax, positions: pd.Series, color: str, ymax: float) -> None:
    counts, edges = np.histogram(positions, bins=N_BINS, range=(0, 1), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, counts, color=color, linewidth=1.5)
    ax.fill_between(centers, counts, color=color, alpha=0.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)


def main() -> None:
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found — run extract_step_positions.py first.")
    df = pd.read_csv(csv_path)
    df["architecture"] = df["framework"].map(ARCHITECTURE)
    unmapped = df[df["architecture"].isna()]["framework"].unique()
    if len(unmapped):
        raise SystemExit(f"Unmapped frameworks, add to framework_architecture.py: {unmapped}")

    datasets = sorted(df["dataset"].unique())
    columns = datasets + [AGGREGATE_COL]
    architectures = [a for a in ARCH_ORDER if (df["architecture"] == a).any()]

    global_ymax = 0
    for arch in architectures:
        for col in columns:
            sub = df[df["architecture"] == arch] if col == AGGREGATE_COL else df[(df["architecture"] == arch) & (df["dataset"] == col)]
            if sub.empty:
                continue
            counts, _ = np.histogram(sub["position"], bins=N_BINS, range=(0, 1), density=True)
            global_ymax = max(global_ymax, counts.max())
    global_ymax *= 1.05

    cmap = plt.get_cmap("Set2")
    arch_color = {a: cmap(i / max(len(architectures) - 1, 1)) for i, a in enumerate(architectures)}

    n_rows, n_cols = len(architectures), len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 1.7), squeeze=False)

    LEFT, RIGHT, TOP, BOTTOM = 0.15, 0.98, 0.72, 0.10
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, hspace=0.55, wspace=0.12)

    for r, arch in enumerate(architectures):
        arch_rows = df[df["architecture"] == arch]
        for c, col in enumerate(columns):
            ax = axes[r][c]
            sub = arch_rows if col == AGGREGATE_COL else arch_rows[arch_rows["dataset"] == col]
            if sub.empty:
                ax.axis("off")
                continue
            line_hist(ax, sub["position"], arch_color[arch], global_ymax)
            n_fw = sub["framework"].nunique()
            ax.set_title(f"n={len(sub)} ({n_fw} fw)", fontsize=7, pad=2)
            ax.tick_params(labelsize=6)
            if r == n_rows - 1:
                ax.set_xlabel("position", fontsize=7)
            else:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])

    for r in range(n_rows):
        pos = axes[r][0].get_position()
        fig.text(LEFT - 0.015, (pos.y0 + pos.y1) / 2, architectures[r], fontsize=8, ha="right", va="center")

    for c, col in enumerate(columns):
        pos = axes[0][c].get_position()
        fig.text((pos.x0 + pos.x1) / 2, TOP + 0.03, col, fontsize=8, fontweight="bold", ha="center", va="bottom")

    fig.suptitle(
        "Error-step position distribution by coordination topology x dataset\n"
        f"x = normalized position (0=first, 1=last), y = density (area=1, shared 0-{global_ymax:.2f}). "
        "Decentralized empty (no such system in these 5 datasets).",
        fontsize=9.5,
        y=0.99,
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(df.groupby(["architecture", "dataset"]).size().to_string())
    print(f"\nSaved -> {fig_path}")


if __name__ == "__main__":
    main()
