"""Block 1 prototype: adjacent-pair distance histogram + Dt threshold,
run on 5 long MAST trajectories. No LLM call - this is only the distance /
threshold step, inspected on its own before building later blocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from data.error_categorization.mast import Sample, json_dir
from experiments.fault_detection.twin_comparison_segment.distance import (
    adjacent_distances,
    detect_dt,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LENGTH_SPLIT_PATH = json_dir / "length_split.json"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
HIST_COLOR = "#2a78d6"
DT_COLOR = "#eb6834"


def pick_sample_ids(min_steps: int = 15, limit: int = 5) -> list[int]:
    manifest = json.loads(LENGTH_SPLIT_PATH.read_text(encoding="utf-8"))
    long_ids = manifest["long_ids"]

    picked = []
    for sample_id in long_ids:
        path = json_dir / f"{sample_id}.json"
        if not path.exists():
            continue
        sample = Sample.model_validate_json(path.read_text(encoding="utf-8"))
        if sample.trajectory and len(sample.trajectory) >= min_steps:
            picked.append(sample_id)
        if len(picked) == limit:
            break
    return picked


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel("count", color=TEXT_SECONDARY, fontsize=10)
    ax.set_xlabel("D (adjacent-pair cosine distance)", color=TEXT_SECONDARY, fontsize=10)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", which="major", color=GRIDLINE, linestyle="-", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)


def plot_sample(sample_id: int, sample: Sample) -> Path:
    steps = [s.content for s in sample.trajectory]
    distances = adjacent_distances(steps)
    dt, method = detect_dt(distances)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.hist(distances, bins=30, color=HIST_COLOR, alpha=0.55, edgecolor=HIST_COLOR, zorder=3)
    ax.axvline(dt, color=DT_COLOR, linestyle="--", linewidth=1.6, label=f"Dt ({method})={dt:.3f}")
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True,
        facecolor=SURFACE, edgecolor=BASELINE, fontsize=8, labelcolor=TEXT_SECONDARY,
    )
    style_axis(ax)
    ax.set_title(
        f"sample {sample_id} ({sample.mas_name}, {len(steps)} steps) - Ds histogram",
        color=TEXT_PRIMARY, fontsize=10.5, loc="left",
    )

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{sample_id}_ds_histogram.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    sample_ids = pick_sample_ids()
    if not sample_ids:
        print("no long MAST sample with trajectory found")
        return

    for sample_id in sample_ids:
        path = json_dir / f"{sample_id}.json"
        sample = Sample.model_validate_json(path.read_text(encoding="utf-8"))
        out_path = plot_sample(sample_id, sample)
        print(f"sample {sample_id}: saved {out_path}")


if __name__ == "__main__":
    main()
