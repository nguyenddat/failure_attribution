"""Block 1 (twin-comparison distance + Dt), ported from
experiments/fault_detection/twin_comparison_segment to the single_fault
who&when hand-crafted dataset.

Why switch datasets: MAST trajectories only carry a trace-level fault label
(no ground-truth step), so block 1's high-D cluster could only be checked
against proxy signals (phase tag, speaker). who&when hand-crafted gives an
explicit `mistake_step`/`mistake_agent` per sample - here we can check
directly whether the true failure boundary actually falls in a high-D pair,
instead of only proxy hypotheses.

No LLM call - distance + threshold + ground-truth alignment only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from experiments.fault_detection.twin_comparison_segment.distance import (
    adjacent_distances,
    detect_dt,
)
from experiments.single_fault.utils.datasets import DATASET_DIRS
from experiments.single_fault.utils.file import load_json

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATASET_DIR = DATASET_DIRS["ww_hand_crafted"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
HIST_COLOR = "#2a78d6"
DT_COLOR = "#eb6834"
MISTAKE_COLOR = "#c0392b"


def pick_sample_ids(min_steps: int = 15, limit: int = 5) -> list[int]:
    picked = []
    for path in sorted(DATASET_DIR.glob("*.json"), key=lambda p: int(p.stem)):
        data = load_json(str(path))
        if len(data["trajectory"]) >= min_steps:
            picked.append(int(path.stem))
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


def mistake_pair_diffs(distances: list[float], mistake_step: int) -> list[tuple[int, float]]:
    """Adjacent-pair diffs touching mistake_step: (mistake_step-1, mistake_step) and
    (mistake_step, mistake_step+1) - i.e. distances[mistake_step-1] and distances[mistake_step]."""
    hits = []
    for idx in (mistake_step - 1, mistake_step):
        if 0 <= idx < len(distances):
            hits.append((idx, distances[idx]))
    return hits


def plot_sample(sample_id: int, data: dict) -> Path:
    steps = [s["content"] for s in data["trajectory"]]
    mistake_step = data["mistake_step"]
    mistake_agent = data["mistake_agent"]

    distances = adjacent_distances(steps)
    dt, method = detect_dt(distances)
    hits = mistake_pair_diffs(distances, mistake_step)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.hist(distances, bins=30, color=HIST_COLOR, alpha=0.55, edgecolor=HIST_COLOR, zorder=3)
    ax.axvline(dt, color=DT_COLOR, linestyle="--", linewidth=1.6, label=f"Dt ({method})={dt:.3f}")

    for idx, d_value in hits:
        in_high = d_value >= dt
        ax.axvline(
            d_value, color=MISTAKE_COLOR, linestyle="-", linewidth=1.8, alpha=0.9,
            label=f"mistake pair ({idx},{idx+1}) D={d_value:.3f} [{'HIGH' if in_high else 'low'}]",
        )

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True,
        facecolor=SURFACE, edgecolor=BASELINE, fontsize=7.5, labelcolor=TEXT_SECONDARY,
    )
    style_axis(ax)
    ax.set_title(
        f"sample {sample_id} ({len(steps)} steps) - mistake_step={mistake_step} ({mistake_agent})",
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
        print("no hand-crafted sample found")
        return

    hit_count = 0
    for sample_id in sample_ids:
        path = DATASET_DIR / f"{sample_id}.json"
        data = load_json(str(path))
        out_path = plot_sample(sample_id, data)

        distances = adjacent_distances([s["content"] for s in data["trajectory"]])
        dt, _ = detect_dt(distances)
        hits = mistake_pair_diffs(distances, data["mistake_step"])
        any_high = any(d >= dt for _, d in hits)
        hit_count += int(any_high)
        print(
            f"sample {sample_id}: saved {out_path} | mistake_step={data['mistake_step']} "
            f"({data['mistake_agent']}) -> {'IN high cluster' if any_high else 'in low cluster'}"
        )

    print(f"\nmistake pair landed in high-D cluster for {hit_count}/{len(sample_ids)} samples")


if __name__ == "__main__":
    main()
