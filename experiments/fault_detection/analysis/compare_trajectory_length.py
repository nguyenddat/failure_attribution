"""Boxplot comparing trajectory step count and token length across
MAST, who&when algorithm-generated, and who&when hand-crafted datasets."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import tiktoken

BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODING = tiktoken.get_encoding("cl100k_base")

DATASETS = [
    ("MAST", BASE_DIR / "data" / "fault_detection" / "json" / "mast"),
    ("who&when\nalgo-generated", BASE_DIR / "data" / "single_fault" / "json" / "who_and_when__algorithm-generated"),
    ("who&when\nhand-crafted", BASE_DIR / "data" / "single_fault" / "json" / "who_and_when__hand-crafted"),
]

# First three slots of the validated categorical palette (references/palette.md)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def token_length(steps: list[dict]) -> int:
    text = "\n".join(s.get("content", "") for s in steps)
    return len(ENCODING.encode(text))


def collect(json_dir: Path) -> tuple[list[int], list[int]]:
    steps_out, tokens_out = [], []
    for file_path in sorted(json_dir.glob("*.json")):
        if not file_path.stem.isdigit():
            continue
        data = json.loads(file_path.read_text(encoding="utf-8"))
        traj = data.get("trajectory") or []
        if not traj:
            continue
        steps_out.append(len(traj))
        tokens_out.append(token_length(traj))
    return steps_out, tokens_out


def style_axis(ax, ylabel: str, log: bool = False):
    ax.set_facecolor(SURFACE)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", which="major", color=GRIDLINE, linestyle="-", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(axis="x", colors=TEXT_PRIMARY, labelsize=9.5)


def draw_boxplot(ax, data_lists: list[list[int]], labels: list[str], ylabel: str, log: bool = False):
    bp = ax.boxplot(
        data_lists,
        tick_labels=labels,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": TEXT_MUTED,
            "markeredgecolor": "none",
            "alpha": 0.4,
        },
        medianprops={"color": TEXT_PRIMARY, "linewidth": 1.8},
        whiskerprops={"color": TEXT_SECONDARY, "linewidth": 1.1},
        capprops={"color": TEXT_SECONDARY, "linewidth": 1.1},
        boxprops={"linewidth": 1.1},
        zorder=3,
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    style_axis(ax, ylabel, log=log)
    ax.set_ylim(top=ax.get_ylim()[1] * 6)

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, data in enumerate(data_lists, start=1):
        median = sorted(data)[len(data) // 2]
        ax.text(
            i,
            0.97,
            f"n={len(data)}\nmed={median}",
            transform=trans,
            ha="center",
            va="top",
            fontsize=8,
            color=TEXT_SECONDARY,
        )


def main() -> None:
    labels = [name for name, _ in DATASETS]
    step_lists, token_lists = [], []
    for _, path in DATASETS:
        steps, tokens = collect(path)
        step_lists.append(steps)
        token_lists.append(tokens)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), facecolor=SURFACE)

    draw_boxplot(axes[0], step_lists, labels, "Steps per trajectory", log=True)
    axes[0].set_title("Step count", color=TEXT_PRIMARY, fontsize=11, loc="left", pad=18)

    draw_boxplot(axes[1], token_lists, labels, "Tokens per trajectory (cl100k_base, log scale)", log=True)
    axes[1].set_title("Token length", color=TEXT_PRIMARY, fontsize=11, loc="left", pad=18)

    fig.suptitle(
        "Trajectory length: MAST vs. who&when (steps vs. tokens)",
        color=TEXT_PRIMARY,
        fontsize=13,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = OUTPUT_DIR / "trajectory_length_comparison.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
