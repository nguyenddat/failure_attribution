from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.error_categorization.mast import json_dir

FAULT_DETECTION_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LENGTH_SPLIT_PATH = json_dir / "length_split.json"

# Methods run on the FULL mast dataset (mixed short + long) -> filtered live by
# file id membership against length_split.json short_ids/long_ids.
FULL_DATASET_SOURCES = {
    "baseline": FAULT_DETECTION_DIR / "baseline" / "output" / "all_at_once.csv",
}

# fixed_size_segment (token-ratio) was also run on the full dataset; already
# pre-split on disk by fixed_size_segment/split_by_length.py.
FIXED_SIZE_LEVELS = ["25pct", "50pct", "70pct", "100pct"]
FIXED_SIZE_SPLIT_DIR = FAULT_DETECTION_DIR / "fixed_size_segment" / "output"

# Methods run ONLY on the long subset (step_count >= threshold) -> contribute
# to the long row only, nothing to plot for short.
LONG_ONLY_SOURCES = {
    "overlap_25%": FAULT_DETECTION_DIR / "overlapping_segment" / "output" / "overlapping_segment_25pct.csv",
    "overlap_50%": FAULT_DETECTION_DIR / "overlapping_segment" / "output" / "overlapping_segment_50pct.csv",
    "overlap_70%": FAULT_DETECTION_DIR / "overlapping_segment" / "output" / "overlapping_segment_70pct.csv",
    "overlap_100%": FAULT_DETECTION_DIR / "overlapping_segment" / "output" / "overlapping_segment_100pct.csv",
    "step_5": FAULT_DETECTION_DIR / "fixed_step_segment" / "output" / "fixed_step_segment_5step.csv",
    "step_7": FAULT_DETECTION_DIR / "fixed_step_segment" / "output" / "fixed_step_segment_7step.csv",
    "step_9": FAULT_DETECTION_DIR / "fixed_step_segment" / "output" / "fixed_step_segment_9step.csv",
    "step_11": FAULT_DETECTION_DIR / "fixed_step_segment" / "output" / "fixed_step_segment_11step.csv",
}

# Fixed order -> fixed color per method (identity, never re-assigned by which
# methods happen to have data in a given row).
METHOD_ORDER = (
    ["baseline"]
    + [f"fixed_{level}" for level in FIXED_SIZE_LEVELS]
    + list(LONG_ONLY_SOURCES.keys())
)

METRIC_GROUPS = {
    "accuracy_metrics": {
        "title": "Multi-label accuracy metrics (mean across common files)",
        "metrics": ["exact_match_ratio", "hamming_loss", "precision", "recall", "f1"],
        "lower_is_better": {"hamming_loss"},
        "log_scale": False,
    },
    "group_metrics": {
        "title": "Group-tolerant accuracy metrics (mean across common files)",
        "metrics": ["group_precision", "group_recall", "group_f1"],
        "lower_is_better": set(),
        "log_scale": False,
    },
    "cost_metrics": {
        "title": "Cost metrics (mean across common files)",
        "metrics": ["input_tokens", "output_tokens", "latency"],
        "lower_is_better": {"input_tokens", "output_tokens", "latency"},
        # tokens (1e2-1e4) and latency (seconds) differ by orders of magnitude;
        # log scale keeps bars readable within a single metric's own panel too.
        "log_scale": True,
    },
}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_LINE = "#52514e"
SURFACE = "#fcfcfb"
IMPROVE_COLOR = "#1baf7a"
REGRESS_COLOR = "#eb6834"
BEST_COLOR = "#2a78d6"


def load_length_ids() -> tuple[set[int], set[int]]:
    manifest = json.loads(LENGTH_SPLIT_PATH.read_text(encoding="utf-8"))
    return set(manifest["short_ids"]), set(manifest["long_ids"])


def file_ids(df: pd.DataFrame) -> pd.Series:
    return df["file_name"].map(lambda name: int(Path(name).stem))


def read_clean(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["error"].isna()]
    return df if not df.empty else None


def discover_result_frames(short_ids: set[int], long_ids: set[int]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    short_frames: dict[str, pd.DataFrame] = {}
    long_frames: dict[str, pd.DataFrame] = {}

    for label, path in FULL_DATASET_SOURCES.items():
        df = read_clean(path)
        if df is None:
            continue
        ids = file_ids(df)
        short_df = df[ids.isin(short_ids)]
        long_df = df[ids.isin(long_ids)]
        if not short_df.empty:
            short_frames[label] = short_df
        if not long_df.empty:
            long_frames[label] = long_df

    for level in FIXED_SIZE_LEVELS:
        label = f"fixed_{level}"
        short_df = read_clean(FIXED_SIZE_SPLIT_DIR / "short" / f"fixed_size_segment_{level}.csv")
        long_df = read_clean(FIXED_SIZE_SPLIT_DIR / "long" / f"fixed_size_segment_{level}.csv")
        if short_df is not None:
            short_frames[label] = short_df
        if long_df is not None:
            long_frames[label] = long_df

    for label, path in LONG_ONLY_SOURCES.items():
        df = read_clean(path)
        if df is None:
            continue
        long_frames[label] = df

    return short_frames, long_frames


def compute_group_means(frames: dict[str, pd.DataFrame], metrics: list[str]) -> tuple[pd.DataFrame, int]:
    if not frames:
        return pd.DataFrame(), 0
    common_files = set.intersection(*(set(df["file_name"]) for df in frames.values()))
    rows = {label: df[df["file_name"].isin(common_files)][metrics].mean() for label, df in frames.items()}
    return pd.DataFrame(rows).T, len(common_files)


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", which="major", color=GRIDLINE, linestyle="-", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)


def draw_floating_panel(
    ax,
    means_df: pd.DataFrame,
    metric: str,
    metric_lower_is_better: bool,
    dataset_label: str,
    n: int,
    log_scale: bool,
):
    style_axis(ax)
    ax.set_title(f"{dataset_label} (n={n})", fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY, loc="left")

    if means_df.empty or "baseline" not in means_df.index:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=TEXT_MUTED, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    baseline_value = float(means_df.loc["baseline", metric])
    labels = [m for m in METHOD_ORDER if m in means_df.index and m != "baseline"]
    if not labels:
        ax.text(0.5, 0.5, "no non-baseline method", ha="center", va="center", color=TEXT_MUTED, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    values = means_df.loc[labels, metric].astype(float)
    best_label = values.idxmin() if metric_lower_is_better else values.idxmax()

    x = np.arange(len(labels))
    bottoms = np.minimum(values.to_numpy(), baseline_value)
    heights = np.abs(values.to_numpy() - baseline_value)

    bar_colors, edge_widths = [], []
    for label, value in zip(labels, values):
        improved = value < baseline_value if metric_lower_is_better else value > baseline_value
        if label == best_label:
            bar_colors.append(BEST_COLOR)
            edge_widths.append(2.2)
        elif improved:
            bar_colors.append(IMPROVE_COLOR)
            edge_widths.append(0.8)
        else:
            bar_colors.append(REGRESS_COLOR)
            edge_widths.append(0.8)

    bars = ax.bar(
        x, heights, bottom=bottoms, width=0.55, color=bar_colors, alpha=0.85,
        edgecolor=[c for c in bar_colors], linewidth=edge_widths, zorder=3,
    )

    ax.axhline(baseline_value, color=BASELINE_LINE, linestyle="--", linewidth=1.4, zorder=2)
    ax.annotate(
        f"baseline={baseline_value:.3g}", xy=(1.0, baseline_value), xycoords=("axes fraction", "data"),
        xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=7.5, color=TEXT_SECONDARY,
    )

    for xi, value, label in zip(x, values, labels):
        va = "bottom" if value >= baseline_value else "top"
        offset = 4 if value >= baseline_value else -4
        text = f"{value:.3g}"
        if label == best_label:
            text += "  ★ best"
        ax.annotate(
            text, xy=(xi, value), xytext=(0, offset), textcoords="offset points",
            ha="center", va=va, fontsize=7.5,
            color=BEST_COLOR if label == best_label else TEXT_SECONDARY,
            fontweight="bold" if label == best_label else "normal",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5, color=TEXT_PRIMARY)
    if log_scale:
        ax.set_yscale("log")


def plot_metric(
    metric: str,
    metric_lower_is_better: bool,
    log_scale: bool,
    short_means: pd.DataFrame,
    n_short: int,
    long_means: pd.DataFrame,
    n_long: int,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=SURFACE)

    draw_floating_panel(axes[0], short_means, metric, metric_lower_is_better, "SHORT", n_short, log_scale)
    draw_floating_panel(axes[1], long_means, metric, metric_lower_is_better, "LONG", n_long, log_scale)

    direction = "lower is better" if metric_lower_is_better else "higher is better"
    fig.suptitle(f"{metric} ({direction}) vs. baseline", fontsize=12.5, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{metric}.png"
    fig.savefig(output_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    short_ids, long_ids = load_length_ids()
    short_frames, long_frames = discover_result_frames(short_ids, long_ids)

    if not short_frames and not long_frames:
        print("No result CSVs found.")
        return

    print(f"Short row methods: {list(short_frames.keys())}")
    print(f"Long row methods: {list(long_frames.keys())}")

    for group_name, group_spec in METRIC_GROUPS.items():
        metrics = group_spec["metrics"]
        short_means, n_short = compute_group_means(short_frames, metrics)
        long_means, n_long = compute_group_means(long_frames, metrics)

        if short_means.empty and long_means.empty:
            print(f"Skipped {group_name}: no data")
            continue

        for metric in metrics:
            output_path = plot_metric(
                metric,
                metric in group_spec["lower_is_better"],
                group_spec["log_scale"],
                short_means, n_short,
                long_means, n_long,
            )
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
