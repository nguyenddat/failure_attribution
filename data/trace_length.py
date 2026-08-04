"""Trace length distribution across every dataset in the repo.

Length is reported in two units:

* **token** - what a model actually pays for. The whole trace is concatenated
  into one string and encoded with ``o200k_base``, the ``gpt-4o-mini``
  tokenizer.
* **step** - how many trajectory entries the trace has.

The token plot answers one question: how many traces do not fit into the
context window of the models we run (:data:`CONTEXT_LIMITS`)?

Every dataset is drawn on the same pair of axes, so the two output files are the
whole picture. This module knows nothing about any dataset schema: each dataset
ships a thin adapter next to its loader that declares a :class:`Dataset` with
two pure functions over one decoded document. ``data/trace_length_all.py``
collects those adapters and draws the figures.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from scipy.stats import gaussian_kde

ENCODING = tiktoken.get_encoding("o200k_base")

# Context window per model we run, in tokens.
CONTEXT_LIMITS = {
    "gpt-4o-mini": 128_000,
    "deepseek-chat": 64_000,
}

LIMIT_COLORS = {
    "gpt-4o-mini": "crimson",
    "deepseek-chat": "teal",
}

# tab10 without its red and cyan, which would read as the context-limit lines.
PALETTE = [
    color
    for index, color in enumerate(plt.get_cmap("tab10").colors)
    if index not in (3, 9)
]


def count_tokens(parts: Iterable[str]) -> int:
    """Concatenate the full content of one trace and count its tokens.

    Joining before encoding (rather than summing per-part counts) matches how
    the text reaches the model.
    """
    return len(ENCODING.encode("\n".join(part for part in parts if part)))


@dataclass
class TraceLength:
    name: str
    # None when the dataset carries no step structure.
    n_steps: Optional[int]
    n_tokens: int


Contents = Callable[[dict], List[str]]
Steps = Callable[[dict], Optional[int]]
# Dataset name -> line colour, shared by both figures.
Colors = Dict[str, tuple]


@dataclass
class Dataset:
    """What one adapter declares about its dataset."""

    name: str
    json_dir: Path
    contents: Contents
    steps: Steps


def collect(dataset: Dataset) -> List[TraceLength]:
    files = [
        path
        for path in sorted(dataset.json_dir.glob("*.json"))
        if path.name != "metadata.json"
    ]
    if not files:
        raise FileNotFoundError(
            f"no JSON files in {dataset.json_dir}; "
            "generate them with the matching `make load_*` target"
        )

    lengths = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            lengths.append(
                TraceLength(
                    name=path.stem,
                    n_steps=dataset.steps(data),
                    n_tokens=count_tokens(dataset.contents(data)),
                )
            )
        except KeyError as error:
            raise KeyError(f"{path}: missing key {error}") from error
    return lengths


def _describe(values: List[int]) -> str:
    array = np.asarray(values)
    return (
        f"min={array.min():,} median={int(np.median(array)):,} "
        f"mean={array.mean():,.1f} p95={int(np.percentile(array, 95)):,} max={array.max():,}"
    )


def summarize(name: str, lengths: List[TraceLength]) -> None:
    tokens = [length.n_tokens for length in lengths]
    steps = [length.n_steps for length in lengths if length.n_steps is not None]

    print(f"{name}  n={len(lengths):,}")
    print(f"  tokens  {_describe(tokens)}")
    print(f"  steps   {_describe(steps) if steps else 'unavailable'}")

    for model, limit in CONTEXT_LIMITS.items():
        over = sum(1 for value in tokens if value > limit)
        print(f"  over {model} ({limit:,}): {over:,} ({over / len(tokens):.1%})")


def _human(value: float) -> str:
    """Format a count for an axis tick: 1.0K, 10K, 100K, 1.0M."""
    for scale, suffix in ((1e6, "M"), (1e3, "K")):
        if value >= scale:
            scaled = value / scale
            return f"{scaled:.1f}{suffix}" if scaled < 10 else f"{scaled:.0f}{suffix}"
    return f"{value:.0f}"


def _kde_curve(values: np.ndarray, pad: float) -> tuple:
    grid = np.linspace(values.min() - pad, values.max() + pad, 512)
    return grid, gaussian_kde(values)(grid)


def _plot_series(ax, series: Dict[str, List[int]], pad: float, colors: Colors) -> tuple:
    """Draw one log10 KDE per dataset; return the shared x range and peak."""
    left, right, top = np.inf, -np.inf, 0.0

    for name, values in series.items():
        positive = [value for value in values if value > 0]
        if len(positive) < len(values):
            print(f"  {name}: dropped {len(values) - len(positive)} zero-length traces")
        if len(positive) < 2:
            print(f"  {name}: too few traces to fit a KDE; not drawn")
            continue

        grid, density = _kde_curve(np.log10(positive), pad)
        ax.plot(
            grid,
            density,
            color=colors[name],
            linewidth=1.6,
            label=f"{name} (n={len(positive):,})",
        )
        left, right, top = min(left, grid.min()), max(right, grid.max()), max(top, density.max())

    return left, right, top


def _log_ticks(ax, left: float, right: float) -> None:
    ticks = np.arange(np.ceil(left), np.floor(right) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_human(10.0**tick) for tick in ticks])


def _finish(fig, ax, out_path: Path) -> Path:
    ax.set_ylabel("Density")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    return out_path


def plot_token_kde(series: Dict[str, List[int]], colors: Colors, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    left, right, top = _plot_series(ax, series, 0.3, colors)

    # Keep every limit line inside the frame even when no trace reaches it.
    limits = [np.log10(limit) for limit in CONTEXT_LIMITS.values()]
    left, right = min(left, min(limits) - 0.15), max(right, max(limits) + 0.15)
    ax.set_xlim(left, right)
    ax.set_ylim(0, top * 1.18)
    ceiling = ax.get_ylim()[1]

    for model, limit in CONTEXT_LIMITS.items():
        position = np.log10(limit)
        ax.axvline(position, color=LIMIT_COLORS[model], linestyle="--", linewidth=1.3)
        ax.text(
            position - 0.02,
            ceiling * 0.97,
            f"{model} Context Limit ({_human(limit)})",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=LIMIT_COLORS[model],
        )

    _log_ticks(ax, left, right)
    ax.set_xlabel("Token Lengths (Log10 Scale)")
    ax.set_title("Token Length Distribution (Log10 Scale)")
    ax.legend(fontsize=8, loc="upper left")

    return _finish(fig, ax, out_path)


def plot_step_kde(series: Dict[str, List[int]], colors: Colors, out_path: Path) -> Path:
    """Step counts span 1..~200 across datasets, so this axis is log10 too."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    left, right, top = _plot_series(ax, series, 0.15, colors)

    ax.set_xlim(left, right)
    ax.set_ylim(0, top * 1.1)
    _log_ticks(ax, left, right)
    ax.set_xlabel("Step Count (Log10 Scale)")
    ax.set_title("Step Count Distribution (Log10 Scale)")
    ax.legend(fontsize=8, loc="upper left")

    return _finish(fig, ax, out_path)


def run(datasets: List[Dataset], out_dir: Path) -> None:
    tokens: Dict[str, List[int]] = {}
    steps: Dict[str, List[int]] = {}

    for dataset in datasets:
        lengths = collect(dataset)
        summarize(dataset.name, lengths)

        tokens[dataset.name] = [length.n_tokens for length in lengths]
        step_values = [length.n_steps for length in lengths if length.n_steps is not None]
        if step_values:
            steps[dataset.name] = step_values
        else:
            print("  no step structure in this dataset; left out of the step plot")

    # Colour is fixed per dataset so both figures read the same way, even
    # though MAST appears in only one of them.
    colors = {
        dataset.name: PALETTE[index % len(PALETTE)]
        for index, dataset in enumerate(datasets)
    }

    plot_token_kde(tokens, colors, out_dir / "trace_token_length.png")
    plot_step_kde(steps, colors, out_dir / "trace_step_length.png")
