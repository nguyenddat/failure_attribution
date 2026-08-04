"""Trace length distribution shared by every dataset in the repo.

Length is reported in two units:

* **token** - what a model actually pays for. The whole trace is concatenated
  into one string and encoded with ``o200k_base``, the ``gpt-4o-mini``
  tokenizer.
* **step** - how many trajectory entries the trace has.

The token plot answers one question: how many traces do not fit into the
context window of the models we run (:data:`CONTEXT_LIMITS`)?

This module knows nothing about any dataset schema. Each dataset ships a thin
adapter next to its loader that supplies a name, a JSON directory and two pure
functions over one decoded document, then calls :func:`run`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

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

KDE_COLOR = "crimson"


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


def collect(json_dir: Path, contents: Contents, steps: Steps) -> List[TraceLength]:
    files = [path for path in sorted(json_dir.glob("*.json")) if path.name != "metadata.json"]
    if not files:
        raise FileNotFoundError(
            f"no JSON files in {json_dir}; generate them with the matching `make load_*` target"
        )

    lengths = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            lengths.append(
                TraceLength(
                    name=path.stem,
                    n_steps=steps(data),
                    n_tokens=count_tokens(contents(data)),
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
    """Format a token count for an axis tick: 1.0K, 10K, 100K, 1.0M."""
    for scale, suffix in ((1e6, "M"), (1e3, "K")):
        if value >= scale:
            scaled = value / scale
            return f"{scaled:.1f}{suffix}" if scaled < 10 else f"{scaled:.0f}{suffix}"
    return f"{value:.0f}"


def _kde_curve(values: np.ndarray, pad: float) -> tuple:
    grid = np.linspace(values.min() - pad, values.max() + pad, 512)
    return grid, gaussian_kde(values)(grid)


def _finish(fig, ax, out_dir: Path, filename: str) -> Path:
    ax.set_ylabel("Density")
    ax.grid(linestyle=":", alpha=0.4)
    ax.margins(y=0.05)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")
    return out_path


def plot_token_kde(name: str, values: List[int], out_dir: Path) -> Optional[Path]:
    positive = [value for value in values if value > 0]
    if len(positive) < len(values):
        print(f"  dropped {len(values) - len(positive)} empty traces from the token plot")
    if len(positive) < 2:
        print("  too few traces to fit a KDE; token plot skipped")
        return None

    log_values = np.log10(positive)
    grid, density = _kde_curve(log_values, pad=0.3)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, density, color=KDE_COLOR, linewidth=1.6, label=name)

    # Keep every limit line inside the frame even when no trace reaches it.
    limits = [np.log10(limit) for limit in CONTEXT_LIMITS.values()]
    left = min(grid.min(), min(limits) - 0.15)
    right = max(grid.max(), max(limits) + 0.15)
    ax.set_xlim(left, right)
    ax.set_ylim(0, density.max() * 1.18)
    top = ax.get_ylim()[1]

    over_counts = []
    for model, limit in CONTEXT_LIMITS.items():
        position = np.log10(limit)
        ax.axvline(position, color=LIMIT_COLORS[model], linestyle="--", linewidth=1.3)
        ax.text(
            position - 0.02,
            top * 0.97,
            f"{model} Context Limit",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=LIMIT_COLORS[model],
        )
        over = sum(1 for value in positive if value > limit)
        over_counts.append(f"{over:,} / {len(positive):,} over {model} ({_human(limit)})")

    ticks = np.arange(np.ceil(left), np.floor(right) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_human(10.0**tick) for tick in ticks])
    ax.set_xlabel("Token Lengths (Log10 Scale)")
    ax.set_title(f"{name}: Token Length Distribution (Log10 Scale)")
    ax.legend(title="\n".join(over_counts), title_fontsize=8, fontsize=8, loc="upper left")

    return _finish(fig, ax, out_dir, f"{name}_token_length.png")


def plot_step_kde(name: str, values: List[int], out_dir: Path) -> Optional[Path]:
    if len(values) < 2:
        print("  too few traces to fit a KDE; step plot skipped")
        return None

    array = np.asarray(values, dtype=float)
    grid, density = _kde_curve(array, pad=max(1.0, array.std() * 0.3))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, density, color=KDE_COLOR, linewidth=1.6, label=name)
    ax.set_xlabel("Step count")
    ax.set_title(f"{name}: Step Count Distribution")
    ax.legend(fontsize=8, loc="upper right")

    return _finish(fig, ax, out_dir, f"{name}_step_length.png")


def run(name: str, json_dir: Path, contents: Contents, steps: Steps, out_dir: Path) -> None:
    lengths = collect(json_dir, contents, steps)
    summarize(name, lengths)

    plot_token_kde(name, [length.n_tokens for length in lengths], out_dir)

    step_values = [length.n_steps for length in lengths if length.n_steps is not None]
    if step_values:
        plot_step_kde(name, step_values, out_dir)
    else:
        print("  no step structure in this dataset; step plot skipped")
