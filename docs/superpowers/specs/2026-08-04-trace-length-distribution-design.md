# Trace length distribution utilities

Date: 2026-08-04

## Goal

Report how long the traces are in every dataset in the repo, in two units:

* **token** - what an LLM actually pays for, counted with `tiktoken`,
* **step** - how many trajectory entries a trace has.

The token plot has to answer one concrete question: *how many traces do not fit
into the context window of the models we run?* The two models in scope are
`gpt-4o-mini` (128K context) and `deepseek-chat` (64K context).

## Scope

Six generated JSON datasets:

| Dataset key | JSON directory |
| --- | --- |
| `mast` | `data/error_categorization/mast/` |
| `who_and_when__algorithm-generated` | `data/error_localization/single_fault/who_and_when__algorithm-generated/` |
| `who_and_when__hand-crafted` | `data/error_localization/single_fault/who_and_when__hand-crafted/` |
| `agent_error_bench` | `data/error_localization/single_fault/agent_error_bench/` |
| `aegis` | `data/error_localization/multi_fault/aegis/` |
| `trail` | `data/error_localization/multi_fault/trail/` |

Out of scope: comparing tokenizers (only `o200k_base` is used), any per-agent or
per-step length breakdown, and any change to the loaders themselves.

## Architecture

One shared core module plus one thin adapter per dataset.

```
data/trace_length.py                                              # core
data/error_categorization/mast_length.py                          # adapter
data/error_localization/single_fault/ww_algorithm_generated_length.py
data/error_localization/single_fault/ww_hand_crafted_length.py
data/error_localization/single_fault/agent_error_bench_length.py
data/error_localization/multi_fault/aegis_length.py
data/error_localization/multi_fault/trail_length.py
```

The core knows nothing about any schema; adapters know nothing about tokenizing
or plotting. An adapter supplies three things: a display name, a JSON directory,
and two pure functions over one decoded JSON document.

### Core interface

```python
ENCODING = tiktoken.get_encoding("o200k_base")   # gpt-4o-mini tokenizer
CONTEXT_LIMITS = {"gpt-4o-mini": 128_000, "deepseek-chat": 64_000}

def count_tokens(parts: Iterable[str]) -> int:
    """Concatenate the full content of one trace and count its tokens."""

@dataclass
class TraceLength:
    name: str            # source file stem
    n_steps: int | None  # None when the dataset has no step structure
    n_tokens: int

def collect(json_dir: Path,
            contents: Callable[[dict], list[str]],
            steps: Callable[[dict], int | None]) -> list[TraceLength]

def summarize(name: str, lengths: list[TraceLength]) -> None
def plot_token_kde(name: str, values: list[int], out_dir: Path) -> Path
def plot_step_kde(name: str, values: list[int], out_dir: Path) -> Path
def run(name: str, json_dir: Path, contents, steps, out_dir: Path) -> None
```

`count_tokens` joins the parts with `"\n"`, skipping empty strings, and encodes
once. Joining before encoding (rather than summing per-part counts) matches how
the text reaches the model.

`collect` skips `metadata.json` and iterates `sorted(json_dir.glob("*.json"))`.
If the directory is empty it raises with a message naming the `make load_*`
target that generates it.

`run` is the single call an adapter makes: collect, summarize, write both plots.
When every trace reports `n_steps is None`, it prints a warning and writes only
the token plot.

### Adapter shape

```python
"""Trace length distribution for AEGIS."""

from pathlib import Path

from data.error_localization.multi_fault.aegis import json_dir
from data.trace_length import run

figures_dir = Path(__file__).resolve().parent / "figures"


def contents(d: dict) -> list[str]:
    return [d["question"], *(s["content"] for s in d["trajectory"]), d.get("final_output", "")]


def steps(d: dict) -> int:
    return len(d["trajectory"])


if __name__ == "__main__":
    run("aegis", json_dir, contents, steps, figures_dir)
```

Loaders currently expose `json_dir` only in `data/error_categorization/mast.py`.
The other five modules build the path locally from `base_dir`. Implementation
lifts that path into a module-level `json_dir` in each loader so adapters import
it instead of restating it. This is the only edit to existing loaders.

## Content and step extraction

"Full content of a trace" means the text a model would be shown: the problem
statement plus the whole trajectory.

| Dataset | `contents` | `steps` |
| --- | --- | --- |
| ww algorithm-generated, ww hand-crafted | `question`, then each `trajectory[i].content` | `len(trajectory)` |
| aegis | `question`, each `trajectory[i].content`, `final_output` | `len(trajectory)` |
| agent_error_bench | `question`, then each step's `observation` and `action` | `len(trajectory)` |
| mast | `raw_trajectory` | `len(trajectory)` when the field is present, else `None` |
| trail | `question`, then per span `json.dumps` of `attributes`, `logs`, `events` | recursive count over `spans` and `child_spans` |

MAST has no step structure of its own: `trajectory` is optional and is filled in
later by `build_agent_behaviors.py`. Its adapter returns `None` when the field is
missing, which makes `run` skip the step plot rather than fail.

TRAIL spans carry their text inside `attributes` (`llm.*`, `tool.*`,
`input.*`, `output.*`), `logs` and `events`, with no single content field, so the
adapter serializes those three containers per span. `attributes` is empty on
some spans; that contributes an empty JSON object, which is correct.

## Plots

Both plots are kernel density estimates via `scipy.stats.gaussian_kde` with the
default Scott bandwidth, drawn with matplotlib, saved at dpi 150.

**Token plot** - `<key>_token_length.png`

* KDE over `log10(n_tokens)`; traces with zero tokens are dropped and reported.
* X axis in log10 space with human tick labels: `1.0K`, `10K`, `100K`, `1.0M`.
* Y axis labelled `Density`.
* Two vertical dashed lines at `log10(limit)`: `gpt-4o-mini` 128,000 in red and
  `deepseek-chat` 64,000 in teal, each annotated with a rotated label reading
  `<model> Context Limit`.
* Title: `<Dataset>: Token Length Distribution (Log10 Scale)`.
* Legend states how many traces exceed each limit, e.g.
  `1,204 / 9,533 over gpt-4o-mini (128K)`.

**Step plot** - `<key>_step_length.png`

* KDE over the raw step counts (linear axis, no limit lines).
* X axis `Step count`, Y axis `Density`.
* Title: `<Dataset>: Step Count Distribution`.

Both files are written to a `figures/` directory beside the adapter, created if
absent.

**Console output** per dataset:

```
aegis  n=9533
  tokens  min=... median=... mean=... p95=... max=...
  steps   min=... median=... mean=... p95=... max=...
  over gpt-4o-mini (128,000): 12 (0.1%)
  over deepseek-chat (64,000): 340 (3.6%)
  saved: .../figures/aegis_token_length.png
  saved: .../figures/aegis_step_length.png
```

## Makefile

One target per dataset plus an aggregate, following the existing `load_*` naming:

```make
length_mast:          ; $(PYTHON) -m data.error_categorization.mast_length
length_ww_algo:       ; $(PYTHON) -m data.error_localization.single_fault.ww_algorithm_generated_length
length_ww_hand:       ; $(PYTHON) -m data.error_localization.single_fault.ww_hand_crafted_length
length_agent_error_bench: ; $(PYTHON) -m data.error_localization.single_fault.agent_error_bench_length
length_aegis:         ; $(PYTHON) -m data.error_localization.multi_fault.aegis_length
length_trail:         ; $(PYTHON) -m data.error_localization.multi_fault.trail_length

length_all: length_mast length_ww_algo length_ww_hand length_agent_error_bench length_aegis length_trail
```

Targets are added to the `.PHONY` list. They do not delete anything; figures are
overwritten in place.

## Replacing the existing MAST script

`data/error_categorization/analyze_trajectory_length.py` already boxplots MAST
token length with `cl100k_base`. It is deleted: `mast_length.py` covers the same
data with `o200k_base`, and keeping both would publish two token counts for the
same traces that disagree by tokenizer.

## Error handling

* Empty or missing JSON directory - raise, naming the `make load_*` target.
* A JSON document missing an expected key - let the `KeyError` propagate with the
  file path attached, so a schema drift is loud rather than silently zero-length.
* Fewer than two traces in a dataset - `gaussian_kde` cannot fit; print the stats
  and skip the plot.

## Testing

No test framework exists in the repo, so verification is by running the targets.

* `make length_all` completes for all six datasets.
* Both PNGs exist per dataset (five datasets; MAST produces only the token plot
  unless `build_agent_behaviors.py` has been run).
* Hand-check one trace: recompute `count_tokens` on a single known file and
  confirm the printed `max`/`min` bracket it.
* Confirm the over-limit counts in the legend equal a direct
  `sum(1 for x in tokens if x > limit)`.

## Dependencies

`tiktoken`, `scipy`, `matplotlib`, `numpy`, all present in the `rs_segment`
conda environment as of 2026-08-04 (scipy 1.18.0, matplotlib 3.11.1).
