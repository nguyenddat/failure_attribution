"""Extract (dataset, framework, error-step position) rows across localization datasets.

Only datasets under ``data/error_localization/`` (not ``error_categorization``
— MAST is excluded: its ``mast_annotation`` is confirmed broken, see
papers/notes/findings/mast_data_finding.md) that carry a resolvable
step-level location for each error:

- TraceElephant: ``mistake_step`` / len(trajectory), single-fault.
- Who&When (hand-crafted): ``mistake_step`` / len(trajectory), single-fault.
  Every trace uses the same system (Magentic-One, per the paper — no
  ``framework`` field in the data itself), so this contributes exactly one
  framework row.
- AgentErrorBench: ``label.critical_failure_step`` / ``metadata.steps``,
  single-fault. No framework field either (architecture fixed at
  ReAct-4-module across all traces, only environment varies) — one
  pseudo-framework row.
- TRAIL: multi-fault. ``errors[].location`` is a span_id; resolved to a
  position by depth-first-flattening the span tree and finding the index.
  No framework field — task_source doubles as a pseudo-framework label
  (paper pairs one architecture per benchmark: OpenDeepResearch for GAIA,
  CodeAct for SWE-Bench — see note_trail.md).
- TELBENCH: multi-fault. ``gold.error_span_ids`` resolved to a position by
  index in the ordered ``spans`` list. Has a real ``meta.framework`` field
  (miroflow / oagent).

AEGIS is excluded: its label granularity is agent-only (``injected_agents``
has no step number), not step-level — see schemas/aegis.py docstring.

Framework labels are canonicalized so the same real system reported under
different casing across datasets (Who&When's "Magentic-One" vs
TraceElephant's "magentic-one") merges into one row.

Usage: python experiments/2.error_step_position_by_framework/src/extract_step_positions.py
"""

import json
from pathlib import Path
from typing import Iterator

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

CANONICAL_FRAMEWORK = {
    "magentic-one": "Magentic-One",
    "captain-agent": "Captain-Agent",
    "swe-agent": "SWE-Agent",
    "miroflow": "MiroFlow",
    "oagent": "OAgent",
}


def canon(name: str) -> str:
    return CANONICAL_FRAMEWORK.get(name.lower(), name)


def from_trace_elephant() -> Iterator[dict]:
    json_dir = REPO_ROOT / "data/error_localization/single_fault/trace_elephant"
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        mistake_step = data["mistake_step"]
        n_steps = len(data["trajectory"])
        if mistake_step is None or n_steps == 0:
            continue
        yield {
            "dataset": "TraceElephant",
            "framework": canon(data["system_name"]),
            "raw_step": mistake_step,
            "n_steps": n_steps,
            "position": mistake_step / n_steps,
        }


def from_who_and_when() -> Iterator[dict]:
    json_dir = REPO_ROOT / "data/error_localization/single_fault/who_and_when__hand-crafted"
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        n_steps = len(data["trajectory"])
        if n_steps == 0:
            continue
        yield {
            "dataset": "Who&When (hand-crafted)",
            "framework": canon("Magentic-One"),
            "raw_step": data["mistake_step"],
            "n_steps": n_steps,
            "position": data["mistake_step"] / n_steps,
        }


def from_agent_error_bench() -> Iterator[dict]:
    json_dir = REPO_ROOT / "data/error_localization/single_fault/agent_error_bench"
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        n_steps = data["metadata"]["steps"]
        step = data["label"]["critical_failure_step"]
        if not n_steps:
            continue
        yield {
            "dataset": "AgentErrorBench",
            "framework": "AgentDebug-ReAct",
            "raw_step": step,
            "n_steps": n_steps,
            "position": min(step / n_steps, 1.0),
        }


def flatten_spans(spans: list) -> list:
    flat = []
    for s in spans:
        flat.append(s)
        flat.extend(flatten_spans(s.get("child_spans") or []))
    return flat


TRAIL_PSEUDO_FRAMEWORK = {
    "gaia": "OpenDeepResearch (TRAIL)",
    "swe_bench": "CodeAct (TRAIL)",
}


def from_trail() -> Iterator[dict]:
    json_dir = REPO_ROOT / "data/error_localization/multi_fault/trail"
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        flat = flatten_spans(data["spans"])
        ids = [s["span_id"] for s in flat]
        n_steps = len(flat)
        if n_steps == 0:
            continue
        framework = TRAIL_PSEUDO_FRAMEWORK[data["task_source"]]
        for error in data["errors"]:
            if error["location"] not in ids:
                continue
            idx = ids.index(error["location"])
            yield {
                "dataset": "TRAIL",
                "framework": framework,
                "raw_step": idx,
                "n_steps": n_steps,
                "position": idx / n_steps,
            }


def from_telbench() -> Iterator[dict]:
    json_dir = REPO_ROOT / "data/error_localization/multi_fault/telbench"
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        span_ids = [s["id"] for s in data["spans"]]
        n_steps = len(span_ids)
        if n_steps == 0:
            continue
        framework = canon(data["meta"]["framework"])
        for sid in data["gold"]["error_span_ids"]:
            if sid not in span_ids:
                continue
            idx = span_ids.index(sid)
            yield {
                "dataset": "TELBENCH",
                "framework": framework,
                "raw_step": idx,
                "n_steps": n_steps,
                "position": idx / n_steps,
            }


EXTRACTORS = [from_trace_elephant, from_who_and_when, from_agent_error_bench, from_trail, from_telbench]


def main() -> None:
    rows = []
    for extractor in EXTRACTORS:
        rows.extend(list(extractor()))

    df = pd.DataFrame(rows)
    print(f"Total (dataset, framework, position) rows: {len(df)}")
    print(df.groupby(["dataset", "framework"]).size().to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "step_positions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
