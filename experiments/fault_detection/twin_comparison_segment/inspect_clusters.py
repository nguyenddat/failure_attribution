"""Evidence-gathering for the low-D / high-D cluster hypothesis (block 1).

For each sample: split adjacent pairs (i, i+1) into low-D cluster (D < Dt)
and high-D cluster (D >= Dt), sample up to 8 pairs per cluster (spread across
the trajectory), and record actual content signals - speaker/agent name and
a keyword-based action-type guess - to test:

  A. role/agent switch (structural, not a failure)
  B. phase transition (design -> coding -> testing, structural, not a failure)
  C. genuine drift/anomaly (off-task, repetition - what we actually want to
     flag for failure attribution)

Output: output/cluster_pairs.csv (raw evidence) + printed aggregate stats.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data.error_categorization.mast import Sample, json_dir
from experiments.fault_detection.twin_comparison_segment.distance import (
    adjacent_distances,
    detect_dt,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LENGTH_SPLIT_PATH = json_dir / "length_split.json"

PAIRS_PER_CLUSTER = 8

# MetaGPT-style logs
_AGENT_FROM_RE = re.compile(r"FROM:\s*(.+?)\s+TO:")
_AGENT_NEW_MSG_RE = re.compile(r"NEW MESSAGES:\s*\n+\s*([A-Za-z0-9_\-]+):")
# ChatDev-style logs: SOP phase marker, e.g. "**[Seminar Conclusion]**", "**[Update Codes]**"
_PHASE_TAG_RE = re.compile(r"\*\*\[([^\]]+)\]\*\*")
# ChatDev dialogue turn: "Programmer: ...", "Chief Technology Officer: ..."
_SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z ]{1,40}?):\s")


def extract_unit(content: str) -> str:
    """Structural unit a step belongs to: ChatDev SOP phase tag if present,
    else the speaking role, else MetaGPT-style FROM/NEW MESSAGES sender."""
    match = _PHASE_TAG_RE.search(content)
    if match:
        return f"phase:{match.group(1).strip()}"

    match = _SPEAKER_RE.match(content.strip())
    if match:
        return f"speaker:{match.group(1).strip()}"

    match = _AGENT_FROM_RE.search(content)
    if match:
        return f"speaker:{match.group(1).strip()}"
    match = _AGENT_NEW_MSG_RE.search(content)
    if match:
        return f"speaker:{match.group(1).strip()}"

    return "unknown"


def classify_action(content: str) -> str:
    lower = content.lower()
    if "action:" in lower and "content:" in lower:
        return "orchestration/requirement"
    if re.search(r"\bdef\s+\w+\(", content) or re.search(r"\bclass\s+\w+", content):
        return "code"
    if "assert " in content or "test_" in lower or "pytest" in lower:
        return "test"
    if any(k in lower for k in ("review", "critical", "comment", "suggest", "improve")):
        return "review"
    if any(k in lower for k in ("error", "traceback", "exception", "fail")):
        return "error/debug"
    return "other/discussion"


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


def spread_sample(indices: list[int], k: int) -> list[int]:
    if len(indices) <= k:
        return indices
    positions = np.linspace(0, len(indices) - 1, k).round().astype(int)
    return sorted({indices[p] for p in positions})


def collect_rows(sample_id: int, sample: Sample) -> list[dict]:
    steps = [s.content for s in sample.trajectory]
    distances = adjacent_distances(steps)
    dt, method = detect_dt(distances)

    low_idx = [i for i, d in enumerate(distances) if d < dt]
    high_idx = [i for i, d in enumerate(distances) if d >= dt]

    rows = []
    for cluster_name, idx_list in (("low", low_idx), ("high", high_idx)):
        for i in spread_sample(idx_list, PAIRS_PER_CLUSTER):
            unit_i = extract_unit(steps[i])
            unit_next = extract_unit(steps[i + 1])
            action_i = classify_action(steps[i])
            action_next = classify_action(steps[i + 1])
            rows.append({
                "sample_id": sample_id,
                "mas_name": sample.mas_name,
                "faults": "|".join(sample.faults),
                "dt": round(dt, 4),
                "dt_method": method,
                "step_i": i,
                "step_next": i + 1,
                "D": round(distances[i], 4),
                "cluster": cluster_name,
                "unit_i": unit_i,
                "unit_next": unit_next,
                "unit_switch": unit_i != unit_next,
                "action_i": action_i,
                "action_next": action_next,
                "same_action_type": action_i == action_next,
                "snippet_i": steps[i][:160].replace("\n", " "),
                "snippet_next": steps[i + 1][:160].replace("\n", " "),
            })
    return rows


def main() -> None:
    sample_ids = pick_sample_ids()
    all_rows = []
    for sample_id in sample_ids:
        path = json_dir / f"{sample_id}.json"
        sample = Sample.model_validate_json(path.read_text(encoding="utf-8"))
        all_rows.extend(collect_rows(sample_id, sample))

    df = pd.DataFrame(all_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "cluster_pairs.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved {len(df)} rows to {out_csv}")

    print("\n=== unit_switch rate by cluster (phase tag / speaker changed) ===")
    print(df.groupby("cluster")["unit_switch"].mean())

    print("\n=== same_action_type rate by cluster ===")
    print(df.groupby("cluster")["same_action_type"].mean())

    print("\n=== action_i x cluster counts ===")
    print(pd.crosstab(df["action_i"], df["cluster"]))

    print("\n=== unit_i -> unit_next transitions, high cluster ===")
    high = df[df["cluster"] == "high"]
    print(high.groupby(["unit_i", "unit_next"]).size().sort_values(ascending=False).head(20))

    print("\n=== unit_i -> unit_next transitions, low cluster ===")
    low = df[df["cluster"] == "low"]
    print(low.groupby(["unit_i", "unit_next"]).size().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()
