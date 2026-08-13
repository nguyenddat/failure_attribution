"""Schema for the TraceElephant dataset (TraceElephant/TraceElephant on HF).

Shared by the loader under ``data/error_localization/single_fault`` and by
any experiment reading the generated JSON files.

Single-fault localization, same shape as ``schemas/who_and_when.py``
(``mistake_step`` + ``mistake_agent``, exactly the field names used by the
source HF dataset itself) — only 220 traces are released, all of them
failed and fully annotated (no missing-label rows, unlike MAST).

``StepRecord.input``/``output``/``tool_logs`` are kept as raw nested dicts
(OpenAI-style chat completion objects) rather than collapsed into a single
``content`` string: the 3 source systems (captain-agent, magentic-one,
swe-agent) don't share one message format, and flattening would drop
information (tool-call arguments, multi-part content, token usage) that a
lossy per-system parser would need to reconstruct differently for each
system anyway.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StepRecord(BaseModel):
    step_id: int
    agent_id: int
    agent_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    tool_logs: Optional[List[Dict[str, Any]]] = None


class Data(BaseModel):
    # problem fields
    task_id: str
    task_instruction: str
    system_name: str  # captain-agent / magentic-one / swe-agent
    benchmark_name: str  # gaia / assistantbench / assistant-bench / swe-bench
    run_id: str
    agent_configuration: Dict[str, Any]
    agent_system_intro: str
    # swe-agent only
    tests_status: Optional[Dict[str, Any]] = None
    # captain-agent / magentic-one only
    ground_truth: Optional[str] = None

    # trajectory fields
    trajectory: List[StepRecord]

    # labels
    mistake_agent: str
    # int for all but 1 trace; source has "none" for 1 magentic-one trace where
    # no single step could be pinned down (kept as None rather than dropped).
    mistake_step: Optional[int] = None
    mistake_reason: str
