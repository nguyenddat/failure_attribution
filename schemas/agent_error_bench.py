"""Schema for the AgentErrorBench dataset (ulab-uiuc/AgentDebug, Google Drive release).

Shared by the loader under ``data/error_localization/single_fault`` and by any
experiment reading the generated JSON files.

Mirrors every raw field from both the trajectory file
(``Original_Failure_Trajectory/<Env>/<trajectory_id>.json``) and its label
(``Label/<env>_labels.json``) — nothing is dropped. An earlier version of
this schema paired up messages into observation/action Step objects and
collapsed the single ``step_annotations`` entry into a flat ``Failure``;
this version keeps ``messages`` as the raw alternating list and
``step_annotations`` as the raw list of dicts (each entry's failing module
is a dynamic dict key, e.g. ``{"step": 5, "planning": {...}}`` — not worth
forcing into a fixed field name).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class Metadata(BaseModel):
    batch_idx: int
    env_id: int
    test_idx: int
    model: str
    steps: int
    won: bool
    timestamp: str
    environment: str
    pid: Optional[str] = None  # gaia only
    gamefile: Optional[str] = None  # alfworld only


class Label(BaseModel):
    trajectory_id: str
    LLM: str
    task_type: str
    critical_failure_step: int
    critical_failure_module: str
    # raw list of {"step": int, "<module>": {"failure_type": str, "reasoning": str}}
    step_annotations: List[Dict[str, Any]]


class Data(BaseModel):
    trajectory_id: str

    messages: List[Message]
    metadata: Metadata

    label: Label
