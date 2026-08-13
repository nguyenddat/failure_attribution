"""Schema for the TRAIL dataset (PatronusAI/TRAIL).

Shared by the loader under ``data/error_localization/multi_fault`` and by any
experiment reading the generated JSON files.

Mirrors every raw field of both ``trace`` (OpenTelemetry span tree) and
``labels`` (``errors`` + ``scores``) — nothing is dropped, including the
OTel transport plumbing (resource/scope metadata, trace_state, links,
duration, status) that an earlier version of this schema filtered out.

The span tree stays nested (``child_spans``) because ``Error.location`` is a
``span_id`` inside that tree, not a step index.

The dataset ships no problem statement column, so ``question`` is recovered
from the trace itself (the "New task:" prompt). ``task_source`` records
which HF config (``gaia`` / ``swe_bench``) the trace came from, since that
information isn't in the trace/labels payload itself.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LogRecord(BaseModel):
    timestamp: str
    trace_id: str
    span_id: str
    trace_flags: int
    severity_text: str
    severity_number: int
    service_name: str
    body: Any
    resource_schema_url: str = ""
    resource_attributes: Dict[str, Any] = {}
    scope_schema_url: str = ""
    scope_name: str = ""
    scope_version: str = ""
    scope_attributes: Dict[str, Any] = {}
    log_attributes: Dict[str, Any] = {}
    evaluations: List[Any] = []
    annotations: List[Any] = []


class Span(BaseModel):
    timestamp: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_state: str = ""
    span_name: str
    span_kind: str = ""
    service_name: str = ""
    resource_attributes: Dict[str, Any] = {}
    scope_name: str = ""
    scope_version: str = ""
    span_attributes: Dict[str, Any]
    duration: str = ""
    status_code: str = ""
    status_message: str = ""
    events: List[Any] = []
    links: List[Any] = []
    logs: List[LogRecord] = []

    child_spans: List["Span"] = []


class Error(BaseModel):
    category: str
    # span_id of the span the error was annotated on
    location: str
    evidence: str = ""
    description: str = ""
    impact: str = ""


class Score(BaseModel):
    # 1-5, but averaged across annotators on some traces so not always integral.
    reliability_score: float
    reliability_reasoning: str = ""
    security_score: float
    security_reasoning: str = ""
    instruction_adherence_score: float
    instruction_adherence_reasoning: str = ""
    plan_opt_score: float
    plan_opt_reasoning: str = ""
    overall: float


class Trace(BaseModel):
    # problem fields
    trace_id: str
    question: str
    task_source: str  # gaia / swe_bench (HF config name)

    # trajectory
    spans: List[Span]

    # labels
    errors: List[Error]
    scores: List[Score]
