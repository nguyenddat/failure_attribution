"""Schema for the TRAIL dataset (PatronusAI/TRAIL).

Shared by the loader under ``data/error_localization/multi_fault`` and by any
experiment reading the generated JSON files.

Only the fields needed for multi-fault error localization are kept; the loader
drops OpenTelemetry plumbing (resource/scope metadata, service name, trace
flags) and the ``input.value``/``output.value`` attributes that merely restate
``llm.input_messages.*``/``llm.output_messages.*`` on LLM spans.

The span tree stays nested (``child_spans``) because ``Error.location`` is a
``span_id`` inside that tree, not a step index.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LogRecord(BaseModel):
    timestamp: str
    severity_text: str
    body: Any


class Span(BaseModel):
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    timestamp: str

    # Semantic attributes (openinference.span.kind, llm.*, tool.*, ...).
    attributes: Dict[str, Any]
    logs: List[LogRecord] = []
    events: List[Dict[str, Any]] = []

    child_spans: List["Span"] = []


class Error(BaseModel):
    category: str
    # span_id of the span the error was annotated on
    location: str
    evidence: str = ""
    description: str = ""
    impact: str = ""


class Scores(BaseModel):
    reliability_score: float
    reliability_reasoning: str
    security_score: float
    security_reasoning: str
    instruction_adherence_score: float
    instruction_adherence_reasoning: str
    plan_opt_score: float
    plan_opt_reasoning: str
    overall: float


class Trace(BaseModel):
    trace_id: str
    # "gaia" | "swe_bench" -- the HF config the trace came from
    source: str

    # trajectory
    spans: List[Span]

    # labels
    errors: List[Error]
    scores: Scores
