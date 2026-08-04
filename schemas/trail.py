"""Schema for the TRAIL dataset (PatronusAI/TRAIL).

Shared by the loader under ``data/error_localization/multi_fault`` and by any
experiment reading the generated JSON files.

Only the fields needed for multi-fault error localization are kept; the loader
drops OpenTelemetry plumbing (resource/scope metadata, service name, trace
flags) and the ``input.value``/``output.value`` attributes that merely restate
``llm.input_messages.*``/``llm.output_messages.*`` on LLM spans.

The span tree stays nested (``child_spans``) because ``Error.location`` is a
``span_id`` inside that tree, not a step index.

The dataset ships no problem statement column, so ``question`` is recovered
from the trace itself (the "New task:" prompt). It ships no reference answer at
all, so there is no ground-truth field.
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


class Trace(BaseModel):
    # problem fields
    question: str

    # trajectory
    spans: List[Span]

    # labels
    errors: List[Error]
