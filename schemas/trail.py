"""Schema for the TRAIL dataset (PatronusAI/TRAIL).

Shared by the loader under ``data/error_localization/multi_fault`` and by any
experiment reading the generated JSON files.

Each trace is kept raw: ``trace`` holds the OpenTelemetry span tree (nested via
``child_spans``) and ``labels`` the multi-fault annotations. ``labels.errors[].location``
is a ``span_id`` somewhere in that tree, not a step index -- flattening spans
into a step-indexed trajectory is left to a separate conversion step.
"""

from typing import Any, Dict

from pydantic import BaseModel


class Trace(BaseModel):
    trace_id: str
    # "gaia" | "swe_bench" -- the HF config the trace came from
    source: str

    trace: Dict[str, Any]

    # labels
    labels: Dict[str, Any]
