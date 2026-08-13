"""Schema for the TELBench dataset (NJU-LINK/TELBench).

Shared by the loader under ``data/error_localization/multi_fault`` and by
any experiment reading the generated JSON files.

Mirrors every field of the decrypted JSONL release — nothing is dropped.
Multi-fault: ``gold.error_span_ids`` has more than 1 entry on ~46% of the
1000 instances (up to 11).
"""

from typing import Dict, List

from pydantic import BaseModel


class Span(BaseModel):
    id: str
    raw: str


class Gold(BaseModel):
    error_span_ids: List[str]


class Annotations(BaseModel):
    stage: Dict[str, str]
    # only spans in gold.error_span_ids have an entry here
    error_type: Dict[str, str]


class Meta(BaseModel):
    bench: str
    difficulty: str
    framework: str
    model: str
    answer_status: str


class Data(BaseModel):
    id: str
    source_id: str
    question: str

    spans: List[Span]

    gold: Gold
    annotations: Annotations
    meta: Meta
