"""Schema for the who&when dataset (Kevin355/Who_and_When).

Shared by the loaders under ``data/error_localization/single_fault`` and by any
experiment reading the generated JSON files.

The Hand-Crafted and Algorithm-Generated splits have different raw columns
(e.g. ``is_corrected``/``groundtruth``/``question_ID``/``mistake_type`` only
exist on Hand-Crafted), so they get separate ``Data`` classes rather than
one shape forced to fit both.

``Data`` (used by ``ww_algorithm_generated.py``) keeps the original minimal
shape. ``HandCraftedData`` (used by ``ww_hand_crafted.py``) mirrors every
raw column of the Hand-Crafted split — nothing dropped.
"""

from typing import List, Optional

from pydantic import BaseModel


class AgentBehavior(BaseModel):
    step: int
    agent_name: str
    content: str


class Data(BaseModel):
    # problem fields
    question: str

    # trajectory fields
    trajectory: List[AgentBehavior]

    # labels
    mistake_step: int
    mistake_agent: str


class HandCraftedData(BaseModel):
    # problem fields
    question: str
    question_id: str
    groundtruth: str
    is_corrected: bool

    # trajectory fields
    trajectory: List[AgentBehavior]

    # labels
    mistake_step: int
    mistake_agent: str
    mistake_reason: str = ""
    # mostly null — only 4/58 rows populated in Hand-Crafted.
    mistake_type: Optional[str] = None
