"""Schema for the AEGIS dataset (Fancylalala/AEGIS).

Shared by the loader under ``data/error_localization/multi_fault`` and by any
experiment reading the generated JSON files.

AEGIS traces are multi-agent conversations with 1-4 *injected* faulty agents,
so localization is at agent granularity (``FaultyAgent.agent_name``), not at a
step index. ``error_type`` reuses the MAST failure-mode codes (``FM-1.1`` ..
``FM-3.3``); ``schemas/mast.py`` holds the names and descriptions.

Mirrors every raw field (``id``, full ``metadata``, ``input``, ``output``,
``ground_truth``) — nothing is dropped. Earlier version of this schema
dropped several fields (see git history); known data-quality quirks kept
intact rather than filtered out so they can be inspected directly:

* ``metadata.num_agents`` sometimes contradicts the length of
  ``ground_truth.injected_agents`` (e.g. num_agents=2 with 3 injected
  agents) — do not assume ``num_injected_agents == len(injected_agents)``.
* ``output.faulty_agents`` is a 3-field subset of ``ground_truth.
  injected_agents`` (missing ``malicious_action_description``) — expected
  to be redundant with it, not verified byte-identical after this rewrite.
* ``ground_truth.is_injection_successful`` — previously observed ``True``
  on every row; kept as a real bool field, not assumed constant.
* per-step ``AgentBehavior.is_injected`` is ``Optional[bool]`` — ``None``
  on most steps; where set, previously observed to disagree with
  ``injected_agents`` more often than agree — do not treat as ground truth.
"""

from typing import List, Optional

from pydantic import BaseModel


class AgentBehavior(BaseModel):
    step: int
    agent_name: str
    content: str
    # Framework stage: initialization / reasoning / discussion / evaluation.
    phase: str = ""
    is_injected: Optional[bool] = None


class OutputFaultyAgent(BaseModel):
    agent_name: str
    # MAST failure-mode code, e.g. "FM-1.1"
    error_type: str
    # prompt_injection / response_corruption / prompt
    injection_strategy: str


class InjectedAgent(BaseModel):
    agent_name: str
    error_type: str
    injection_strategy: str
    malicious_action_description: str = ""


class Metadata(BaseModel):
    framework: str
    benchmark: str
    model: str
    num_agents: int
    num_injected_agents: int
    task_type: str


class Input(BaseModel):
    query: str
    conversation_history: List[AgentBehavior]
    final_output: str = ""


class Output(BaseModel):
    faulty_agents: List[OutputFaultyAgent]


class GroundTruth(BaseModel):
    correct_answer: str
    injected_agents: List[InjectedAgent]
    is_injection_successful: bool


class Data(BaseModel):
    id: str
    split: str  # train / validation / test

    metadata: Metadata
    input: Input
    output: Output
    ground_truth: GroundTruth
