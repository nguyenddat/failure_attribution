"""Schema for the AEGIS dataset (Fancylalala/AEGIS).

Shared by the loader under ``data/error_localization/multi_fault`` and by any
experiment reading the generated JSON files.

AEGIS traces are multi-agent conversations with 1-4 *injected* faulty agents,
so localization is at agent granularity (``FaultyAgent.agent_name``), not at a
step index. ``error_type`` reuses the MAST failure-mode codes (``FM-1.1`` ..
``FM-3.3``); ``schemas/mast.py`` holds the names and descriptions.

Only the fields needed for localization are kept. The loader drops:

* ``id`` and the ``metadata.model``/``num_agents``/``num_injected_agents``
  fields (``num_agents`` contradicts the injected-agent list on many rows, and
  the counts are derivable),
* ``output.faulty_agents``, byte-identical to ``ground_truth.injected_agents``
  on all 9533 rows,
* ``ground_truth.is_injection_successful``, ``True`` on every row,
* the per-step ``is_injected`` flag: it is null on 80% of steps, it leaks the
  label into the trajectory, and where it is set it disagrees with
  ``injected_agents`` on more traces than it agrees with.
"""

from typing import List

from pydantic import BaseModel


class AgentBehavior(BaseModel):
    step: int
    agent_name: str
    content: str

    # Framework stage: initialization / reasoning / discussion / evaluation.
    phase: str = ""


class FaultyAgent(BaseModel):
    agent_name: str
    # MAST failure-mode code, e.g. "FM-1.1"
    error_type: str
    # prompt_injection / response_corruption / prompt
    injection_strategy: str
    description: str = ""


class Data(BaseModel):
    # problem fields
    question: str
    framework: str
    benchmark: str
    task_type: str
    split: str

    # trajectory fields
    trajectory: List[AgentBehavior]
    final_output: str = ""

    # labels
    correct_answer: str
    faulty_agents: List[FaultyAgent]
