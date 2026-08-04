"""Schema for the AgentErrorBench dataset (local files, not Hugging Face).

Shared by the loader under ``data/error_localization/single_fault`` and by any
experiment reading the generated JSON files.

Every trajectory carries exactly one annotated failure, so this is single-fault
localization with two extra labels: the failing *module* and the *failure type*
inside that module. ``critical_failure_step``/``critical_failure_module`` in the
raw labels merely restate the single ``step_annotations`` entry, so they collapse
into one :class:`Failure`.

The raw messages strictly alternate ``user``/``assistant``, so each step pairs
the environment observation with the agent action that answered it; the step
index then matches the annotated step directly.

Only the fields needed for localization are kept. The loader drops
``trajectory_id``, ``metadata.won`` (``False`` on all 200 trajectories),
``metadata.steps`` (equal to the number of actions everywhere),
``metadata.environment`` (equal to ``task_type``), ``metadata.model`` (a noisier
spelling of the label's ``LLM``) and the run bookkeeping (``batch_idx``,
``env_id``, ``test_idx``, ``timestamp``, ``gamefile``, ``pid``).
"""

from typing import List

from pydantic import BaseModel

# The annotations name the same module two ways.
MODULE_ALIASES = {"planning": "plan"}

# Casing/word-order variants of the same failure type.
FAILURE_TYPE_ALIASES = {"plan_inefficient": "inefficient_plan"}


def normalize_module(module: str) -> str:
    module = module.strip().lower()
    return MODULE_ALIASES.get(module, module)


def normalize_failure_type(failure_type: str) -> str:
    failure_type = failure_type.strip().lower()
    return FAILURE_TYPE_ALIASES.get(failure_type, failure_type)


class Step(BaseModel):
    step: int
    # Environment output; on step 1 this is the prompt stating the task.
    observation: str
    action: str


class Failure(BaseModel):
    step: int
    # plan / action / memory / reflection / system
    module: str
    failure_type: str = ""
    reasoning: str = ""


class Data(BaseModel):
    # problem fields
    question: str
    task_type: str
    model: str

    # trajectory fields
    trajectory: List[Step]

    # labels
    failure: Failure
