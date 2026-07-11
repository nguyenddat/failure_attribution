# Step-Based Segmentation Directional Context Change

## Current State

- `single_fault/src/step_based_multi_step.py` evaluates each target step `i` with surrounding context from both sides.
- `get_surrounding_steps(...)` splits `num_steps` across previous and next steps, then backfills from the other side when one side is short.
- Output methods currently only represent this bidirectional assumption, for example `step_based_multi_step_w5`.

## Desired State

- Keep the existing bidirectional assumption for backward compatibility.
- Add two additional assumptions for the same `num_steps` value:
  - previous-only: use up to `num_steps` steps before `i`, never use later steps
  - next-only: use up to `num_steps` steps after `i`, never use earlier steps
- Store each assumption as a separate method so accuracy and cost can be compared directly.

## Handling Plan

- Refactor context selection in `single_fault/src/step_based_multi_step.py` to support explicit context modes.
- Run each dataset/file for all `(num_steps, context_mode)` combinations and write results under distinct method names.
- Update `single_fault/evaluate_segmentation.py` so the new method names are grouped, sorted, and labeled cleanly in summaries/plots.
