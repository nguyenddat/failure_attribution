# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Runtime

Activate the `idea_segment` conda environment before running any Python or pytest command:

```bash
conda activate idea_segment
```

If a required library is missing or the environment isn't ready, stop and report it — do not install dependencies or edit `requirements.txt`.

## Commands

`OPENROUTER_API_KEY` must be set in `.env` (loaded via `python-dotenv`) — experiment runs call an LLM through OpenRouter.

## Project purpose

Research pipeline for **failure attribution in multi-agent LLM systems**: given a failed agent trajectory (a sequence of per-agent steps) and a ground-truth `mistake_agent`/`mistake_step`, predict which agent and which step caused the failure. `single_fault/` is the active pipeline (single-fault-per-trajectory). `data/` holds input datasets. `multi_fault/` under `data/` is a separate, less-developed dataset track (not yet mirrored by a `multi_fault` methods pipeline).

`README.md` at repo root is an unrelated curated paper list ("Awesome Agentic Failure Attribution") plus notes on a Codex launcher (`bin/codex-coding-agent`) — it is not project usage documentation.

## Architecture

### Data flow

1. **Datasets** (`data/single_fault/json/<dataset_name>/*.json`, numbered `0.json`, `1.json`, ...) each hold `question`, `trajectory` (list of `{step, agent_name, content}`), `mistake_step`, `mistake_agent`. Dataset directories are registered in `single_fault/utils/datasets.py::DATASET_DIRS` (keys: `ww_algorithm_generated`, `ww_hand_crafted`).
2. **Methods** (`single_fault/methods/`) consume one trajectory file and return `(AccuracyMetrics, CostMetrics)` (see `single_fault/utils/schema.py`). Two families:
   - `methods/baselines/` — simple LLM-prompted baselines: `all_at_once`, `step_by_step`, `step_based_multi_step` (windowed context, `ContextMode` = surrounding/previous_only/next_only), `token_based_multi_step` (token-budget windows).
   - `methods/chief/` — the CHIEF method, split into pipeline blocks (`normalization`, `causal_construction` → `block_3_otar_parsing` → `block_4_hierarchical_causal_graph` → `block_5_virtual_oracles` → `block_6_backtracking` → `block_7_attribution`). Reference papers live in `methods/papers/` (CHIEF, FAMAS, who&when).
3. **Prompting** (`single_fault/system_prompt/`) defines the LangChain prompt + Pydantic output parser per method (`all_at_one.py`, `step_by_step.py`, `task_decomposition.py`, `subtask_alignment.py`). `single_fault/utils/get_chat_completion.py::get_prompt` maps a method name string to its `(prompt, parser)` pair; `get_chat_completion` runs the model call, wraps the parser in `OutputFixingParser` for self-repair, and returns cost metrics.
4. **Models** (`single_fault/utils/models.py`) — all LLM/embedding calls go through `openrouter.ai` via `langchain_openai`, keyed by short names in `models` dict (currently `gpt-4o-mini`, `embedding`). `get_model` caches clients.
5. **Experiments** (`single_fault/experiments/<name>/run.py`) wire a dataset × a set of `MethodConfig`s (method name, metadata method, runner fn) through `single_fault/experiments/shared.py::run_method_configs_for_dataset`, which is the shared driver: it iterates dataset files, resumes from existing per-dataset CSVs (skips already-complete rows via `has_complete_method_result`/`has_complete_method_cost`), calls the method's runner, and incrementally writes accuracy/cost CSVs to `output/` after every file (safe to interrupt/resume). Output paths are centralized in `single_fault/utils/experiment_paths.py`.
6. **Accuracy** — `agent_accuracy`/`step_accuracy` are 0/1 correctness of predicted agent/step vs. ground truth, compared via `single_fault/utils/accuracy.py::agent_names_match`.

### Key conventions

- `Metadata(model_name, method)` + a method-specific `*Input` pydantic model (`AllAtOnceInput`, `StepByStepInput`, `TaskDecompositionInput`, `SubtaskAlignmentInput`) is the standard call signature into `get_chat_completion`.
- Method runner functions follow the signature `(data: dict, metadata: Metadata, ...) -> tuple[AccuracyMetrics, CostMetrics]` and are designed to be resumable (accept optional partially-filled `accuracy_metrics`/`cost_metrics`).
- Every experiment's `output/` directory (CSVs, PNGs, markdown reports) is treated as a generated artifact — don't hand-edit or delete without being asked.
- CHIEF's `normalization` step turns raw trajectory dicts into a `TrajectoryIntakeArtifact` (validated `StepRecord` list) before any downstream block runs on it.

## Agent configs

`.codex/agents/coding_agent.toml` and `.codex/agents/method_agent.toml` define two Codex agent personas for this repo (Vietnamese-language instructions): a coding agent (respects `single_fault/` as the main pipeline, `data/` as input, treats `output/` dirs as generated artifacts, requires `conda activate segment` before running Python/pytest, must not add dependencies or edit `requirements.txt` on its own) and a paper-methodology-analysis agent (reads PDFs in `background/source/pdfs/`, writes Vietnamese HLD documents to `background/reviews/methodology_design/`, never writes code).
