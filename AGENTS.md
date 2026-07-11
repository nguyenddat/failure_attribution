# Repository Guidelines

## Runtime Environment
- Agents must run `conda activate segment` before any Python- or pytest-based command so the correct Python runtime is available.
- If a required Python library is missing, activate `segment`, install it with `pip install <package>`, then update `requirements.txt` with `pip freeze > requirements.txt`.

## Project Structure & Module Organization
- `single_fault/` contains the main analysis pipeline: evaluation scripts in `single_fault/src/`, shared helpers in `single_fault/utils/`, prompt templates in `single_fault/system_prompt/`, and generated figures or summaries in `single_fault/output/` and `single_fault/data/`.
- `data/` holds dataset loaders and JSON inputs for `single_fault` and `multi_fault` experiments.
- `requirements.txt` lists Python dependencies. `README.md` is the primary project overview and should be checked before adding new workflows.

## Build, Test, and Development Commands
- Run `conda activate segment` before executing any of the commands below.
- `uv run python single_fault/evaluate_segmentation.py` runs the segmentation evaluation workflow.
- `uv run python single_fault/analyze_accuracy_by_length.py` and `uv run python single_fault/dataset_characteristics.py` generate analysis outputs.
- `uv run python single_fault/plot_length_bucket_comparison.py` and `uv run python single_fault/plot_length_bucket_cost_comparison.py` regenerate comparison figures.
- `uv run pytest -q` runs tests when present.
- Use `uv run python <script>` for new entrypoints so the project environment stays consistent after activating `segment`.
- If dependencies are missing during execution, run `pip install <package>` inside `segment`, then refresh `requirements.txt` with `pip freeze > requirements.txt`.

## Coding Style & Naming Conventions
- Use Python 3 style with 4-space indentation and type hints where practical.
- Prefer small, composable functions and descriptive names that match existing modules such as `step_based.py`, `token_based.py`, and `all_at_once.py`.
- Keep files and directories lowercase with underscores. Use clear dataset and method prefixes in outputs, for example `ww_hand_crafted` or `step_based_multi_step`.
- Follow the repository’s existing formatting unless a formatter is already configured for the target file.

## Testing Guidelines
- Add tests alongside code changes when practical, and name them after the behavior under test, such as `test_results.py` or `test_step_based.py`.
- Prefer deterministic checks for parsing, dataset loading, scoring, and result aggregation.
- Re-run `conda activate segment`, then the relevant `uv run python ...` script and `uv run pytest -q` after changes that affect evaluation logic or generated outputs.
- If testing or scripts reveal missing dependencies, install them in `segment` and persist the environment state back to `requirements.txt` with `pip freeze > requirements.txt`.

## Commit & Pull Request Guidelines
- Keep commit messages short and action-oriented, for example `fix segmentation scoring` or `add length bucket plot`.
- PRs should describe the changed workflow, the datasets or scripts affected, and any regenerated outputs.
- Include screenshots or sample figures when a change affects plots, charts, or saved artifacts.

## Data & Output Notes
- Treat `data/` and `single_fault/output/` as curated or generated project assets; avoid deleting them unless you are intentionally regenerating results.
- Prefer incremental edits so reruns stay reproducible and easy to compare.
