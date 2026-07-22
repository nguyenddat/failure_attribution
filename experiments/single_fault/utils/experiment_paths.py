from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

BASELINE_OUTPUT_DIR = EXPERIMENTS_DIR / "baseline" / "output"
STEP_BASED_SEGMENTATION_OUTPUT_DIR = (
    EXPERIMENTS_DIR / "step_based_segmentation" / "output"
)
TOKEN_BASED_SEGMENTATION_OUTPUT_DIR = (
    EXPERIMENTS_DIR / "token_based_segmentation" / "output"
)
DATASET_ANALYSIS_OUTPUT_DIR = EXPERIMENTS_DIR / "dataset_analysis" / "output"
STEP_BASED_CONTEXT_MODE_COMPARISON_OUTPUT_DIR = (
    EXPERIMENTS_DIR / "step_based_context_mode_comparison" / "output"
)
STEP_BASED_LENGTH_ANALYSIS_DIR = STEP_BASED_SEGMENTATION_OUTPUT_DIR / "length_analysis"
TOKEN_BASED_EVALUATION_DIR = TOKEN_BASED_SEGMENTATION_OUTPUT_DIR / "evaluation"


def baseline_accuracy_path(dataset_key: str) -> Path:
    return BASELINE_OUTPUT_DIR / f"{dataset_key}.csv"


def baseline_cost_path(dataset_key: str) -> Path:
    return BASELINE_OUTPUT_DIR / f"{dataset_key}_cost.csv"


def step_based_accuracy_path(dataset_key: str) -> Path:
    return (
        STEP_BASED_SEGMENTATION_OUTPUT_DIR / f"{dataset_key}_step_based_multi_step.csv"
    )


def step_based_cost_path(dataset_key: str) -> Path:
    return (
        STEP_BASED_SEGMENTATION_OUTPUT_DIR
        / f"{dataset_key}_step_based_multi_step_cost.csv"
    )


def token_based_accuracy_path(dataset_key: str) -> Path:
    return (
        TOKEN_BASED_SEGMENTATION_OUTPUT_DIR
        / f"{dataset_key}_token_based_multi_step.csv"
    )


def token_based_cost_path(dataset_key: str) -> Path:
    return (
        TOKEN_BASED_SEGMENTATION_OUTPUT_DIR
        / f"{dataset_key}_token_based_multi_step_cost.csv"
    )
