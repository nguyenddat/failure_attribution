from single_fault.experiments.baseline.run import build_method_configs as build_baseline_method_configs
from single_fault.experiments.step_based_segmentation.run import (
    build_method_configs as build_step_based_method_configs,
)
from single_fault.experiments.step_based_context_mode_comparison.run import (
    COMPARISON_CONTEXT_MODES,
    FIXED_NUM_STEPS,
    build_method_configs as build_step_based_context_mode_method_configs,
)
from single_fault.experiments.token_based_segmentation.run import (
    build_method_configs as build_token_based_method_configs,
)


def test_baseline_experiment_contains_two_baselines() -> None:
    method_names = [config.method_name for config in build_baseline_method_configs()]
    assert method_names == ["all_at_once", "step_by_step"]


def test_step_based_experiment_contains_expected_window_configs() -> None:
    method_names = [config.method_name for config in build_step_based_method_configs()]
    assert len(method_names) == 12
    assert "step_based_multi_step_w5" in method_names
    assert "step_based_multi_step_prev_w8" in method_names
    assert "step_based_multi_step_next_w8" in method_names


def test_token_based_experiment_contains_expected_token_levels() -> None:
    method_names = [config.method_name for config in build_token_based_method_configs()]
    assert method_names == [
        "token_based_multi_step_25pct",
        "token_based_multi_step_40pct",
        "token_based_multi_step_50pct",
    ]


def test_step_based_context_mode_comparison_contains_two_fixed_modes() -> None:
    method_names = [config.method_name for config in build_step_based_context_mode_method_configs()]
    assert FIXED_NUM_STEPS == 5
    assert [mode.value for mode in COMPARISON_CONTEXT_MODES] == ["previous_only", "next_only"]
    assert method_names == [
        "step_based_multi_step_prev_w5",
        "step_based_multi_step_next_w5",
    ]
