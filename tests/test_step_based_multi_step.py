from single_fault.evaluate_segmentation import build_label, sort_methods
from single_fault.src.step_based_multi_step import (
    ContextMode,
    build_method_name,
    get_context_steps,
)


def build_trajectory(length: int) -> list[dict]:
    return [
        {
            "step": idx,
            "agent_name": f"agent_{idx}",
            "content": f"step {idx}",
        }
        for idx in range(length)
    ]


def test_get_context_steps_surrounding_backfills_from_other_side() -> None:
    trajectory = build_trajectory(6)

    context = get_context_steps(
        trajectory=trajectory,
        current_step=1,
        num_steps=4,
        context_mode=ContextMode.SURROUNDING,
    )

    assert [step["step"] for step in context] == [0, 2, 3, 4]


def test_get_context_steps_previous_only_never_uses_future_steps() -> None:
    trajectory = build_trajectory(6)

    context = get_context_steps(
        trajectory=trajectory,
        current_step=4,
        num_steps=3,
        context_mode=ContextMode.PREVIOUS_ONLY,
    )

    assert [step["step"] for step in context] == [1, 2, 3]


def test_get_context_steps_next_only_never_uses_past_steps() -> None:
    trajectory = build_trajectory(6)

    context = get_context_steps(
        trajectory=trajectory,
        current_step=1,
        num_steps=3,
        context_mode=ContextMode.NEXT_ONLY,
    )

    assert [step["step"] for step in context] == [2, 3, 4]


def test_build_method_name_encodes_direction() -> None:
    assert build_method_name(5, ContextMode.SURROUNDING) == "step_based_multi_step_w5"
    assert build_method_name(5, ContextMode.PREVIOUS_ONLY) == "step_based_multi_step_prev_w5"
    assert build_method_name(5, ContextMode.NEXT_ONLY) == "step_based_multi_step_next_w5"


def test_evaluation_labels_and_sort_support_directional_methods() -> None:
    methods = [
        "step_based_multi_step_next_w7",
        "step_based_multi_step_w5",
        "step_based_multi_step_prev_w5",
        "step_based_multi_step_next_w5",
    ]

    assert build_label("step_based_multi_step_prev_w5") == "Step-based\nprev, w=5"
    assert build_label("step_based_multi_step_next_w5") == "Step-based\nnext, w=5"
    assert sort_methods(methods) == [
        "step_based_multi_step_prev_w5",
        "step_based_multi_step_w5",
        "step_based_multi_step_next_w5",
        "step_based_multi_step_next_w7",
    ]
