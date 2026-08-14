from __future__ import annotations

from unittest.mock import patch

from baseline.who_and_when.methods import (
    all_at_once_single_file,
    step_by_step_single_file,
)


def _sample_data():
    return {
        "question": "solve it",
        "trajectory": [
            {"step": 0, "agent_name": "Planner", "content": "plan"},
            {"step": 1, "agent_name": "Coder", "content": "wrong code"},
        ],
        "mistake_agent": "Coder",
        "mistake_step": 1,
    }


def test_all_at_once_scores_correct_prediction():
    with patch(
        "baseline.who_and_when.methods.invoke_structured",
        return_value=({"step_number": 1}, {"latency": 0.1, "input_tokens": 3, "output_tokens": 1}),
    ):
        accuracy, cost = all_at_once_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Coder"
    assert accuracy["pred_step"] == 1
    assert accuracy["agent_accuracy"] == 1.0
    assert accuracy["step_accuracy"] == 1.0
    assert cost["input_tokens"] == 3


def test_step_by_step_stops_at_first_flagged_step():
    call_results = iter(
        [
            ({"error_found": False}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
            ({"error_found": True}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
        ]
    )
    with patch(
        "baseline.who_and_when.methods.invoke_structured",
        side_effect=lambda *a, **k: next(call_results),
    ):
        accuracy, cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Coder"
    assert accuracy["pred_step"] == 1
    assert accuracy["step_accuracy"] == 1.0
    assert cost["input_tokens"] == 2


def test_step_by_step_not_found_when_no_step_flagged():
    with patch(
        "baseline.who_and_when.methods.invoke_structured",
        return_value=({"error_found": False}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
    ):
        accuracy, cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Not Found"
    assert accuracy["pred_step"] == -1
    assert accuracy["step_accuracy"] == 0.0
