from __future__ import annotations

from unittest.mock import patch

from experiments.single_fault.experiments.baseline.trace_elephant.methods import (
    all_at_once_single_file,
    format_trajectory,
    step_by_step_single_file,
)


def _sample_data():
    return {
        "task_instruction": "solve it",
        "trajectory": [
            {
                "step_id": 0,
                "agent_id": "a0",
                "agent_name": "Orchestrator",
                "input": {"goal": "start"},
                "output": {"plan": "do x"},
                "tool_logs": [],
            },
            {
                "step_id": 1,
                "agent_id": "a1",
                "agent_name": "Worker",
                "input": {"task": "do x"},
                "output": {"result": "wrong"},
                "tool_logs": ["search"],
            },
        ],
        "mistake_agent": "Worker",
        "mistake_step": 1,
    }


def test_format_trajectory_includes_input_output_tool_logs():
    text = format_trajectory(_sample_data()["trajectory"])
    assert "step 1 - Worker" in text
    assert '"result": "wrong"' in text
    assert '"search"' in text


def test_all_at_once_scores_correct_prediction():
    with patch(
        "experiments.single_fault.experiments.baseline.trace_elephant.methods.invoke_structured",
        return_value=({"step_number": 1}, {"latency": 0.1, "input_tokens": 3, "output_tokens": 1}),
    ):
        accuracy, cost = all_at_once_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Worker"
    assert accuracy["pred_step"] == 1
    assert accuracy["agent_accuracy"] == 1.0
    assert accuracy["step_accuracy"] == 1.0


def test_step_by_step_not_found_when_no_step_flagged():
    with patch(
        "experiments.single_fault.experiments.baseline.trace_elephant.methods.invoke_structured",
        return_value=({"error_found": False}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
    ):
        accuracy, cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Not Found"
    assert accuracy["pred_step"] == -1
