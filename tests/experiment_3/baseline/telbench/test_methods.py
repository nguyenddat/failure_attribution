from __future__ import annotations

from unittest.mock import patch

from baseline.telbench.methods import (
    all_at_once_single_file,
    step_by_step_single_file,
)


def _sample_data():
    return {
        "id": "0001",
        "question": "What is the answer?",
        "spans": [
            {"id": "s001", "raw": "retrieve info"},
            {"id": "s002", "raw": "verify source"},
            {"id": "s003", "raw": "wrong conclusion"},
        ],
        "gold": {"error_span_ids": ["s003"]},
    }


def test_all_at_once_returns_predicted_span_and_metrics():
    with patch(
        "baseline.telbench.methods.invoke_structured",
        return_value=({"span_id": "s003"}, {"latency": 0.5, "input_tokens": 10, "output_tokens": 2}),
    ):
        accuracy, cost = all_at_once_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_span"] == "s003"
    assert accuracy["metrics"]["fea"] == 1.0
    assert accuracy["exceeded_max_token_limit"] is False
    assert cost == {"latency": 0.5, "input_tokens": 10, "output_tokens": 2}


def test_all_at_once_marks_exceeded_and_keeps_not_found():
    with patch(
        "baseline.telbench.methods.invoke_structured",
        side_effect=Exception("context_length_exceeded"),
    ):
        accuracy, cost = all_at_once_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_span"] is None
    assert accuracy["exceeded_max_token_limit"] is True
    assert accuracy["metrics"]["fea"] == 0.0
    assert cost == {"latency": 0.0, "input_tokens": 0, "output_tokens": 0}


def test_step_by_step_stops_at_first_error_span():
    call_results = iter(
        [
            ({"error_found": False}, {"latency": 0.1, "input_tokens": 5, "output_tokens": 1}),
            ({"error_found": False}, {"latency": 0.1, "input_tokens": 6, "output_tokens": 1}),
            ({"error_found": True}, {"latency": 0.1, "input_tokens": 7, "output_tokens": 1}),
        ]
    )
    with patch(
        "baseline.telbench.methods.invoke_structured",
        side_effect=lambda *a, **k: next(call_results),
    ):
        accuracy, cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_span"] == "s003"
    assert accuracy["metrics"]["fea"] == 1.0
    assert accuracy["exceeded_max_token_limit"] is False
    assert cost["input_tokens"] == 5 + 6 + 7


def test_step_by_step_keeps_prior_state_when_exceeded_mid_scan():
    call_results = iter(
        [
            ({"error_found": True}, {"latency": 0.1, "input_tokens": 5, "output_tokens": 1}),
        ]
    )

    def _side_effect(*args, **kwargs):
        try:
            return next(call_results)
        except StopIteration:
            raise RuntimeError("This model's maximum context length is 128000 tokens") from None

    with patch(
        "baseline.telbench.methods.invoke_structured",
        side_effect=_side_effect,
    ):
        # Force the loop to keep going past span s001 by using data with the
        # error span first, then a span that would trigger the (mocked)
        # second call to raise.
        data = _sample_data()
        data["gold"]["error_span_ids"] = ["s001"]
        accuracy, _cost = step_by_step_single_file(data, model_name="gpt-4o-mini")

    # error_found=True at span s001 -> loop returns immediately, second call
    # never happens, so no exceeded flag and pred_span is s001.
    assert accuracy["pred_span"] == "s001"
    assert accuracy["exceeded_max_token_limit"] is False


def test_step_by_step_not_found_when_no_span_flagged():
    with patch(
        "baseline.telbench.methods.invoke_structured",
        return_value=({"error_found": False}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
    ):
        accuracy, _cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_span"] is None
    assert accuracy["metrics"]["fea"] == 0.0
