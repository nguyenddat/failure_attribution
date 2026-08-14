from __future__ import annotations

from baseline.telbench.metrics import (
    compute_metrics,
    first_error_span,
)


def test_first_error_span_picks_earliest_by_position():
    order = ["s001", "s002", "s003", "s004"]
    assert first_error_span(order, ["s003", "s001"]) == "s001"
    assert first_error_span(order, ["s004"]) == "s004"


def test_compute_metrics_exact_hit_single_gold():
    result = compute_metrics(pred_span="s001", gold_spans=["s001"], gt_first_error="s001")
    assert result == {"fea": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_compute_metrics_hit_but_not_first_error_multi_gold():
    # pred hits a gold span, but not the earliest one -> fea is 0, P/R/F1 still count the hit
    result = compute_metrics(
        pred_span="s003", gold_spans=["s001", "s003"], gt_first_error="s001"
    )
    assert result["fea"] == 0.0
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest_approx(2 / 3)


def test_compute_metrics_miss():
    result = compute_metrics(pred_span="s099", gold_spans=["s001"], gt_first_error="s001")
    assert result == {"fea": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_compute_metrics_not_found():
    result = compute_metrics(pred_span=None, gold_spans=["s001"], gt_first_error="s001")
    assert result == {"fea": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def pytest_approx(value):
    import pytest

    return pytest.approx(value)
