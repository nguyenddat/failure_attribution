from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from baseline.telbench.run import run_telbench


def _write_sample(dir_path: Path, index: int, spans: list[dict], gold: list[str]):
    data = {
        "id": f"{index:04d}",
        "question": "q",
        "spans": spans,
        "gold": {"error_span_ids": gold},
    }
    (dir_path / f"{index}.json").write_text(json.dumps(data))


def test_run_telbench_writes_csv_for_both_methods(tmp_path: Path):
    data_dir = tmp_path / "telbench"
    data_dir.mkdir()
    spans = [{"id": "s001", "raw": "a"}, {"id": "s002", "raw": "b"}]
    _write_sample(data_dir, 0, spans, ["s002"])

    output_path = tmp_path / "output" / "telbench.csv"

    def fake_all_at_once(data, model_name):
        return (
            {"pred_span": "s002", "metrics": {"fea": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}, "exceeded_max_token_limit": False},
            {"latency": 0.1, "input_tokens": 1, "output_tokens": 1},
        )

    def fake_step_by_step(data, model_name):
        return (
            {"pred_span": None, "metrics": {"fea": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}, "exceeded_max_token_limit": False},
            {"latency": 0.2, "input_tokens": 2, "output_tokens": 2},
        )

    with patch(
        "baseline.telbench.run.all_at_once_single_file",
        side_effect=fake_all_at_once,
    ), patch(
        "baseline.telbench.run.step_by_step_single_file",
        side_effect=fake_step_by_step,
    ):
        result_path = run_telbench(data_dir=data_dir, output_path=output_path, model_name="gpt-4o-mini")

    assert result_path == output_path
    df = pd.read_csv(output_path)
    assert len(df) == 1
    assert df.loc[0, "all_at_once_pred_span"] == "s002"
    assert df.loc[0, "step_by_step_fea"] == 0.0


def test_run_telbench_skips_already_complete_rows(tmp_path: Path):
    data_dir = tmp_path / "telbench"
    data_dir.mkdir()
    spans = [{"id": "s001", "raw": "a"}]
    _write_sample(data_dir, 0, spans, ["s001"])
    output_path = tmp_path / "output" / "telbench.csv"

    call_count = {"n": 0}

    def fake_method(data, model_name):
        call_count["n"] += 1
        return (
            {"pred_span": "s001", "metrics": {"fea": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}, "exceeded_max_token_limit": False},
            {"latency": 0.1, "input_tokens": 1, "output_tokens": 1},
        )

    with patch(
        "baseline.telbench.run.all_at_once_single_file",
        side_effect=fake_method,
    ), patch(
        "baseline.telbench.run.step_by_step_single_file",
        side_effect=fake_method,
    ):
        run_telbench(data_dir=data_dir, output_path=output_path, model_name="gpt-4o-mini")
        run_telbench(data_dir=data_dir, output_path=output_path, model_name="gpt-4o-mini")

    assert call_count["n"] == 2  # not 4 -> second run skipped both methods
