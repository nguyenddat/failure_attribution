from __future__ import annotations

from pathlib import Path

from baseline.telbench.results import (
    has_complete_method_result,
    load_or_init_results,
    sort_results,
    update_method_result,
    upsert_base_row,
)


def test_upsert_base_row_creates_and_updates_row(tmp_path: Path):
    df = load_or_init_results(tmp_path / "out.csv")
    df = upsert_base_row(df, "0.json", ["s001", "s003"], "s001")
    assert list(df.loc[0, ["file", "gold_spans", "gt_first_error"]]) == [
        "0.json",
        "s001;s003",
        "s001",
    ]

    df = upsert_base_row(df, "0.json", ["s001"], "s001")
    assert len(df) == 1
    assert df.loc[0, "gold_spans"] == "s001"


def test_update_method_result_and_completeness(tmp_path: Path):
    df = load_or_init_results(tmp_path / "out.csv")
    df = upsert_base_row(df, "0.json", ["s001"], "s001")

    assert has_complete_method_result(df, "0.json", "all_at_once") is False

    df = update_method_result(
        df,
        file_name="0.json",
        method_name="all_at_once",
        pred_span="s001",
        metrics={"fea": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        exceeded_max_token_limit=False,
        latency=1.2,
        input_tokens=100,
        output_tokens=10,
    )

    assert has_complete_method_result(df, "0.json", "all_at_once") is True
    assert df.loc[0, "all_at_once_pred_span"] == "s001"
    assert df.loc[0, "all_at_once_fea"] == 1.0
    assert df.loc[0, "all_at_once_exceeded_max_token_limit"] == False


def test_sort_results_orders_by_numeric_file_stem(tmp_path: Path):
    df = load_or_init_results(tmp_path / "out.csv")
    df = upsert_base_row(df, "10.json", ["s001"], "s001")
    df = upsert_base_row(df, "2.json", ["s001"], "s001")
    sorted_df = sort_results(df)
    assert list(sorted_df["file"]) == ["2.json", "10.json"]
