from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.single_fault.utils.datasets import DATASET_DIRS
from experiments.single_fault.utils.experiment_paths import (
    baseline_accuracy_path,
    step_based_accuracy_path,
    STEP_BASED_LENGTH_ANALYSIS_DIR,
)


LENGTH_ANALYSIS_DIR = STEP_BASED_LENGTH_ANALYSIS_DIR
BASE_COLUMNS = ["file", "gt_agent", "gt_step"]
METHOD_PAIRS = [
    ("all_at_once_agent_acc", "step_based_multi_step_w5_agent_acc", "agent"),
    ("all_at_once_step_acc", "step_based_multi_step_w5_step_acc", "step"),
]
BUCKET_BINS = [0, 10, 20, 40, 60, 10_000]
BUCKET_LABELS = ["<10", "10-19", "20-39", "40-59", "60+"]
MIN_GROUP_SIZE = 5


def load_dataset_results(dataset_key: str) -> pd.DataFrame:
    baseline_path = baseline_accuracy_path(dataset_key)
    step_based_path = step_based_accuracy_path(dataset_key)

    baseline_df = pd.read_csv(baseline_path)
    step_based_df = pd.read_csv(step_based_path)
    return baseline_df.merge(step_based_df, on=BASE_COLUMNS, how="inner")


def load_lengths(dataset_dir: Path, files: pd.Series) -> list[int]:
    lengths: list[int] = []
    for file_name in files:
        row = json.loads((dataset_dir / file_name).read_text(encoding="utf-8"))
        lengths.append(len(row.get("trajectory", [])))
    return lengths


def build_exact_length_table(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [left for left, _, _ in METHOD_PAIRS] + [
        right for _, right, _ in METHOD_PAIRS
    ]
    grouped = df.groupby("length", as_index=True)
    result = grouped[metric_columns].mean().round(3)
    result.insert(0, "n", grouped.size())
    return result.reset_index()


def build_bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [left for left, _, _ in METHOD_PAIRS] + [
        right for _, right, _ in METHOD_PAIRS
    ]
    bucket_df = df.copy()
    bucket_df["bucket"] = pd.cut(
        bucket_df["length"], bins=BUCKET_BINS, labels=BUCKET_LABELS, right=False
    )
    grouped = bucket_df.groupby("bucket", observed=False)
    result = grouped[metric_columns].mean().round(3)
    result.insert(0, "n", grouped.size())
    return result.reset_index()


def build_threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for threshold in sorted(df["length"].unique()):
        short_df = df[df["length"] < threshold]
        long_df = df[df["length"] >= threshold]
        if len(short_df) < MIN_GROUP_SIZE or len(long_df) < MIN_GROUP_SIZE:
            continue

        record: dict[str, float | int | str] = {
            "threshold": int(threshold),
            "short_n": int(len(short_df)),
            "long_n": int(len(long_df)),
        }

        for all_col, w5_col, metric_name in METHOD_PAIRS:
            short_all = float(short_df[all_col].mean())
            long_all = float(long_df[all_col].mean())
            long_w5 = float(long_df[w5_col].mean())

            record[f"{metric_name}_all_short"] = round(short_all, 3)
            record[f"{metric_name}_all_long"] = round(long_all, 3)
            record[f"{metric_name}_w5_long"] = round(long_w5, 3)
            record[f"{metric_name}_drop"] = round(short_all - long_all, 3)
            record[f"{metric_name}_w5_gain"] = round(long_w5 - long_all, 3)

        records.append(record)

    return pd.DataFrame(records)


def summarize_thresholds(threshold_df: pd.DataFrame) -> dict[str, str]:
    summary: dict[str, str] = {}
    for metric_name in ["agent", "step"]:
        if threshold_df.empty:
            summary[f"{metric_name}_best_drop_threshold"] = "n/a"
            summary[f"{metric_name}_best_gain_threshold"] = "n/a"
            continue

        best_drop_row = threshold_df.loc[threshold_df[f"{metric_name}_drop"].idxmax()]
        best_gain_row = threshold_df.loc[
            threshold_df[f"{metric_name}_w5_gain"].idxmax()
        ]

        summary[f"{metric_name}_best_drop_threshold"] = (
            f">={int(best_drop_row['threshold'])} "
            f"(drop={best_drop_row[f'{metric_name}_drop']:.3f}, long_n={int(best_drop_row['long_n'])})"
        )
        summary[f"{metric_name}_best_gain_threshold"] = (
            f">={int(best_gain_row['threshold'])} "
            f"(gain={best_gain_row[f'{metric_name}_w5_gain']:.3f}, long_n={int(best_gain_row['long_n'])})"
        )
    return summary


def write_markdown_report(
    dataset_key: str,
    exact_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> None:
    summary = summarize_thresholds(threshold_df)
    report_path = LENGTH_ANALYSIS_DIR / f"{dataset_key}_summary.md"

    lines = [
        f"# {dataset_key} length analysis",
        "",
        "## Key takeaways",
        f"- Strongest `all_at_once` agent split: `{summary['agent_best_drop_threshold']}`",
        f"- Strongest `step_based_w5` agent gain over `all_at_once`: `{summary['agent_best_gain_threshold']}`",
        f"- Strongest `all_at_once` step split: `{summary['step_best_drop_threshold']}`",
        f"- Strongest `step_based_w5` step gain over `all_at_once`: `{summary['step_best_gain_threshold']}`",
        "",
        "## Bucket view",
        "",
        dataframe_to_markdown(bucket_df),
        "",
        "## Exact length view",
        "",
        dataframe_to_markdown(exact_df),
        "",
        "## Threshold view",
        "",
        dataframe_to_markdown(threshold_df),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"

    headers = [str(column) for column in df.columns]
    rows = [
        [format_markdown_value(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    separator = ["---"] * len(headers)

    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(table_lines)


def format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def main() -> None:
    LENGTH_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_key, dataset_dir in DATASET_DIRS.items():
        df = load_dataset_results(dataset_key)
        df["length"] = load_lengths(dataset_dir, df["file"])

        exact_df = build_exact_length_table(df)
        bucket_df = build_bucket_table(df)
        threshold_df = build_threshold_table(df)

        df.sort_values(["length", "file"]).to_csv(
            LENGTH_ANALYSIS_DIR / f"{dataset_key}_joined.csv", index=False
        )
        exact_df.to_csv(
            LENGTH_ANALYSIS_DIR / f"{dataset_key}_exact_length.csv", index=False
        )
        bucket_df.to_csv(LENGTH_ANALYSIS_DIR / f"{dataset_key}_bucket.csv", index=False)
        threshold_df.to_csv(
            LENGTH_ANALYSIS_DIR / f"{dataset_key}_threshold.csv", index=False
        )
        write_markdown_report(dataset_key, exact_df, bucket_df, threshold_df)

    print(f"Saved length-based analysis to: {LENGTH_ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
