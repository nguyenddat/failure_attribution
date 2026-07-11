from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_RESULT_COLUMNS = ["file", "gt_agent", "gt_step"]


def build_result_columns(method_name: str) -> tuple[str, str]:
    return f"{method_name}_agent_acc", f"{method_name}_step_acc"


def build_prediction_columns(method_name: str) -> tuple[str, str]:
    return f"{method_name}_pred_agent", f"{method_name}_pred_step"


def build_cost_columns(method_name: str) -> tuple[str, str, str]:
    return (
        f"{method_name}_latency",
        f"{method_name}_input_tokens",
        f"{method_name}_output_tokens",
    )


def load_or_init_dataset_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=BASE_RESULT_COLUMNS)

    for column in BASE_RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")

    return df


def upsert_base_row(df: pd.DataFrame, file_name: str, gt_agent: str, gt_step: int) -> pd.DataFrame:
    row_mask = df["file"] == file_name
    if row_mask.any():
        df.loc[row_mask, "gt_agent"] = gt_agent
        df.loc[row_mask, "gt_step"] = gt_step
        return df

    df.loc[len(df)] = {
        "file": file_name,
        "gt_agent": gt_agent,
        "gt_step": gt_step,
    }
    return df


def update_method_result(
    df: pd.DataFrame,
    file_name: str,
    method_name: str,
    agent_accuracy: float,
    step_accuracy: float,
) -> pd.DataFrame:
    agent_acc_col, step_acc_col = build_result_columns(method_name)
    if agent_acc_col not in df.columns:
        df[agent_acc_col] = pd.Series(dtype="float")
    if step_acc_col not in df.columns:
        df[step_acc_col] = pd.Series(dtype="float")

    row_mask = df["file"] == file_name
    df.loc[row_mask, agent_acc_col] = agent_accuracy
    df.loc[row_mask, step_acc_col] = step_accuracy
    return df


def update_method_prediction(
    df: pd.DataFrame,
    file_name: str,
    method_name: str,
    pred_agent: str,
    pred_step: int,
) -> pd.DataFrame:
    pred_agent_col, pred_step_col = build_prediction_columns(method_name)
    if pred_agent_col not in df.columns:
        df[pred_agent_col] = pd.Series(dtype="object")
    if pred_step_col not in df.columns:
        df[pred_step_col] = pd.Series(dtype="float")

    row_mask = df["file"] == file_name
    df.loc[row_mask, pred_agent_col] = pred_agent
    df.loc[row_mask, pred_step_col] = pred_step
    return df


def update_method_cost(
    df: pd.DataFrame,
    file_name: str,
    method_name: str,
    latency: float,
    input_tokens: int,
    output_tokens: int,
) -> pd.DataFrame:
    latency_col, input_tokens_col, output_tokens_col = build_cost_columns(method_name)
    if latency_col not in df.columns:
        df[latency_col] = pd.Series(dtype="float")
    if input_tokens_col not in df.columns:
        df[input_tokens_col] = pd.Series(dtype="float")
    if output_tokens_col not in df.columns:
        df[output_tokens_col] = pd.Series(dtype="float")

    row_mask = df["file"] == file_name
    df.loc[row_mask, latency_col] = latency
    df.loc[row_mask, input_tokens_col] = input_tokens
    df.loc[row_mask, output_tokens_col] = output_tokens
    return df


def has_complete_method_result(df: pd.DataFrame, file_name: str, method_name: str) -> bool:
    agent_acc_col, step_acc_col = build_result_columns(method_name)
    if agent_acc_col not in df.columns or step_acc_col not in df.columns:
        return False

    row = df.loc[df["file"] == file_name, [agent_acc_col, step_acc_col]]
    if row.empty:
        return False

    return bool(row.notna().all(axis=1).iloc[0])


def has_complete_method_cost(df: pd.DataFrame, file_name: str, method_name: str) -> bool:
    latency_col, input_tokens_col, output_tokens_col = build_cost_columns(method_name)
    required_cols = [latency_col, input_tokens_col, output_tokens_col]
    if any(column not in df.columns for column in required_cols):
        return False

    row = df.loc[df["file"] == file_name, required_cols]
    if row.empty:
        return False

    return bool(row.notna().all(axis=1).iloc[0])


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    def sort_key(series: pd.Series) -> pd.Series:
        normalized = series.astype(str).str.replace(".json", "", regex=False)
        return pd.to_numeric(normalized, errors="coerce")

    return df.sort_values(by="file", key=sort_key, na_position="last").reset_index(drop=True)
