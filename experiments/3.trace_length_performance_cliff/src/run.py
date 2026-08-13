from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.chat_models import model_names  # noqa: E402
from experiments.single_fault.methods.baselines.all_at_once import (  # noqa: E402
    all_at_once_single_file,
)
from experiments.single_fault.utils.file import format_agent_behaviors, load_json  # noqa: E402
from experiments.single_fault.utils.schema import Metadata  # noqa: E402

from datasets import DATASET_DIRS, NORMALIZERS  # noqa: E402
from export_excel import (  # noqa: E402
    ACCURACY_COLUMNS,
    COST_COLUMNS,
    is_row_done,
    load_or_init,
    save,
    upsert_row,
)
from token_check import count_tokens, exceeds_threshold  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("experiment_3")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
ACCURACY_PATH = RESULTS_DIR / "accuracy.xlsx"
COST_PATH = RESULTS_DIR / "cost.xlsx"

SRC_DIR = Path(__file__).resolve().parent
TELBENCH_SUBSAMPLE_PATH = SRC_DIR / "telbench_subsample.json"


def _load_telbench_subsample() -> Optional[set[str]]:
    if not TELBENCH_SUBSAMPLE_PATH.exists():
        return None
    return set(json.loads(TELBENCH_SUBSAMPLE_PATH.read_text(encoding="utf-8")))


def iter_dataset_files(
    dataset_key: str, dataset_dir: Path, limit: Optional[int]
) -> list[Path]:
    files = sorted(dataset_dir.glob("*.json"), key=lambda p: int(p.stem))
    if dataset_key == "telbench":
        subsample = _load_telbench_subsample()
        if subsample is not None:
            files = [f for f in files if f.name in subsample]
    return files[:limit] if limit else files


def is_fully_done(accuracy_df, cost_df, model_key: str, dataset_key: str, file_name: str) -> bool:
    return is_row_done(accuracy_df, model_key, dataset_key, file_name) and is_row_done(
        cost_df, model_key, dataset_key, file_name
    )


def process_file(
    model_key: str, dataset_key: str, file_path: Path
) -> Optional[tuple[dict, dict]]:
    raw = load_json(file_path)
    normalized = NORMALIZERS[dataset_key](raw)
    if normalized is None:
        logger.warning(
            "skip %s/%s/%s: missing label after normalize",
            model_key,
            dataset_key,
            file_path.name,
        )
        return None

    trajectory = normalized["trajectory"]
    num_steps = len(trajectory)
    chat_content = format_agent_behaviors(trajectory)
    token_count = count_tokens(chat_content)
    avg_token_count = token_count / num_steps if num_steps else 0.0

    base = {
        "model": model_key,
        "dataset": dataset_key,
        "file": file_path.name,
        "gt_agent": normalized["mistake_agent"],
        "gt_step": normalized["mistake_step"],
        "num_steps": num_steps,
        "avg_token_count": avg_token_count,
    }

    if exceeds_threshold(model_key, token_count):
        logger.error(
            "token_overflow model=%s dataset=%s file=%s tokens=%d",
            model_key,
            dataset_key,
            file_path.name,
            token_count,
        )
        accuracy_row = {
            **base,
            "pred_agent": None,
            "pred_step": None,
            "agent_accuracy": None,
            "step_accuracy": None,
            "status": "token_overflow",
        }
        cost_row = {
            **base,
            "latency": None,
            "input_tokens": None,
            "output_tokens": None,
            "status": "token_overflow",
        }
        return accuracy_row, cost_row

    metadata = Metadata(model_name=model_key, method="all_at_once")
    accuracy_metrics, cost_metrics = all_at_once_single_file(normalized, metadata)

    accuracy_row = {
        **base,
        "pred_agent": accuracy_metrics.pred_agent,
        "pred_step": accuracy_metrics.pred_step,
        "agent_accuracy": accuracy_metrics.agent_accuracy,
        "step_accuracy": accuracy_metrics.step_accuracy,
        "status": "ok",
    }
    cost_row = {
        **base,
        "latency": cost_metrics.latency,
        "input_tokens": cost_metrics.input_tokens,
        "output_tokens": cost_metrics.output_tokens,
        "status": "ok",
    }
    return accuracy_row, cost_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    accuracy_df = load_or_init(ACCURACY_PATH, ACCURACY_COLUMNS)
    cost_df = load_or_init(COST_PATH, COST_COLUMNS)

    for model_key in model_names:
        for dataset_key, dataset_dir in DATASET_DIRS.items():
            files = iter_dataset_files(dataset_key, dataset_dir, args.limit)
            progress = tqdm(files, desc=f"{model_key}/{dataset_key}")
            for file_path in progress:
                if is_fully_done(accuracy_df, cost_df, model_key, dataset_key, file_path.name):
                    continue

                result = process_file(model_key, dataset_key, file_path)
                if result is None:
                    continue
                accuracy_row, cost_row = result

                accuracy_df = upsert_row(accuracy_df, accuracy_row)
                cost_df = upsert_row(cost_df, cost_row)
                save(accuracy_df, ACCURACY_PATH)
                save(cost_df, COST_PATH)


if __name__ == "__main__":
    main()
