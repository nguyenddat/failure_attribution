from __future__ import annotations  # noqa: EXE002

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import DATASET_DIRS, NORMALIZERS  # noqa: E402
from export_excel import (  # noqa: E402
    ACCURACY_COLUMNS,
    COST_COLUMNS,
    is_row_done,
    load_or_init,
    save,
    upsert_row,
)
from methods import fixed_window_all_at_once_single_file  # noqa: E402
from segments import WINDOW_SIZES, method_name  # noqa: E402

from experiments.single_fault.utils.file import load_json  # noqa: E402, RUF100

MODEL_NAME = "gpt-4o-mini"

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("experiment_5")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
ACCURACY_PATH = RESULTS_DIR / "accuracy.xlsx"
COST_PATH = RESULTS_DIR / "cost.xlsx"


def iter_dataset_files(dataset_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(dataset_dir.glob("*.json"), key=lambda p: int(p.stem))
    return files[:limit] if limit else files


def process_file(dataset_key: str, window_size: int, file_path: Path) -> tuple[dict, dict] | None:
    raw = load_json(file_path)
    normalized = NORMALIZERS[dataset_key](raw)
    if normalized is None:
        logger.warning(
            "skip %s/%s: missing label after normalize", dataset_key, file_path.name
        )
        return None

    num_steps = len(normalized["trajectory"])
    accuracy, cost = fixed_window_all_at_once_single_file(
        data=normalized, window_size=window_size, model_name=MODEL_NAME
    )

    base = {
        "model": MODEL_NAME,
        "dataset": dataset_key,
        "method": method_name(window_size),
        "window_size": window_size,
        "file": file_path.name,
        "num_steps": num_steps,
    }
    accuracy_row = {
        **base,
        "gt_agent": accuracy["gt_agent"],
        "gt_step": accuracy["gt_step"],
        "pred_agent": accuracy["pred_agent"],
        "pred_step": accuracy["pred_step"],
        "agent_accuracy": accuracy["agent_accuracy"],
        "step_accuracy": accuracy["step_accuracy"],
        "status": "ok",
    }
    cost_row = {
        **base,
        "latency": cost["latency"],
        "input_tokens": cost["input_tokens"],
        "output_tokens": cost["output_tokens"],
        "status": "ok",
    }
    return accuracy_row, cost_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    accuracy_df = load_or_init(ACCURACY_PATH, ACCURACY_COLUMNS)
    cost_df = load_or_init(COST_PATH, COST_COLUMNS)

    for dataset_key, dataset_dir in DATASET_DIRS.items():
        files = iter_dataset_files(dataset_dir, args.limit)
        for window_size in WINDOW_SIZES:
            method = method_name(window_size)
            progress = tqdm(files, desc=f"{dataset_key}/{method}")
            for file_path in progress:
                if is_row_done(
                    accuracy_df, MODEL_NAME, dataset_key, method, file_path.name
                ) and is_row_done(cost_df, MODEL_NAME, dataset_key, method, file_path.name):
                    continue

                result = process_file(dataset_key, window_size, file_path)
                if result is None:
                    continue
                accuracy_row, cost_row = result

                accuracy_df = upsert_row(accuracy_df, accuracy_row)
                cost_df = upsert_row(cost_df, cost_row)
                save(accuracy_df, ACCURACY_PATH)
                save(cost_df, COST_PATH)


if __name__ == "__main__":
    main()
