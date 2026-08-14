from __future__ import annotations

from pathlib import Path

from experiments.single_fault.experiments.baseline.telbench.methods import (
    all_at_once_single_file,
    step_by_step_single_file,
)
from experiments.single_fault.experiments.baseline.telbench.metrics import (
    first_error_span,
)
from experiments.single_fault.experiments.baseline.telbench.results import (
    has_complete_method_result,
    load_or_init_results,
    sort_results,
    update_method_result,
    upsert_base_row,
)
from experiments.single_fault.utils.experiment_paths import BASELINE_OUTPUT_DIR
from experiments.single_fault.utils.file import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[5]
TELBENCH_DATA_DIR = (
    PROJECT_ROOT / "data" / "error_localization" / "multi_fault" / "telbench"
)
TELBENCH_OUTPUT_PATH = BASELINE_OUTPUT_DIR / "telbench.csv"
DEFAULT_MODEL_NAME = "gpt-4o-mini"


def _sample_file_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.json"), key=lambda path: int(path.stem))


def run_telbench(data_dir: Path, output_path: Path, model_name: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_or_init_results(output_path)

    methods = {
        "all_at_once": all_at_once_single_file,
        "step_by_step": step_by_step_single_file,
    }

    for file_path in _sample_file_paths(data_dir):
        data = load_json(file_path)
        span_ids_in_order = [span["id"] for span in data["spans"]]
        gold_spans = data["gold"]["error_span_ids"]
        gt_first_error = first_error_span(span_ids_in_order, gold_spans)

        df = upsert_base_row(df, file_path.name, gold_spans, gt_first_error)

        for method_name, method_fn in methods.items():
            if has_complete_method_result(df, file_path.name, method_name):
                continue

            accuracy, cost = method_fn(data, model_name)
            df = update_method_result(
                df,
                file_name=file_path.name,
                method_name=method_name,
                pred_span=accuracy["pred_span"],
                metrics=accuracy["metrics"],
                exceeded_max_token_limit=accuracy["exceeded_max_token_limit"],
                latency=cost["latency"],
                input_tokens=cost["input_tokens"],
                output_tokens=cost["output_tokens"],
            )
            df = sort_results(df)
            df.to_csv(output_path, index=False)

    df = sort_results(df)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    result_path = run_telbench(
        data_dir=TELBENCH_DATA_DIR,
        output_path=TELBENCH_OUTPUT_PATH,
        model_name=DEFAULT_MODEL_NAME,
    )
    print(f"Saved telbench baseline results to: {result_path}")


if __name__ == "__main__":
    main()
