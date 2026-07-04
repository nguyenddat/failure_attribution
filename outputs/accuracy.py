from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ACCURACY_DIR = BASE_DIR / "accuracy"
REPORT_PATH = BASE_DIR / "accuracy_report.md"


@dataclass
class MethodAccuracy:
    method: str
    num_samples: int
    avg_agent_accuracy: float
    avg_step_accuracy: float


def parse_accuracy_value(value: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return 1.0
    if normalized in {"false", "0"}:
        return 0.0

    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported accuracy value: {value!r}") from exc


def compute_method_accuracy(csv_path: Path) -> MethodAccuracy:
    agent_scores: list[float] = []
    step_scores: list[float] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"agent_accuracy", "step_accuracy"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

        for row in reader:
            agent_scores.append(parse_accuracy_value(row["agent_accuracy"]))
            step_scores.append(parse_accuracy_value(row["step_accuracy"]))

    num_samples = len(agent_scores)
    if num_samples == 0:
        raise ValueError(f"{csv_path.name} does not contain any data rows.")

    return MethodAccuracy(
        method=csv_path.stem,
        num_samples=num_samples,
        avg_agent_accuracy=sum(agent_scores) / num_samples,
        avg_step_accuracy=sum(step_scores) / num_samples,
    )


def build_report(results: list[MethodAccuracy]) -> str:
    lines = [
        "# Accuracy Report",
        "",
        f"Generated from CSV files in `{ACCURACY_DIR}`.",
        "",
        "| Method | Samples | Avg Agent Accuracy | Avg Step Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]

    for result in sorted(results, key=lambda item: item.method):
        lines.append(
            f"| {result.method} | {result.num_samples} | "
            f"{result.avg_agent_accuracy:.4f} ({result.avg_agent_accuracy * 100:.2f}%) | "
            f"{result.avg_step_accuracy:.4f} ({result.avg_step_accuracy * 100:.2f}%) |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    csv_files = sorted(ACCURACY_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {ACCURACY_DIR}")

    results = [compute_method_accuracy(csv_path) for csv_path in csv_files]
    report_content = build_report(results)
    REPORT_PATH.write_text(report_content, encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
