import json
import csv
from pathlib import Path
from typing import Any, Dict, List

from src.step_by_step import step_by_step_single_file
from src.utils.file import load_json
from src.utils.models import model_names


def _numeric_sort_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def run_benchmark(
    data_dir: Path,
    samples: int = 5,
) -> Dict[str, Any]:
    sample_files = sorted(data_dir.glob("*.json"), key=_numeric_sort_key)[:samples]
    all_results: Dict[str, Any] = {}
    active_models = [m for m in model_names if ":free" not in m]

    for model_idx, model_name in enumerate(active_models, start=1):
        print(f"\n=== Running model {model_idx}/{len(active_models)}: {model_name} ===")
        details: List[Dict[str, Any]] = []
        for sample_idx, fp in enumerate(sample_files, start=1):
            print(f"[{model_name}] sample {sample_idx}/{len(sample_files)} -> {fp.name}")
            data = load_json(fp)
            total_steps = len(data["history"])
            result = step_by_step_single_file(
                model_name=model_name,
                data=data,
                current_step=0,
                total_steps=total_steps,
            )

            gt_agent = data.get("mistake_agent")
            gt_step_raw = data.get("mistake_step")
            gt_step = int(gt_step_raw) if gt_step_raw is not None else None

            pred_agent = result.get("mistake_agent")
            pred_step = result.get("mistake_step")
            metrics = result.get("metrics", {})

            details.append(
                {
                    "file": fp.name,
                    "pred_agent": pred_agent,
                    "pred_step": pred_step,
                    "gt_agent": gt_agent,
                    "gt_step": gt_step,
                    "agent_correct": pred_agent == gt_agent,
                    "step_correct": pred_step == gt_step,
                    "latency_s": float(metrics.get("latency_s_total", 0.0) or 0.0),
                    "cost_usd": float(metrics.get("cost_usd_estimate_total", 0.0) or 0.0),
                }
            )

        n = len(details)
        agent_acc = (sum(1 for r in details if r["agent_correct"]) / n) if n else 0.0
        step_acc = (sum(1 for r in details if r["step_correct"]) / n) if n else 0.0
        avg_latency_s = (sum(r["latency_s"] for r in details) / n) if n else 0.0
        total_price_usd = sum(r["cost_usd"] for r in details)
        avg_price_usd = (total_price_usd / n) if n else 0.0

        all_results[model_name] = {
            "summary": {
                "num_samples": n,
                "agent_level_accuracy": agent_acc,
                "step_level_accuracy": step_acc,
                "avg_latency_s": avg_latency_s,
                "avg_price_usd": avg_price_usd,
                "total_price_usd": total_price_usd,
            },
            "details": details,
        }

    return all_results


def save_outputs(results: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "step_by_step_3models_5samples_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_path = out_dir / "step_by_step_3models_5samples_comparison.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Step-by-Step Benchmark (3 models, 5 samples)\n\n")
        f.write("| Model | Agent Level Accuracy | Step Level Accuracy | Avg Latency (s) | Avg Price (USD) | Total Price (USD) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for model_name, payload in results.items():
            s = payload["summary"]
            f.write(
                f"| {model_name} | {s['agent_level_accuracy']:.2%} | {s['step_level_accuracy']:.2%} | "
                f"{s['avg_latency_s']:.2f} | ${s['avg_price_usd']:.6f} | ${s['total_price_usd']:.6f} |\n"
            )

    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "Who&When" / "Algorithm-Generated"
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "handcrafted_step_by_step_results.csv"
    method = "step_by_step"
    active_models = [m for m in model_names if ":free" not in m]
    sample_files = sorted(data_dir.glob("*.json"), key=_numeric_sort_key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "method",
                "model",
                "gt agent",
                "gt step",
                "pred agent",
                "pred step",
                "agent-level accuracy",
                "step-level accuracy",
                "total retries",
                "latency",
                "cost (price)",
            ]
        )

        for model_idx, model_name in enumerate(active_models, start=1):
            print(f"\n=== Running model {model_idx}/{len(active_models)}: {model_name} ===")
            for sample_idx, fp in enumerate(sample_files, start=1):
                print(f"[{model_name}] sample {sample_idx}/{len(sample_files)} -> {fp.name}")
                data = load_json(fp)
                total_steps = len(data["history"])
                result = step_by_step_single_file(
                    model_name=model_name,
                    data=data,
                    current_step=0,
                    total_steps=total_steps,
                )

                gt_agent = data.get("mistake_agent")
                gt_step_raw = data.get("mistake_step")
                gt_step = int(gt_step_raw) if gt_step_raw is not None else None

                pred_agent = result.get("mistake_agent")
                pred_step = result.get("mistake_step")
                metrics = result.get("metrics", {})

                agent_acc = 1 if pred_agent == gt_agent else 0
                step_acc = 1 if pred_step == gt_step else 0
                latency = float(metrics.get("latency_s_total", 0.0) or 0.0)
                cost = float(metrics.get("cost_usd_estimate_total", 0.0) or 0.0)
                total_retries = int(metrics.get("total_retries", 0) or 0)

                writer.writerow(
                    [
                        f"hand-crafted/{fp.name}",
                        method,
                        model_name,
                        gt_agent,
                        gt_step,
                        pred_agent,
                        pred_step,
                        agent_acc,
                        step_acc,
                        total_retries,
                        f"{latency:.6f}",
                        f"{cost:.10f}",
                    ]
                )
                # Flush after each sample so partial progress is persisted immediately.
                f.flush()

    print(f"\nSaved CSV: {csv_path}")
