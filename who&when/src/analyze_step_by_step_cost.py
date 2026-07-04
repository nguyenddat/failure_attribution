from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, median


def numeric_file_order(file_name: str) -> int:
    stem = Path(file_name).stem
    try:
        return int(stem)
    except ValueError:
        return 10**9


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_rows(project_root: Path) -> list[dict[str, object]]:
    accuracy_path = project_root / "outputs" / "accuracy" / "step_by_step.csv"
    cost_path = project_root / "outputs" / "cost" / "Algorithm-Generated_step_by_step_cost.csv"

    accuracy_rows = read_csv_rows(accuracy_path)
    cost_rows = read_csv_rows(cost_path)
    cost_by_file = {row["file name"]: row for row in cost_rows}

    rows: list[dict[str, object]] = []
    for row in accuracy_rows:
        file_name = row["file name"]
        if file_name not in cost_by_file:
            continue

        cost_row = cost_by_file[file_name]
        merged = {
            "file_name": file_name,
            "file_order": numeric_file_order(file_name),
            "gt_agent": row["gt_agent"],
            "gt_step": int(row["gt_step"]),
            "pred_agent": row["pred_agent"],
            "pred_step": int(row["pred_step"]),
            "agent_accuracy": parse_bool(row["agent_accuracy"]),
            "step_accuracy": parse_bool(row["step_accuracy"]),
            "latency": float(cost_row["latency"]),
            "input_tokens": int(float(cost_row["input_tokens"])),
            "output_tokens": int(float(cost_row["output_tokens"])),
            "num_input_steps": int(float(cost_row["num_input_steps"])),
        }
        merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
        merged["predicted_calls_if_found"] = merged["pred_step"] + 1
        rows.append(merged)

    rows.sort(key=lambda item: (int(item["file_order"]), str(item["file_name"])))
    return rows


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def describe_numeric(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def print_summary(rows: list[dict[str, object]]) -> None:
    total_samples = len(rows)
    correct_rows = [row for row in rows if bool(row["step_accuracy"])]
    wrong_rows = [row for row in rows if not bool(row["step_accuracy"])]

    print("Step-by-step cost analysis")
    print(f"Total samples: {total_samples}")
    print(f"Step accuracy: {len(correct_rows) / total_samples:.4f} ({len(correct_rows)}/{total_samples})")
    print()

    pred_counts: dict[int, int] = {}
    for row in rows:
        step = int(row["pred_step"])
        pred_counts[step] = pred_counts.get(step, 0) + 1

    print("Predicted step distribution (sorted by step index):")
    for step, count in sorted(pred_counts.items()):
        print(f"  step {step}: {count}")
    print()

    for field in ["gt_step", "pred_step", "num_input_steps", "input_tokens", "output_tokens", "total_tokens", "latency"]:
        stats = describe_numeric(rows, field)
        print(
            f"{field}: min={stats['min']:.2f}, max={stats['max']:.2f}, "
            f"mean={stats['mean']:.2f}, median={stats['median']:.2f}"
        )
    print()

    early_threshold = 3
    early_pred_ratio = safe_mean([1.0 if int(row["pred_step"]) <= early_threshold else 0.0 for row in rows])
    early_gt_ratio = safe_mean([1.0 if int(row["gt_step"]) <= early_threshold else 0.0 for row in rows])
    print(f"Predictions at step <= {early_threshold}: {early_pred_ratio:.4f}")
    print(f"Ground-truth steps at step <= {early_threshold}: {early_gt_ratio:.4f}")
    print()

    mean_tokens_correct = safe_mean([float(row["total_tokens"]) for row in correct_rows])
    mean_tokens_wrong = safe_mean([float(row["total_tokens"]) for row in wrong_rows])
    mean_pred_correct = safe_mean([float(row["pred_step"]) for row in correct_rows])
    mean_pred_wrong = safe_mean([float(row["pred_step"]) for row in wrong_rows])

    print(f"Average total tokens when step prediction is correct: {mean_tokens_correct:.2f}")
    print(f"Average total tokens when step prediction is wrong: {mean_tokens_wrong:.2f}")
    print(f"Average predicted step when correct: {mean_pred_correct:.2f}")
    print(f"Average predicted step when wrong: {mean_pred_wrong:.2f}")
    print()

    corr_pred_tokens = pearson_corr(
        [float(row["pred_step"]) for row in rows],
        [float(row["total_tokens"]) for row in rows],
    )
    corr_steps_tokens = pearson_corr(
        [float(row["num_input_steps"]) for row in rows],
        [float(row["total_tokens"]) for row in rows],
    )
    print(f"Correlation(pred_step, total_tokens): {corr_pred_tokens:.4f}")
    print(f"Correlation(num_input_steps, total_tokens): {corr_steps_tokens:.4f}")
    print()

    print("Interpretation hint:")
    print(
        "Step-by-step only becomes expensive when the detected step is pushed later, "
        "because each additional prediction reuses a longer prefix of the history. "
        "If many predictions stop near the beginning, average cost can stay low."
    )


def write_analysis_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "file_name",
        "file_order",
        "gt_agent",
        "gt_step",
        "pred_agent",
        "pred_step",
        "agent_accuracy",
        "step_accuracy",
        "latency",
        "input_tokens",
        "output_tokens",
        "num_input_steps",
        "total_tokens",
        "predicted_calls_if_found",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def svg_circle(cx: float, cy: float, color: str, radius: float = 4, stroke: str = "none") -> str:
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius}" '
        f'fill="{color}" stroke="{stroke}" stroke-width="1" />'
    )


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" font-family="Arial">{text}</text>'


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def write_step_scatter_svg(rows: list[dict[str, object]], output_path: Path) -> None:
    width = 1400
    height = 640
    margin_left = 70
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_step = max(max(int(row["gt_step"]), int(row["pred_step"])) for row in rows)
    max_x = max(len(rows) - 1, 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white" />',
        svg_text(width / 2, 28, "Step-by-step prediction outcomes by sample", size=20, anchor="middle"),
    ]

    for step in range(max_step + 1):
        y = scale(step, 0, max_step, margin_top + plot_height, margin_top)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#dddddd" stroke-dasharray="4,4" />')
        parts.append(svg_text(margin_left - 10, y + 4, str(step), anchor="end"))

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black" />')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black" />')
    parts.append(svg_text(width / 2, height - 18, "Sample index (sorted by file name)", size=14, anchor="middle"))
    parts.append(svg_text(20, margin_top + plot_height / 2, "Step index", size=14))

    for index, row in enumerate(rows):
        x = scale(index, 0, max_x, margin_left, margin_left + plot_width)
        gt_y = scale(int(row["gt_step"]), 0, max_step, margin_top + plot_height, margin_top)
        if bool(row["step_accuracy"]):
            parts.append(svg_circle(x, gt_y, "navy"))
        else:
            pred_y = scale(int(row["pred_step"]), 0, max_step, margin_top + plot_height, margin_top)
            parts.append(f'<line x1="{x:.2f}" y1="{gt_y:.2f}" x2="{x:.2f}" y2="{pred_y:.2f}" stroke="#bbbbbb" />')
            parts.append(svg_circle(x, gt_y, "red"))
            parts.append(svg_circle(x, pred_y, "gold", stroke="black"))

    legend_x = width - 300
    legend_y = 70
    parts.append(svg_circle(legend_x, legend_y, "navy"))
    parts.append(svg_text(legend_x + 14, legend_y + 4, "GT step (correct prediction)"))
    parts.append(svg_circle(legend_x, legend_y + 24, "red"))
    parts.append(svg_text(legend_x + 14, legend_y + 28, "GT step (wrong prediction)"))
    parts.append(svg_circle(legend_x, legend_y + 48, "gold", stroke="black"))
    parts.append(svg_text(legend_x + 14, legend_y + 52, "Predicted step (wrong prediction)"))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_cost_scatter_svg(rows: list[dict[str, object]], output_path: Path) -> None:
    width = 900
    height = 620
    margin_left = 80
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_step = min(int(row["pred_step"]) for row in rows)
    max_step = max(int(row["pred_step"]) for row in rows)
    min_tokens = min(float(row["total_tokens"]) for row in rows)
    max_tokens = max(float(row["total_tokens"]) for row in rows)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white" />',
        svg_text(width / 2, 28, "Step-by-step cost vs predicted step", size=20, anchor="middle"),
    ]

    for tick in range(min_step, max_step + 1):
        x = scale(tick, min_step, max_step, margin_left, margin_left + plot_width)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#eeeeee" />')
        parts.append(svg_text(x, margin_top + plot_height + 20, str(tick), anchor="middle"))

    y_ticks = 6
    for tick_index in range(y_ticks + 1):
        token_value = min_tokens + (max_tokens - min_tokens) * tick_index / y_ticks
        y = scale(token_value, min_tokens, max_tokens, margin_top + plot_height, margin_top)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#eeeeee" />')
        parts.append(svg_text(margin_left - 10, y + 4, f"{token_value:.0f}", anchor="end"))

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black" />')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black" />')
    parts.append(svg_text(width / 2, height - 18, "Predicted step", size=14, anchor="middle"))
    parts.append(svg_text(24, margin_top + plot_height / 2, "Total tokens", size=14))

    for row in rows:
        x = scale(int(row["pred_step"]), min_step, max_step, margin_left, margin_left + plot_width)
        y = scale(float(row["total_tokens"]), min_tokens, max_tokens, margin_top + plot_height, margin_top)
        color = "navy" if bool(row["step_accuracy"]) else "red"
        parts.append(svg_circle(x, y, color))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_step_distribution_svg(rows: list[dict[str, object]], output_path: Path) -> None:
    width = 980
    height = 620
    margin_left = 80
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_step = max(max(int(row["gt_step"]), int(row["pred_step"])) for row in rows)
    min_step = min(min(int(row["gt_step"]), int(row["pred_step"])) for row in rows)

    gt_counts = {step: 0 for step in range(min_step, max_step + 1)}
    pred_counts = {step: 0 for step in range(min_step, max_step + 1)}
    for row in rows:
        gt_counts[int(row["gt_step"])] += 1
        pred_counts[int(row["pred_step"])] += 1

    max_count = max(max(gt_counts.values()), max(pred_counts.values()), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white" />',
        svg_text(width / 2, 28, "Ground-truth vs predicted step distribution", size=20, anchor="middle"),
    ]

    y_ticks = 6
    for tick_index in range(y_ticks + 1):
        count_value = max_count * tick_index / y_ticks
        y = scale(count_value, 0, max_count, margin_top + plot_height, margin_top)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" '
            'stroke="#eeeeee" />'
        )
        parts.append(svg_text(margin_left - 10, y + 4, f"{count_value:.0f}", anchor="end"))

    for step in range(min_step, max_step + 1):
        x = scale(step, min_step, max_step, margin_left, margin_left + plot_width)
        parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" '
            'stroke="#f5f5f5" />'
        )
        parts.append(svg_text(x, margin_top + plot_height + 20, str(step), anchor="middle"))

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black" />')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black" />')
    parts.append(svg_text(width / 2, height - 18, "Step index", size=14, anchor="middle"))
    parts.append(svg_text(28, margin_top + plot_height / 2, "Count", size=14))

    gt_points: list[str] = []
    pred_points: list[str] = []
    for step in range(min_step, max_step + 1):
        x = scale(step, min_step, max_step, margin_left, margin_left + plot_width)
        gt_y = scale(gt_counts[step], 0, max_count, margin_top + plot_height, margin_top)
        pred_y = scale(pred_counts[step], 0, max_count, margin_top + plot_height, margin_top)
        gt_points.append(f"{x:.2f},{gt_y:.2f}")
        pred_points.append(f"{x:.2f},{pred_y:.2f}")

    parts.append(
        f'<polyline fill="none" stroke="red" stroke-width="3" points="{" ".join(gt_points)}" />'
    )
    parts.append(
        f'<polyline fill="none" stroke="goldenrod" stroke-width="3" points="{" ".join(pred_points)}" />'
    )

    for step in range(min_step, max_step + 1):
        x = scale(step, min_step, max_step, margin_left, margin_left + plot_width)
        gt_y = scale(gt_counts[step], 0, max_count, margin_top + plot_height, margin_top)
        pred_y = scale(pred_counts[step], 0, max_count, margin_top + plot_height, margin_top)
        parts.append(svg_circle(x, gt_y, "red", radius=4))
        parts.append(svg_circle(x, pred_y, "gold", radius=4, stroke="black"))

    legend_x = width - 240
    legend_y = 70
    parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="red" stroke-width="3" />')
    parts.append(svg_text(legend_x + 28, legend_y + 4, "Ground-truth step"))
    parts.append(f'<line x1="{legend_x}" y1="{legend_y + 24}" x2="{legend_x + 20}" y2="{legend_y + 24}" stroke="goldenrod" stroke-width="3" />')
    parts.append(svg_text(legend_x + 28, legend_y + 28, "Predicted step"))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "outputs" / "step_by_step_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(project_root)
    print_summary(rows)

    csv_output = output_dir / "step_by_step_analysis.csv"
    step_plot_output = output_dir / "step_by_step_gt_vs_pred_plot.svg"
    cost_plot_output = output_dir / "step_by_step_cost_vs_pred_step.svg"
    dist_plot_output = output_dir / "step_by_step_gt_vs_pred_distribution.svg"

    write_analysis_csv(rows, csv_output)
    write_step_scatter_svg(rows, step_plot_output)
    write_cost_scatter_svg(rows, cost_plot_output)
    write_step_distribution_svg(rows, dist_plot_output)

    print()
    print(f"Saved merged analysis CSV to: {csv_output}")
    print(f"Saved requested step plot to: {step_plot_output}")
    print(f"Saved cost plot to: {cost_plot_output}")
    print(f"Saved distribution plot to: {dist_plot_output}")


if __name__ == "__main__":
    main()
