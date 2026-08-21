"""Cost-accuracy Pareto: tiền bỏ ra đổi lấy accuracy — exp 3 + 4 + 5.

Vẽ 4 hình: {cost, latency} × {step_accuracy, agent_accuracy}.

Mỗi điểm là một phương pháp: x = chi phí trung bình mỗi file (USD, tính từ
input/output token trong cost.xlsx), y = accuracy trung bình. Đường nét
đứt nối các điểm Pareto-optimal (không có phương pháp nào vừa rẻ hơn vừa chính
xác hơn). Tách 2 panel short/long giống plot_pooled_all.py vì luận điểm của cả
series là fixed window chỉ đáng tiền ở vùng trace dài.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from plot_pooled_all import (  # noqa: E402
    ACCURACY_PATHS,
    DATASETS,
    FIGURES_DIR,
    METRICS,
    MIN_STEPS_LONG,
    MODEL_NAME,
    WINDOW_SIZES,
    _load,
)

# gpt-4o-mini: $0.15 / 1M input token, $0.60 / 1M output token.
PRICE_INPUT = 0.15 / 1e6
PRICE_OUTPUT = 0.60 / 1e6

JOIN_KEYS = ["model", "dataset", "method", "file"]

# (cột x, nhãn trục x, tiêu đề)
X_AXES = [
    ("cost", "chi phí trung bình mỗi file (USD, log scale)", "Cost"),
    ("latency", "latency trung bình mỗi file (giây, log scale)", "Latency"),
]

# (label, color, marker, experiment key, method name template)
SERIES = [
    ("baseline all_at_once", "tab:gray", "s", "baselines", "all_at_once"),
    ("baseline step_by_step", "tab:red", "s", "baselines", "step_by_step"),
    ("per-step, both", "tab:blue", "o", "step", "fixed_window_w{w}_both"),
    ("per-step, prev_only", "tab:orange", "o", "step", "fixed_window_w{w}_prev_only"),
    ("per-step, next_only", "tab:green", "o", "step", "fixed_window_w{w}_next_only"),
    ("window all_at_once", "tab:purple", "^", "all_at_once", "fixed_window_all_at_once_w{w}"),
]


def _load_cost(exp_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(exp_dir / "results" / "tables" / "cost.xlsx")
    df = df[
        (df["model"] == MODEL_NAME)
        & (df["dataset"].isin(DATASETS))
        & (df["status"] == "ok")
    ]
    df = df.copy()
    df["cost"] = df["input_tokens"] * PRICE_INPUT + df["output_tokens"] * PRICE_OUTPUT
    return df[JOIN_KEYS + ["cost", "latency"]]


def _merged() -> pd.DataFrame:
    frames = []
    for key, path in ACCURACY_PATHS.items():
        accuracy = _load(path)[JOIN_KEYS + ["step_accuracy", "agent_accuracy", "num_steps"]]
        frames.append(accuracy.merge(_load_cost(path), on=JOIN_KEYS, how="inner"))
    return pd.concat(frames, ignore_index=True)


def _pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Điểm giữ lại khi không có điểm nào vừa rẻ hơn (x nhỏ hơn) vừa chính xác hơn."""
    frontier: list[tuple[float, float]] = []
    best_accuracy = -1.0
    for cost, accuracy in sorted(points):
        if accuracy > best_accuracy:
            frontier.append((cost, accuracy))
            best_accuracy = accuracy
    return frontier


def _plot_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    subtitle: str,
    x_col: str,
    x_label: str,
    value_col: str,
) -> None:
    means = df.groupby("method")[[x_col, value_col]].mean()
    points: list[tuple[float, float]] = []

    for label, color, marker, _exp_key, template in SERIES:
        methods = (
            [template]
            if "{w}" not in template
            else [template.format(w=w) for w in WINDOW_SIZES]
        )
        xs, ys = [], []
        for w, method in zip(WINDOW_SIZES, methods) if "{w}" in template else [(None, template)]:
            if method not in means.index:
                continue
            x, accuracy = means.loc[method, x_col], means.loc[method, value_col]
            xs.append(x)
            ys.append(accuracy)
            points.append((x, accuracy))
            if w is not None:
                ax.annotate(
                    f"w{w}",
                    (x, accuracy),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                    color=color,
                )
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, color=color, label=label, linewidth=1, alpha=0.8, zorder=3)

    frontier = _pareto(points)
    ax.step(
        [p[0] for p in frontier],
        [p[1] for p in frontier],
        where="post",
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="Pareto frontier",
        zorder=2,
    )
    ax.scatter(
        [p[0] for p in frontier],
        [p[1] for p in frontier],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{value_col} (mean)")
    n_files = len(df.drop_duplicates(["dataset", "file"]))
    ax.set_title(f"{subtitle} (n={n_files} file)")
    ax.grid(alpha=0.3, zorder=0)


def main() -> None:
    df = _merged()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for value_col, name_part in METRICS:
        for x_col, x_label, title in X_AXES:
            fig, (ax_short, ax_long) = plt.subplots(
                1, 2, figsize=(16, 6), sharey=True, constrained_layout=True
            )
            _plot_ax(
                ax_short,
                df[df["num_steps"] <= MIN_STEPS_LONG],
                f"num_steps <= {MIN_STEPS_LONG}",
                x_col,
                x_label,
                value_col,
            )
            _plot_ax(
                ax_long,
                df[df["num_steps"] > MIN_STEPS_LONG],
                f"num_steps > {MIN_STEPS_LONG}",
                x_col,
                x_label,
                value_col,
            )

            handles, labels = ax_short.get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=9)
            fig.suptitle(
                f"{title}-{value_col} Pareto: short vs long trajectories — experiment 3 + 4 + 5 "
                f"({' + '.join(DATASETS)} pooled, {MODEL_NAME})"
            )

            out_path = FIGURES_DIR / f"pooled_{x_col}_{name_part}_pareto.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"saved {out_path}")

    for value_col, _ in METRICS:
        summary = (
            df.groupby("method")[["cost", "latency", value_col]]
            .mean()
            .sort_values(value_col, ascending=False)
        )
        summary["acc_per_cent"] = summary[value_col] / (summary["cost"] * 100)
        print(summary.round({"cost": 5, "latency": 1, value_col: 3, "acc_per_cent": 2}))


if __name__ == "__main__":
    main()
