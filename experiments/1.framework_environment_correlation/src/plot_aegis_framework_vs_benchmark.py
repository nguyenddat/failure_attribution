"""Confusion-matrix-style heatmap: framework x benchmark (AEGIS).

Same purpose as MAST's plot_mas_name_vs_benchmark.py / TraceElephant's
plot_trace_elephant_system_vs_benchmark.py. Here the confound is total:
agentverse/dylan/llm_debate/macnet share the same 5 benchmarks (GSM8K,
HumanEval, MATH, MMLU, SciBench), while magentic_one and smoagents each
run on their own single, framework-specific benchmark label
("magentic+gaia", "smol+gaia") not shared with anyone else.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_aegis_framework_vs_benchmark.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

json_dir = REPO_ROOT / "data/error_localization/multi_fault/aegis"
fig_path = RESULTS_DIR / "aegis_framework_x_benchmark.png"


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"framework": data["metadata"]["framework"], "benchmark": data["metadata"]["benchmark"]})
    return pd.DataFrame(rows)


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_aegis` first.")
    df = load_dataframe()

    matrix = pd.crosstab(df["framework"], df["benchmark"])

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(7, matrix.shape[1] * 1.3), max(4, matrix.shape[0] * 0.8)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "# traces"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(f"AEGIS: framework x benchmark (n={len(df)} traces)")
    plt.xlabel("benchmark")
    plt.ylabel("framework")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(matrix.to_string())
    print(f"\nSaved -> {fig_path}")


if __name__ == "__main__":
    main()
