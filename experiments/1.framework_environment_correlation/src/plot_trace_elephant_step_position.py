"""Failure-step position distribution by system (TraceElephant).

Histogram of ``mistake_step / n_steps`` (0=first step, 1=last step) per
system_name. Visual check of the TraceElephant paper's §4.2.2 claim:
CaptainAgent (dynamic team formation) has dispersed failure steps across
the timeline, while Magentic-One/SWE-Agent (fixed central orchestrator)
concentrate failures in early steps.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_trace_elephant_step_position.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

json_dir = REPO_ROOT / "data/error_localization/single_fault/trace_elephant"
fig_path = RESULTS_DIR / "trace_elephant_step_position_by_system.png"


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        n_steps = len(data["trajectory"])
        mistake_step = data["mistake_step"]
        if mistake_step is None:
            continue
        rows.append({"system_name": data["system_name"], "position": mistake_step / n_steps})
    return pd.DataFrame(rows)


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_trace_elephant` first.")
    df = load_dataframe()

    systems = df.groupby("system_name").size().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, len(systems), figsize=(len(systems) * 4.2, 4.5), sharey=True)

    for ax, sysn in zip(axes, systems):
        pos = df[df["system_name"] == sysn]["position"]
        ax.hist(pos, bins=10, range=(0, 1), color="steelblue", edgecolor="white")
        ax.axvline(pos.mean(), color="crimson", linestyle="--", linewidth=1, label=f"mean={pos.mean():.2f}")
        ax.set_title(f"{sysn} (n={len(pos)})", fontsize=11)
        ax.set_xlabel("failure step position\n(0=first step, 1=last step)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("# traces")
    fig.suptitle(
        "TraceElephant: failure-step position by system\n"
        "(paper claims CaptainAgent dispersed, Magentic-One/SWE-Agent early-concentrated)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Saved -> {fig_path}")


if __name__ == "__main__":
    main()
