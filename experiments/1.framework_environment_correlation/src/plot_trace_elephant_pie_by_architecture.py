"""Failure-role composition by coordination topology (TraceElephant).

Groups the 3 TraceElephant systems into topology buckets per the
standardized taxonomy in
`experiments/0.framework_topology_taxonomy/finding_notes.md`, not own
guess:

- captain-agent: exp 0 classifies CaptainAgent as **Centralized** (dynamic)
  — a Captain hub assembles + directs the team at runtime, still a single
  hub, not a fixed multi-level hierarchy -> Centralized
- magentic-one: single central Orchestrator plans and delegates to
  WebSurfer/FileSurfer/Coder -> Centralized
- swe-agent: exp 0 classifies SWE-Agent as **Single-agent** — a single
  control loop, not multi-agent coordination at all -> Single-agent
- Hierarchical/Decentralized: empty — none of the 3 systems fit those
  buckets under exp 0's taxonomy. Left empty rather than forcing a fit.

Usage: python experiments/1.framework_environment_correlation/src/plot_trace_elephant_pie_by_architecture.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

from trace_elephant_agent_roles import ROLE_ORDER, classify

json_dir = REPO_ROOT / "data/error_localization/single_fault/trace_elephant"
fig_path = RESULTS_DIR / "trace_elephant_pie_role_by_architecture.png"

# From experiments/0.framework_topology_taxonomy/finding_notes.md — see docstring.
ARCHITECTURE = {
    "captain-agent": "Centralized",
    "magentic-one": "Centralized",
    "swe-agent": "Single-agent",
}
ARCH_ORDER = ["Hierarchical", "Centralized", "Decentralized", "Single-agent"]

ROLE_COLORS = {
    "Orchestrator": plt.get_cmap("Purples")(0.6),
    "Worker": plt.get_cmap("Reds")(0.6),
    "Verification": plt.get_cmap("Greens")(0.6),
}


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"system_name": data["system_name"], "role": classify(data["mistake_agent"])})
    return pd.DataFrame(rows)


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_trace_elephant` first.")
    df = load_dataframe()
    df["architecture"] = df["system_name"].map(ARCHITECTURE)

    n_per_sys = df.groupby("system_name").size()
    members = {
        arch: sorted(s for s, a in ARCHITECTURE.items() if a == arch) for arch in ARCH_ORDER
    }
    non_empty = [arch for arch in ARCH_ORDER if members[arch]]

    fig, axes = plt.subplots(1, len(non_empty), figsize=(max(len(non_empty) * 4.4, 9.5), 5.8))
    if len(non_empty) == 1:
        axes = [axes]

    for ax, arch in zip(axes, non_empty):
        sub = df[df["architecture"] == arch]
        counts = sub["role"].value_counts().reindex(ROLE_ORDER).dropna()
        n_traces = int(sum(n_per_sys[s] for s in members[arch]))
        sys_list = ", ".join(members[arch])
        label = f"{arch}\n({sys_list})\nn={n_traces} traces"
        ax.pie(
            counts.values,
            colors=[ROLE_COLORS[r] for r in counts.index],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title(label, fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLORS[r]) for r in ROLE_ORDER]
    fig.legend(handles, ROLE_ORDER, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "TraceElephant: failure-role composition by coordination topology\n"
        "Topology per experiment 0's taxonomy — Centralized pools captain-agent+\n"
        "magentic-one, swe-agent alone is Single-agent",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 0.82])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(df.groupby("architecture")["role"].value_counts().unstack().fillna(0).astype(int).to_string())
    print(f"\nSaved -> {fig_path}")


if __name__ == "__main__":
    main()
