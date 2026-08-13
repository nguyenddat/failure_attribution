"""Injected-error-category composition by coordination topology (AEGIS).

Groups the 6 AEGIS frameworks into topology buckets per the standardized
taxonomy in `experiments/0.framework_topology_taxonomy/finding_notes.md`,
not the AEGIS paper's §4.1 wording directly — exp 0 draws a line the paper
text blurs: a framework whose topology is fixed by its own design vs. a
*library* whose topology depends on how the caller configures it:

- agentverse: solver-critic-evaluator hierarchy -> Hierarchical
- magentic_one: single Orchestrator hub -> Centralized
- smoagents: library, topology set by config (manager+tool-agents is one
  possible shape, not the only one) -> Variable
- macnet: paper's own "configurable network topologies (chain/star/tree/
  fully-connected)" is exactly exp 0's Variable case, not a fixed
  Decentralized graph -> Variable
- dylan: dynamic graph-based agent interactions, no single authority -> Decentralized
- llm_debate: peer debate, no single authority -> Decentralized

magentic_one/smoagents always inject exactly 1 agent with 1
injection_strategy and are 100% confounded with their own benchmark (see
analyze_aegis_correlation.py) — smoagents' Variable slice (now pooled with
macnet instead of magentic_one's Centralized bucket) should still be read
with that caveat in mind.

Usage: python experiments/1.framework_environment_correlation/src/plot_aegis_pie_by_architecture.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

from schemas.mast import MAST_METADATA

json_dir = REPO_ROOT / "data/error_localization/multi_fault/aegis"
fig_path = RESULTS_DIR / "aegis_pie_group_by_architecture.png"

# From experiments/0.framework_topology_taxonomy/finding_notes.md — see docstring.
ARCHITECTURE = {
    "agentverse": "Hierarchical",
    "magentic_one": "Centralized",
    "smoagents": "Variable",
    "macnet": "Variable",
    "dylan": "Decentralized",
    "llm_debate": "Decentralized",
}
ARCH_ORDER = ["Hierarchical", "Centralized", "Decentralized", "Variable"]

GROUP_COLORS = {
    "System Design Issues": plt.get_cmap("Purples")(0.6),
    "Inter-Agent Misalignment": plt.get_cmap("Reds")(0.6),
    "Task Verification": plt.get_cmap("Greens")(0.6),
}


def code_to_group() -> dict:
    return {m.code: g.name for g in MAST_METADATA.groups for m in g.failure_modes}


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        framework = data["metadata"]["framework"]
        for agent in data["ground_truth"]["injected_agents"]:
            code = agent["error_type"].removeprefix("FM-")
            rows.append({"framework": framework, "code": code})
    return pd.DataFrame(rows)


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_aegis` first.")
    df = load_dataframe()
    df["architecture"] = df["framework"].map(ARCHITECTURE)
    code2group = code_to_group()
    df["group"] = df["code"].map(code2group)
    group_names = [g.name for g in MAST_METADATA.groups]

    n_traces_per_fw = {}
    for path in json_dir.glob("*.json"):
        fw = json.loads(path.read_text(encoding="utf-8"))["metadata"]["framework"]
        n_traces_per_fw[fw] = n_traces_per_fw.get(fw, 0) + 1
    members = {
        arch: sorted(fw for fw, a in ARCHITECTURE.items() if a == arch) for arch in ARCH_ORDER
    }

    fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(len(ARCH_ORDER) * 4.4, 5.5))

    for ax, arch in zip(axes, ARCH_ORDER):
        sub = df[df["architecture"] == arch]
        counts = sub["group"].value_counts().reindex(group_names).dropna()
        n_traces = sum(n_traces_per_fw[fw] for fw in members[arch])
        fw_list = ", ".join(members[arch])
        label = f"{arch}\n({fw_list})\nn={n_traces} traces, {int(counts.sum())} tags"
        ax.pie(
            counts.values,
            colors=[GROUP_COLORS[g] for g in counts.index],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title(label, fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in group_names]
    fig.legend(handles, group_names, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "AEGIS: injected error-category composition by coordination topology\n"
        "Topology per experiment 0's taxonomy (smoagents/macnet = Variable, config-dependent "
        "libraries, not Centralized/Decentralized) — magentic_one/smoagents always inject "
        "1 agent/1 strategy, interpret cautiously",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 0.86])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(df.groupby("architecture")["group"].value_counts().unstack().fillna(0).astype(int).to_string())
    print(f"\nSaved -> {fig_path}")


if __name__ == "__main__":
    main()
