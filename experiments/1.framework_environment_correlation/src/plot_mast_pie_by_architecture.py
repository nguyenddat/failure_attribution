"""Failure-category composition by coordination topology (MAST).

Groups the 7 mas_name frameworks into topology buckets per the standardized
taxonomy in `experiments/0.framework_topology_taxonomy/finding_notes.md`
(not own guess) — MetaGPT is **Pipeline**, not Hierarchical (assembly-line
SOP, no manager hub), and AG2/AutoGen is **Variable** (library, topology is
config-dependent), not Decentralized. Each of the 4 usable frameworks maps
to a distinct bucket, so every pie is a single framework — no pooling.

AppWorld/HyperAgent/OpenManus are excluded entirely: their mast_annotation
is a known index-aligned duplication bug (see
analyze_mas_annotation_correlation.py), so folding them into any bucket
would contaminate it with fake data.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_mast_pie_by_architecture.py
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
from mast_known_issues import DUPLICATE_ANNOTATION_BUG

json_dir = REPO_ROOT / "data/error_categorization/mast"
fig_path = RESULTS_DIR / "mast_pie_group_by_architecture.png"

# From experiments/0.framework_topology_taxonomy/finding_notes.md:
# - ChatDev: fixed role hierarchy -> Hierarchical
# - MetaGPT: SOP assembly-line, sequential handoff, no manager hub -> Pipeline
# - Magentic-One: single Orchestrator hub delegates to workers -> Centralized
# - AG2/AutoGen (raw mas_name): library, topology set by config, not fixed -> Variable
# AppWorld/HyperAgent/OpenManus omitted (annotation bug), even though AppWorld's
# own "star topology" (paper's wording) would also read as Centralized per exp 0.
ARCHITECTURE = {
    "ChatDev": "Hierarchical",
    "MetaGPT": "Pipeline",
    "Magentic": "Centralized",
    "AG2": "Variable",
}
ARCH_ORDER = ["Hierarchical", "Pipeline", "Centralized", "Variable"]

GROUP_COLORS = {
    "System Design Issues": plt.get_cmap("Purples")(0.6),
    "Inter-Agent Misalignment": plt.get_cmap("Reds")(0.6),
    "Task Verification": plt.get_cmap("Greens")(0.6),
}


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        if path.name == "metadata.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["mas_name"] in DUPLICATE_ANNOTATION_BUG:
            continue
        row = {"mas_name": data["mas_name"]}
        row.update(data["mast_annotation"])
        rows.append(row)
    return pd.DataFrame(rows)


def code_to_group() -> dict:
    return {m.code: g.name for g in MAST_METADATA.groups for m in g.failure_modes}


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_mast` first.")
    df = load_dataframe()
    df["architecture"] = df["mas_name"].map(ARCHITECTURE)

    code2group = code_to_group()
    codes = list(code2group)
    group_names = [g.name for g in MAST_METADATA.groups]
    n_per_arch = df.groupby("architecture").size()
    members = {
        arch: sorted(fw for fw, a in ARCHITECTURE.items() if a == arch) for arch in ARCH_ORDER
    }

    fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(len(ARCH_ORDER) * 4.4, 5.5))

    for ax, arch in zip(axes, ARCH_ORDER):
        tags = df[df["architecture"] == arch][codes].sum()
        by_group = tags.groupby(code2group).sum().reindex(group_names).fillna(0)
        total_tags = int(by_group.sum())
        n = int(n_per_arch.get(arch, 0))
        fw_list = ", ".join(members[arch])
        label = f"{arch}\n({fw_list})\nn={n} traces, {total_tags} tags"
        ax.pie(
            by_group.values,
            colors=[GROUP_COLORS[g] for g in by_group.index],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title(label, fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in group_names]
    fig.legend(handles, group_names, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "MAST: failure-category composition by coordination topology\n"
        "Topology per experiment 0's taxonomy — AppWorld/HyperAgent/OpenManus excluded "
        "(index-aligned annotation duplication bug); every bucket here is 1 framework alone",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.13, 1, 0.86])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(df.groupby("architecture")[codes].sum().assign(n_traces=n_per_arch).to_string())
    print(f"\nSaved -> {fig_path}")


if __name__ == "__main__":
    main()
