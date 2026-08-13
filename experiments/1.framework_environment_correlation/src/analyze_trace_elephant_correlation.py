"""Correlation between ``system_name`` and failure attribution (TraceElephant).

Two analyses, mirroring MAST's analyze_mas_annotation_correlation.py:

1. ``system_name x role`` (Orchestrator/Worker/Verification, from
   trace_elephant_agent_roles.py) — chi-square test of independence,
   Cramer's V.
2. ``system_name x failure-step position`` — mistake_step normalized by
   trajectory length (n_steps), bucketed into early/mid/late thirds. Tests
   the TraceElephant paper's own §4.2.2 claim: CaptainAgent (dynamic team
   formation) has dispersed failure steps, while Magentic-One/SWE-Agent
   (fixed central orchestrator) concentrate failures in early steps.

Also reports the system_name x benchmark_name confound (near-total except
GAIA, run by both captain-agent and magentic-one).

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/analyze_trace_elephant_correlation.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

from trace_elephant_agent_roles import classify

json_dir = REPO_ROOT / "data/error_localization/single_fault/trace_elephant"


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        n_steps = len(data["trajectory"])
        mistake_step = data["mistake_step"]
        rows.append(
            {
                "system_name": data["system_name"],
                "benchmark_name": data["benchmark_name"],
                "mistake_agent": data["mistake_agent"],
                "role": classify(data["mistake_agent"]),
                "mistake_step": mistake_step,
                "n_steps": n_steps,
                "position": (mistake_step / n_steps) if mistake_step is not None else None,
            }
        )
    return pd.DataFrame(rows)


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    chi2, p, _, _ = chi2_contingency(table)
    n = table.values.sum()
    k = min(table.shape)
    v = (chi2 / (n * (k - 1))) ** 0.5 if k > 1 else float("nan")
    return p, v


def confound_report(df: pd.DataFrame) -> None:
    print("=== Confound check: system_name x benchmark_name ===")
    print(pd.crosstab(df["system_name"], df["benchmark_name"]).to_string())
    print(
        "-> confounded except GAIA (run by both captain-agent and magentic-one) "
        "— the only benchmark-matched comparison point.\n"
    )


def role_association(df: pd.DataFrame) -> None:
    print("=== system_name x role (Orchestrator/Worker/Verification) ===")
    table = pd.crosstab(df["system_name"], df["role"])
    print(table.to_string())
    p, v = cramers_v(table)
    print(f"chi2 p-value={p:.4g}, Cramer's V={v:.3f}\n")

    print("Role share within each system:")
    print((pd.crosstab(df["system_name"], df["role"], normalize="index") * 100).round(1).to_string())
    print()


def step_position_report(df: pd.DataFrame) -> None:
    print("=== Failure step position (mistake_step / n_steps) by system ===")
    valid = df.dropna(subset=["position"])
    print(valid.groupby("system_name")["position"].describe().round(3).to_string())
    print()

    bucket = pd.cut(valid["position"], bins=[0, 1 / 3, 2 / 3, 1.0001], labels=["early", "mid", "late"])
    table = pd.crosstab(valid["system_name"], bucket)
    print("Bucketed early/mid/late (thirds of trajectory):")
    print(table.to_string())
    p, v = cramers_v(table)
    print(f"chi2 p-value={p:.4g}, Cramer's V={v:.3f}")
    print(
        "-> paper claims CaptainAgent (dynamic team) is dispersed, "
        "Magentic-One/SWE-Agent (fixed orchestrator) concentrate early. "
        "Compare 'early' share across systems above.\n"
    )


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_trace_elephant` first.")
    df = load_dataframe()
    print(f"Loaded {len(df)} traces, {df['system_name'].nunique()} systems.\n")

    confound_report(df)
    role_association(df)
    step_position_report(df)

    out_path = RESULTS_DIR / "trace_elephant_role_by_system.csv"
    pd.crosstab(df["system_name"], df["role"]).to_csv(out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
