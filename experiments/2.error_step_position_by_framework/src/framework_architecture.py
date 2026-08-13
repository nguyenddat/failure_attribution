"""Topology classification for every framework/pseudo-framework in step_positions.csv.

Sourced from the standardized taxonomy in
`experiments/0.framework_topology_taxonomy/finding_notes.md`, not own
guess (that was the earlier version of this file — corrected to match
Experiment 1's fix, see `experiments/1.framework_environment_correlation/
finding_notes.md` "Phân loại topology"):

- Captain-Agent (TraceElephant): Centralized — exp 0 classifies CaptainAgent
  as Centralized (dynamic): a Captain hub assembles + directs the team at
  runtime, still a single hub, not a fixed multi-level hierarchy. Was
  wrongly Hierarchical in the previous version of this file.
- Magentic-One (TraceElephant + Who&When): Centralized — explicit single
  Orchestrator hub, per exp 0.
- SWE-Agent (TraceElephant): Single-agent — exp 0: single control loop, not
  multi-agent coordination at all. Was wrongly folded into Centralized in
  the previous version of this file.
- OpenDeepResearch (TRAIL/GAIA): Hierarchical — exp 0, sourced from
  note_trail.md quoting the TRAIL paper directly: "hierarchical
  multi-agent (OpenDeepResearch...)".
- CodeAct (TRAIL/SWE-bench): Single-agent — exp 0, sourced from
  note_trail.md: "single-agent (CodeAct + sandbox...)". Was wrongly
  Centralized in the previous version of this file.
- AgentDebug-ReAct (AgentErrorBench): Single-agent — exp 0: fixed 4-module
  (memory/reflection/plan/action) single-agent loop, no inter-agent
  coordination at all. Was wrongly Centralized in the previous version of
  this file.
- MiroFlow / OAgent (TELBENCH): Centralized — exp 0 marks these "?"
  (unresolved, no architecture description found for TELBENCH's systems).
  Kept as Centralized here (own guess, unchanged) since exp 0 gives no
  better answer — weakest-evidence entries in this bucket.

Decentralized bucket is empty across these 5 datasets — none use a
peer-to-peer/consensus/graph-based system (that pattern only showed up in
AEGIS's llm_debate/dylan/macnet, which is excluded from this experiment —
no usable step-level label).
"""

ARCHITECTURE = {
    "Captain-Agent": "Centralized",
    "OpenDeepResearch (TRAIL)": "Hierarchical",
    "Magentic-One": "Centralized",
    "SWE-Agent": "Single-agent",
    "CodeAct (TRAIL)": "Single-agent",
    "AgentDebug-ReAct": "Single-agent",
    "MiroFlow": "Centralized",
    "OAgent": "Centralized",
}

ARCH_ORDER = ["Hierarchical", "Centralized", "Decentralized", "Single-agent"]
