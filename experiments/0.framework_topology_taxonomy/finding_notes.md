# Experiment 0: framework/topology taxonomy — nền cho experiment sau

2 trục:
- **Topology** (Hierarchical / Pipeline / Centralized / Decentralized /
Single-agent / Variable — thư viện không cố định 1 topology) ×
- **Dynamic?** (team/topology sinh runtime theo task, hay cố định).

| Framework | Dataset dùng | Topology | Dynamic? |
|---|---|---|---|
| MetaGPT | MAST | Pipeline | static |
| ChatDev | MAST | Hierarchical | static |
| HyperAgent | MAST | Hierarchical | static |
| OpenManus | MAST | Hierarchical | static |
| AppWorld | MAST | Centralized | static |
| AG2/AutoGen (raw `mas_name`) | MAST | Variable | config-dependent |
| CaptainAgent (template trong AG2) | Who&When, TraceElephant | Centralized | dynamic |
| Magentic-One | MAST, Who&When, AEGIS, TraceElephant | Centralized | static |
| MacNet | AEGIS | Variable | config-dependent |
| DyLAN | AEGIS | Decentralized | dynamic |
| LLM-Debate | AEGIS | Decentralized | static |
| AgentVerse | AEGIS | Hierarchical | static |
| SmolAgents (thư viện) | AEGIS | Variable | config-dependent |
| HF OpenDeepResearch (build trên SmolAgents, manager+search) | TRAIL | Hierarchical | static |
| CodeAct agent | TRAIL | Single-agent | static |
| SWE-Agent | TraceElephant | Single-agent | static |
| MiroFlow | TELBENCH | ? | ? |
| OAgent | TELBENCH | ? | ? |
| custom 4-module (Memory/Reflection/Planning/Action) | AgentErrorBench | Single-agent | static |