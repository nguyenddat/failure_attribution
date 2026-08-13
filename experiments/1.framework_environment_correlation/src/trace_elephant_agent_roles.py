"""Role classification for TraceElephant's ``mistake_agent`` values.

``mistake_agent`` names are system-specific (captain-agent's
"WebServing_Expert" vs magentic-one's "WebSurfer" vs swe-agent's
"str_replace_editor") — there is no shared taxonomy like MAST's 14 codes.
This mirrors the 2-way split the TraceElephant paper itself uses in
§4.2.2 ("Failure Responsible Agents"): agents that interact with external
environments/tools (>50% of failures per the paper) vs the
Orchestrator/planner (18-29%).

captain-agent also has 2 verification-specific roles with no equivalent in
the other 2 systems (DataVerification_Expert, Verification_Expert) — kept
as a 3rd "Verification" bucket rather than folded into Worker, since
collapsing them would hide a role type unique to that system.
"""

ROLE = {
    # captain-agent
    "Orchestrator": "Orchestrator",
    "WebServing_Expert": "Worker",
    "InformationExtraction_Expert": "Worker",
    "CSVHandling_Expert": "Worker",
    "PythonProgramming_Expert": "Worker",
    "DataVerification_Expert": "Verification",
    "Verification_Expert": "Verification",
    # magentic-one
    "WebSurfer": "Worker",
    "FileSurfer": "Worker",
    "Coder": "Worker",
    # swe-agent
    "str_replace_editor": "Worker",
    "bash": "Worker",
}

ROLE_ORDER = ["Orchestrator", "Worker", "Verification"]


def classify(mistake_agent: str) -> str:
    return ROLE.get(mistake_agent, "Unknown")
