"""Schema for the MAST dataset (mcemri/MAST-Data).

Shared by the loader under ``data/error_categorization`` and by any experiment
reading the generated JSON files.

MAST traces are unstructured logs: a sample carries the whole log as a single
``raw_trajectory`` string and the loader does not split it into steps. The
optional ``trajectory`` field is filled in later by
``data/error_categorization/build_agent_behaviors.py`` for experiments that
need segmentation.
"""

from typing import List

from pydantic import BaseModel


class FailureMode(BaseModel):
    code: str
    name: str
    description: str


class FailureGroup(BaseModel):
    name: str
    failure_modes: List[FailureMode]


class Metadata(BaseModel):
    source: str
    paper: str
    repo: str
    note: str
    groups: List[FailureGroup]


class AgentBehavior(BaseModel):
    step: int
    content: str


class Sample(BaseModel):
    mas_name: str
    raw_trajectory: str
    trajectory: List[AgentBehavior] | None = None

    faults: List[str]


# MAST taxonomy (arXiv:2503.13657) - 14 failure modes, 3 groups
MAST_METADATA = Metadata(
    source="MAST: Multi-Agent System Failure Taxonomy",
    paper="Why Do Multi-Agent LLM Systems Fail? (arXiv:2503.13657)",
    repo="https://github.com/multi-agent-systems-failure-taxonomy/MAST",
    note=(
        "mast_annotation trong moi trace la nhan multi-label cap TOAN TRACE, "
        "khong gan voi step/agent cu the nao trong trajectory."
    ),
    groups=[
        FailureGroup(
            name="System Design Issues",
            failure_modes=[
                FailureMode(
                    code="1.1",
                    name="Disobey Task Specification",
                    description="Agent khong tuan thu yeu cau/rang buoc ro rang cua task (sai dinh dang, bo qua chi dan tuan tu, sai tool contract).",
                ),
                FailureMode(
                    code="1.2",
                    name="Disobey Role Specification",
                    description="Agent hanh dong ngoai pham vi vai tro duoc phan cong, lan sang trach nhiem cua agent khac.",
                ),
                FailureMode(
                    code="1.3",
                    name="Step Repetition",
                    description="Agent lap lai hanh dong/buoc da thuc hien ma khong tien trien them, do theo doi trang thai kem.",
                ),
                FailureMode(
                    code="1.4",
                    name="Loss of Conversation History",
                    description="Agent mat/quen ngu canh truoc do (context bi cat hoac quay lai trang thai cu).",
                ),
                FailureMode(
                    code="1.5",
                    name="Unaware of Termination Conditions",
                    description="Agent khong nhan ra khi nao task da hoan thanh va nen dung, gay lang phi tai nguyen/vong lap thua.",
                ),
            ],
        ),
        FailureGroup(
            name="Inter-Agent Misalignment",
            failure_modes=[
                FailureMode(
                    code="2.1",
                    name="Conversation Reset",
                    description="Hoi thoai bi khoi dong lai mot cach khong can thiet, lam mat tien do da dat duoc.",
                ),
                FailureMode(
                    code="2.2",
                    name="Fail to Ask for Clarification",
                    description="Agent tu gia dinh thay vi hoi lai khi thong tin thieu/mo ho.",
                ),
                FailureMode(
                    code="2.3",
                    name="Task Derailment",
                    description="Agent di lech khoi muc tieu/task chinh, sa vao hanh dong khong lien quan.",
                ),
                FailureMode(
                    code="2.4",
                    name="Information Withholding",
                    description="Agent khong chia se thong tin quan trong cho agent khac, lam giam hieu qua phoi hop chung.",
                ),
                FailureMode(
                    code="2.5",
                    name="Ignored Other Agent's Input",
                    description="Agent bo qua/khong xem xet phan hoi hay de xuat tu agent khac.",
                ),
                FailureMode(
                    code="2.6",
                    name="Reasoning-Action Mismatch",
                    description="Agent xac dinh dung buoc tiep theo trong reasoning nhung lai thuc hien hanh dong khac/khong lien quan.",
                ),
            ],
        ),
        FailureGroup(
            name="Task Verification",
            failure_modes=[
                FailureMode(
                    code="3.1",
                    name="Premature Termination",
                    description="Agent ket thuc task qua som, truoc khi muc tieu thuc su dat duoc.",
                ),
                FailureMode(
                    code="3.2",
                    name="No or Incomplete Verification",
                    description="Thieu buoc kiem tra ket qua, hoac verification co ton tai nhung khong bao phu day du cac khia canh can thiet.",
                ),
                FailureMode(
                    code="3.3",
                    name="Incorrect Verification",
                    description='Agent xac nhan sai ket qua ("hallucinating success") - bao thanh cong du output thuc te sai/chua dat.',
                ),
            ],
        ),
    ],
)


def render_taxonomy(metadata: Metadata = None) -> str:
    metadata = metadata or MAST_METADATA
    lines = []
    for group in metadata.groups:
        lines.append(f"## {group.name}")
        for mode in group.failure_modes:
            lines.append(f"- {mode.code} {mode.name}: {mode.description}")
    return "\n".join(lines)
