from typing import Optional

from pydantic import BaseModel, Field


class AllAtOnceInput(BaseModel):
    problem: str
    chat_content: str


class StepByStepInput(BaseModel):
    problem: str
    current_step_content: str
    chat_content: str


class TaskDecompositionInput(BaseModel):
    problem: str
    chat_content: str
    exemplars: str


class SubtaskAlignmentInput(BaseModel):
    problem: str
    chat_content: str
    subtasks: str


class Metadata(BaseModel):
    model_name: str = Field(..., description="Ten mo hinh")
    method: str = Field(..., description="Phuong phap")


class AccuracyMetrics(BaseModel):
    gt_agent: str = Field(..., description="Ten agent dung")
    gt_step: int = Field(..., description="Buoc dung")
    pred_agent: str = Field(..., description="Ten agent du doan")
    pred_step: int = Field(..., description="Buoc du doan")
    agent_accuracy: float = Field(..., description="Do chinh xac cua agent")
    step_accuracy: float = Field(..., description="Do chinh xac cua buoc")


class CostMetrics(BaseModel):
    num_input_steps: int = Field(int, description="So buoc dau vao")
    latency: float = Field(float, description="Thoi gian chay 1 sample")
    input_tokens: int = Field(int, description="So luong token dau vao")
    output_tokens: int = Field(int, description="So luong token dau ra")
    input_cost: float = Field(float, description="Chi phi dau vao")
    output_cost: float = Field(float, description="Chi phi dau ra")
    total_cost: float = Field(float, description="Tong chi phi")

    class Config:
        populate_by_name = True
        extra = "ignore"


class StepRecord(BaseModel):
    step_id: int = Field(
        ..., description="Chi so step da duoc chuan hoa theo thu tu trajectory"
    )
    agent_name: str = Field(..., min_length=1, description="Ten agent cua step")
    content: str = Field(..., description="Noi dung step")
    raw_step_id: int = Field(..., description="Chi so step goc trong log")


class NormalizedTrajectory(BaseModel):
    steps: list[StepRecord] = Field(..., description="Danh sach step da duoc chuan hoa")


class TrajectoryIntakeArtifact(BaseModel):
    question: str = Field(..., description="Task instruction cua case")
    ground_truth: Optional[str] = Field(default=None, description="Ground truth neu co")
    steps: list[StepRecord] = Field(..., description="Trajectory da duoc chuan hoa")
    total_steps: int = Field(..., ge=0, description="Tong so step sau chuan hoa")
    mistake_step: Optional[int] = Field(default=None, description="Step loi goc neu co")
    mistake_agent: Optional[str] = Field(
        default=None, description="Agent loi goc neu co"
    )
