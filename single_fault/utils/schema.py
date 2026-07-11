from typing import Optional

from pydantic import BaseModel, Field

class AllAtOnceInput(BaseModel):
    problem: str
    chat_content: str

class StepByStepInput(BaseModel):
    problem: str
    current_step_content: str
    chat_content: str


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
