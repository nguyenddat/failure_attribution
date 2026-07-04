from typing import Optional

from pydantic import BaseModel, Field


class StepByStepInput(BaseModel):
    problem: str = Field(..., description="Cau hoi")
    ground_truth: Optional[str] = Field(..., description="Cau tra loi dung")
    current_conversation_history: str = Field(..., description="Lich su tro chuyen hien tai")
    idx: int = Field(..., description="Buoc hien tai")
    agent_name: str = Field(..., description="Ten agent hien tai")


class AllAtOnceInput(BaseModel):
    problem: str = Field(..., description="Cau hoi")
    ground_truth: Optional[str] = Field(..., description="Cau tra loi dung")
    chat_content: str = Field(..., description="Lich su tro chuyen")


class BinarySearchInput(BaseModel):
    problem: str = Field(..., description="Cau hoi")
    ground_truth: Optional[str] = Field(..., description="Cau tra loi dung")
    chat_segment_history: str = Field(..., description="Lich su tro chuyen trong phan doan hien tai")
    start_step: int = Field(..., description="Bat dau phan doan")
    end_step: int = Field(..., description="Ket thuc phan doan")


class MultiStepInput(BaseModel):
    problem: str = Field(..., description="Cau hoi")
    ground_truth: Optional[str] = Field(..., description="Cau tra loi dung")
    previous_evaluation_history: str = Field(..., description="Lich su tom tat tu cac lan danh gia truoc")
    current_steps: str = Field(..., description="Cac step hien tai duoc danh gia cung luc")
    num_steps: int = Field(..., description="So step hien tai duoc dua vao cung luc")


class MultiStepCoarseInput(BaseModel):
    problem: str = Field(..., description="Cau hoi")
    ground_truth: Optional[str] = Field(..., description="Cau tra loi dung")
    previous_summaries: str = Field(..., description="Tom tat coarse scan tu cac chunk truoc")
    current_chunk: str = Field(..., description="Chunk hien tai can coarse scan")


class MultiStepRefineInput(BaseModel):
    problem: str = Field(..., description="Cau hoi")
    ground_truth: Optional[str] = Field(..., description="Cau tra loi dung")
    coarse_scan_context: str = Field(..., description="Thong tin tom tat va nghi ngo tu tang coarse")
    refinement_window: str = Field(..., description="Vung hep can refine de chot step va agent")


class Metadata(BaseModel):
    model_name: str = Field(..., description="Ten mo hinh")
    method: str = Field(..., description="Phuong phap")
    with_gt: bool = Field(..., description="Truyen gt vao prompt hay khong?")


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
