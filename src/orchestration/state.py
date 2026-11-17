"""Typed workflow input and JSON-compatible shared state."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import Field

from src.debugging.diagnosis import QualityLimits
from src.tools.contracts import Contract, SourceSpec


class WorkflowRequest(Contract):
    user_request: str = Field(min_length=1, max_length=4000)
    source: SourceSpec
    brain: Literal["offline_mock", "deepseek"] = "offline_mock"
    max_iterations: int = Field(default=2, ge=0, le=8)
    max_experiments: int = Field(default=3, ge=1, le=9)
    tool_timeout_s: float = Field(default=45, gt=0, le=600)
    tool_retries: int = Field(default=1, ge=0, le=3)
    quality_limits: QualityLimits = Field(default_factory=QualityLimits)
    initial_parameters: dict = Field(default_factory=dict)
    reference_frame: int | None = Field(default=None, ge=0)
    enable_feedback: bool = True
    enable_memory: bool = True


class CameraState(TypedDict, total=False):
    task_id: str
    user_request: str
    request: dict
    task_type: str
    intent: dict
    brain_decision: dict
    input_artifacts: list[str]
    capture_plan: dict
    capture_survey: dict
    scene_features: dict
    motion_results: dict
    photography_results: dict
    iq_results: dict
    debugging_results: Annotated[list[dict], operator.add]
    iteration_count: int
    experiment_count: int
    execution_plan: list[str]
    pipeline_config: dict
    proposed_config: dict | None
    faults: list[str]
    attempts: Annotated[list[dict], operator.add]
    agent_trace: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]
    memory_reviews: Annotated[list[dict], operator.add]
    final_artifact_id: str | None
    final_rank: list[float]
    report_id: str
    status: str
    termination_reason: str
