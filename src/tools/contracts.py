"""Validated request/response contracts for offline image tool calls."""

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.camera.capture_plan import CapturePlan
from src.camera.capture_simulator import SimulationConfig
from src.computational.pipeline import PipelineConfig
from src.debugging.diagnosis import Fault, QualityLimits
from src.scene.analysis import SceneFeatures


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SourceSpec(Contract):
    simulation: SimulationConfig | None = None
    burst_id: str | None = None
    raw_manifest: Path | None = None

    @field_validator("burst_id")
    @classmethod
    def normalize_id(cls, value):
        return UUID(value).hex if value else None

    @model_validator(mode="after")
    def one_source(self):
        if sum(value is not None for value in (self.simulation, self.burst_id, self.raw_manifest)) != 1:
            raise ValueError("select exactly one simulation, Burst ID or local RAW manifest")
        return self


class LocalInput(Contract):
    artifact_root: Path


class SurveyInput(LocalInput):
    source: SourceSpec


class SurveyOutput(Contract):
    features: SceneFeatures
    input_artifact_id: str | None


class CaptureInput(LocalInput):
    simulation: SimulationConfig
    plan: CapturePlan


class ArtifactOutput(Contract):
    artifact_id: str


class ArtifactInput(LocalInput):
    artifact_id: str


class PhotographyInput(ArtifactInput):
    plan: CapturePlan
    config: PipelineConfig


class ValidationInput(ArtifactInput):
    fault: Fault
    limits: QualityLimits
