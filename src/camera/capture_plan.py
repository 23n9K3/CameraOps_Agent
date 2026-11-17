"""Validated offline capture instructions in physical units."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

ExposureSeconds = Annotated[float, Field(gt=0, le=1)]
AnalogGain = Annotated[float, Field(ge=1, le=32)]


class CapturePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    capture_mode: Literal["burst", "bracket", "motion_aware"] = "burst"
    exposure_times: tuple[ExposureSeconds, ...] = Field(min_length=1, max_length=16)
    gains: tuple[AnalogGain, ...] = Field(min_length=1, max_length=16)
    inter_frame_gap_s: float = Field(default=0.012, ge=0, le=1)
    reference_frame: int = Field(default=0, ge=0)
    selected_frames: tuple[int, ...] | None = None
    fusion_strategy: Literal["night_temporal", "hdr_exposure", "motion_reference"] | None = None
    constraints: tuple[str, ...] = ()
    rationale: str = ""

    @computed_field
    @property
    def frame_count(self) -> int:
        return len(self.exposure_times)

    @model_validator(mode="before")
    @classmethod
    def validate_serialized_count(cls, data):
        if isinstance(data, dict) and "frame_count" in data:
            data = data.copy()
            count = data.pop("frame_count")
            times = data.get("exposure_times")
            if type(count) is not int or not isinstance(times, (list, tuple)) or count != len(times):
                raise ValueError("frame_count must equal the number of exposures")
        return data

    @model_validator(mode="after")
    def validate_frames(self) -> "CapturePlan":
        if len(self.gains) != self.frame_count:
            raise ValueError("provide one analog gain per exposure")
        if self.reference_frame >= self.frame_count:
            raise ValueError("reference_frame must identify an existing zero-based frame")
        if self.selected_frames is not None:
            if tuple(sorted(set(self.selected_frames))) != self.selected_frames:
                raise ValueError("selected_frames must be unique and in capture order")
            if any(not 0 <= index < self.frame_count for index in self.selected_frames):
                raise ValueError("selected_frames must identify existing zero-based frames")
            if self.reference_frame not in self.selected_frames:
                raise ValueError("selected_frames must include the reference frame")
        return self
