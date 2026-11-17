"""Validation rules for measured registration and ghosting symptoms."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.iq.evaluation import IQResult

Fault = Literal["registration_failure", "ghosting"]


class QualityLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_alignment_error: float = Field(default=0.035, gt=0, le=1)
    max_ghosting_score: float = Field(default=0.012, gt=0, le=1)
    min_overlap_ratio: float = Field(default=0.75, gt=0, le=1)


class Diagnosis(BaseModel):
    fault: Fault
    hypothesis: str
    evidence: dict
    validation_result: Literal["confirmed", "rejected", "inconclusive"]
    confidence: float = Field(ge=0, le=1)
    confidence_basis: str
    recommended_action: str


def _alignment_violations(iq: IQResult, limits: QualityLimits) -> list[dict]:
    return [row for row in iq.registration_checks if row["overlap_ratio"] < limits.min_overlap_ratio or
            (row["alignment_residual"] is not None and row["alignment_residual"] > limits.max_alignment_error)]


def detected_faults(iq: IQResult, limits: QualityLimits) -> list[Fault]:
    candidates: list[Fault] = []
    if _alignment_violations(iq, limits):
        candidates.append("registration_failure")
    ghosting = iq.metrics["ghosting_score"]
    if ghosting.valid and ghosting.value > limits.max_ghosting_score:
        candidates.append("ghosting")
    return candidates


def validate_fault(fault: Fault, iq: IQResult, limits: QualityLimits) -> Diagnosis:
    if fault == "registration_failure":
        offenders = _alignment_violations(iq, limits)
        metrics = iq.metrics["alignment_error"]
        available = metrics.valid or bool(offenders)
        support = max([0.0, *[
            max((row["alignment_residual"] or 0) / limits.max_alignment_error,
                limits.min_overlap_ratio / max(row["overlap_ratio"], 1e-6)) for row in offenders
        ]])
        evidence = {"artifact_id": iq.artifact_id, "offending_frames": offenders,
                    "measured_max_residual": metrics.value, "residual_limit": limits.max_alignment_error,
                    "min_overlap_ratio": limits.min_overlap_ratio}
        hypothesis = "Global registration did not reach sufficient alignment on comparable image support."
        action = "Enable global registration with phase-correlation initialization followed by ECC."
        confirmed = bool(offenders)
    else:
        metrics = iq.metrics["ghosting_score"]
        available = metrics.valid
        support = (metrics.value or 0) / limits.max_ghosting_score
        confirmed = available and support > 1
        evidence = {"artifact_id": iq.artifact_id, "metric": metrics.model_dump(mode="json"),
                    "coherent_motion_support_pixels": iq.ghosting_support_pixels, "limit": limits.max_ghosting_score}
        hypothesis = "The output retains structures that disagree with the reference in coherent motion regions."
        action = "Enable deghosting, protect the reference in dynamic regions and tighten motion thresholds."
    return Diagnosis(fault=fault, hypothesis=hypothesis, evidence=evidence,
                     validation_result="confirmed" if confirmed else ("rejected" if available else "inconclusive"),
                     confidence=min(1.0, support / 2) if confirmed else 0.0,
                     confidence_basis="Evidence strength is min(1, violation-to-threshold ratio / 2); not a calibrated probability.",
                     recommended_action=action)
