"""Convert confirmed diagnoses into validated local pipeline changes."""

from src.computational.pipeline import PipelineConfig

from .diagnosis import Diagnosis


def propose_revision(current: PipelineConfig, diagnoses: list[Diagnosis]) -> PipelineConfig | None:
    changes = {}
    for record in diagnoses:
        if record.validation_result != "confirmed":
            continue
        if record.fault == "registration_failure":
            changes.update(registration=True, registration_method="phase_ecc")
        elif record.fault == "ghosting":
            changes.update(deghost=True, protect_motion_reference=True,
                           motion_threshold=min(current.motion_threshold, 0.045),
                           flow_threshold_px=min(current.flow_threshold_px, 0.8))
    revised = PipelineConfig.model_validate(current.model_dump() | changes)
    return revised if revised != current else None
