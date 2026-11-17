"""Turn suspected faults into tool-validated diagnoses and parameter revisions."""

from pathlib import Path

from src.computational.pipeline import PipelineConfig
from src.debugging.diagnosis import Fault, QualityLimits
from src.debugging.repair_strategies import propose_revision
from src.memory.experiment_memory import ExperimentMemory
from src.memory.retrieval import similar_repairs
from src.memory.strategy_history import citation
from src.tools.contracts import ValidationInput
from src.tools.registry import ToolRunner


def debug(runner: ToolRunner, artifact_root: Path, artifact_id: str, faults: list[Fault],
          limits: QualityLimits, current: PipelineConfig):
    diagnoses = [runner.call("validate_diagnosis", ValidationInput(artifact_root=artifact_root, artifact_id=artifact_id,
                                                                  fault=fault, limits=limits)) for fault in faults]
    return diagnoses, propose_revision(current, diagnoses)


def consult_history(revised: PipelineConfig | None, diagnoses: list, scene: dict, memory: ExperimentMemory,
                    *, mode: str, source_kind: str, task_id: str) -> tuple[PipelineConfig | None, list[dict]]:
    reviews = []
    for diagnosis in diagnoses:
        if diagnosis.validation_result != "confirmed":
            continue
        hits = similar_repairs(memory, scene, mode=mode, fault=diagnosis.fault,
                               source_kind=source_kind, exclude_group=task_id)
        evidence = diagnosis.evidence
        actual = evidence["metric"]["value"] if diagnosis.fault == "ghosting" else evidence["measured_max_residual"]
        prior = hits[0] if hits else None
        applied = False
        comparable = actual is not None and prior is not None and 0.5 <= actual / prior["before_value"] <= 2
        if revised is not None and comparable and diagnosis.fault == "ghosting" and evidence["coherent_motion_support_pixels"] >= 32:
            historical_threshold = prior["parameters"]["motion_threshold"]
            severity_scale = (prior["before_value"] / actual) ** 0.25
            threshold = max(0.02, min(revised.motion_threshold, historical_threshold * severity_scale))
            applied = threshold < revised.motion_threshold
            if applied:
                revised = PipelineConfig.model_validate(revised.model_dump() | {"motion_threshold": threshold})
        reviews.append({"agent": "camera_debugging", "fault": diagnosis.fault, "candidates": [citation(row) for row in hits],
                        "current_measurement": actual, "comparable_severity": comparable, "applied": applied,
                        "reason": "history considered after live diagnosis; revised output requires fresh IQ validation"})
    return revised, reviews
