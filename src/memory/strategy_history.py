"""Record outcome evidence and expose compact provenance for agent decisions."""

import json
from pathlib import Path

from src.storage.artifact_store import ArtifactStore

from .experiment_memory import Experiment, ExperimentMemory

FAULT_METRICS = {"registration_failure": "alignment_error", "ghosting": "ghosting_score"}


def citation(row: dict) -> dict:
    return {key: row[key] for key in ("group_id", "number", "output_artifact_id", "scene_distance")}


def record_verified_repairs(memory: ExperimentMemory, group_id: str, before: dict, after: dict,
                            diagnoses: list[dict]) -> None:
    for diagnosis in diagnoses:
        fault = diagnosis["fault"]
        if diagnosis["validation_result"] != "confirmed" or fault in after["faults"]:
            continue
        name = FAULT_METRICS[fault]
        left = before["iq"]["metrics"][name]["value"]
        right = after["iq"]["metrics"][name]["value"]
        if left is not None and right is not None and 0 <= right < left:
            memory.record_repair(group_id, before["iteration"], after["iteration"], fault, left, right)


def import_reports(memory: ExperimentMemory, artifact_root: Path) -> dict:
    """Backfill pre-memory LangGraph reports with still-available source and output artifacts."""
    photos = ArtifactStore(artifact_root / "photography")
    bursts = ArtifactStore(artifact_root / "burst")
    inserted, skipped = 0, []
    for path in sorted((artifact_root / "reports").glob("*/metadata.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            attempts = payload.get("attempts", [])
            if payload.get("artifact_type") != "camera_workflow_report" or not attempts:
                continue
            input_id = payload["input_artifacts"][0]
            source = bursts.load_metadata(input_id)
            simulation = source.get("simulation_config")
            capture_plan = payload["capture_plan"]
            source_kind = "synthetic_survey" if "offline simulation" in capture_plan["constraints"] else "existing_raw"
            diagnostics = {record["iteration"]: record["diagnoses"] for record in payload["debugging_results"]}
            records = []
            for attempt in attempts:
                output_id = attempt["artifact_id"]
                metadata = photos.load_metadata(output_id)
                iq = attempt["iq"]
                if metadata["source"]["input_artifact_id"] != input_id or iq["artifact_id"] != output_id:
                    raise ValueError("report does not match the input and output artifact IDs")
                records.append(Experiment(
                    group_id=payload["task_id"], number=attempt["iteration"], kind="workflow", mode=payload["task_type"],
                    source_kind=source_kind, status="needs_review" if attempt["faults"] else "passed", input_artifact_id=input_id,
                    output_artifact_id=output_id, seed=simulation["seed"] if simulation else None,
                    runtime_ms=iq["metrics"]["runtime"]["value"], survey=payload["capture_survey"], scene=payload["scene_features"],
                    capture_plan=capture_plan, parameters=attempt["parameters"], metrics=iq,
                    configuration={"imported_report_id": payload["artifact_id"], "simulation": simulation,
                                   "user_request": payload["user_request"], "quality_limits": payload["quality_limits"]},
                    diagnoses=diagnostics.get(attempt["iteration"], []),
                ))
            inserted += sum(memory.record(record) for record in records)
            for before, after in zip(attempts, attempts[1:]):
                record_verified_repairs(memory, payload["task_id"], before, after, diagnostics.get(before["iteration"], []))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            skipped.append({"report": str(path), "reason": str(exc)})
    return {"inserted": inserted, "skipped": skipped, "database": str(memory.path)}
