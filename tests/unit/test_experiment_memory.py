import pytest

from src.agents import camera_debugging_agent, capture_planner_agent, supervisor
from src.computational.pipeline import PipelineConfig
from src.debugging.diagnosis import Diagnosis
from src.memory.experiment_memory import Experiment, ExperimentMemory
from src.memory.retrieval import similar_strategies
from src.scene.analysis import SceneFeatures


def _observations(**changes):
    return SceneFeatures.model_validate({"brightness_median": 0.24, "shadow_ratio": 0.18, "highlight_ratio": 0.03,
        "luminance_p99": 0.42, "laplacian_variance": 0.012, "histogram_16": [200] * 16,
        "motion_fraction": 0.012, "moving_flow_p90_px": 2.0,
        "exposures": [], "saturation_by_frame": [], "frame_sharpness": []} | changes)


def _store(memory, group_id, scene, **values):
    defaults = {"group_id": group_id, "number": 0, "kind": "workflow", "mode": "night", "status": "passed",
                "source_kind": "synthetic_survey", "input_artifact_id": "raw-" + group_id,
                "output_artifact_id": "photo-" + group_id, "runtime_ms": 24.0, "survey": scene, "scene": scene}
    memory.record(Experiment.model_validate(defaults | values))


def test_memory_persists_filters_and_rejects_incompatible_scene(tmp_path):
    path = tmp_path / "memory.sqlite3"
    memory = ExperimentMemory(path)
    scene = _observations().model_dump()
    _store(memory, "close", scene)
    _store(memory, "bright_motion", _observations(brightness_median=0.9, motion_fraction=0.5).model_dump())
    _store(memory, "workflow_failed", scene, status="needs_review")
    _store(memory, "different_source", scene, source_kind="existing_raw")
    _store(memory, "different_mode", scene, mode="hdr")
    reopened = ExperimentMemory(path)
    assert [row["group_id"] for row in reopened.history(artifact_id="photo-close")] == ["close"]
    result = similar_strategies(reopened, scene, mode="night", source_kind="synthetic_survey")
    assert [row["group_id"] for row in result] == ["close"]
    assert result[0]["scene_distance"] == 0
    assert len(reopened.history(mode="night", status="needs_review")) == 1


def test_planner_uses_only_validated_history_advice(tmp_path):
    memory = ExperimentMemory(tmp_path / "memory.sqlite3")
    scene = _observations()
    _store(memory, "previous", scene.model_dump(), capture_plan={"exposure_times": [0.002] * 7, "gains": [8] * 7})
    plan = capture_planner_agent.plan_capture(supervisor.interpret_goal("night denoise"), scene, simulated=True, gap=0.015)
    revised, accepted = capture_planner_agent.consult_history(plan, scene, memory, source_kind="synthetic_survey",
                                                             task_id="current", max_frames=8)
    assert accepted["applied"] and revised.frame_count == 7
    assert revised.exposure_times == (plan.exposure_times[0],) * 7
    assert revised.gains == (plan.gains[0],) * 7
    unchanged, rejected = capture_planner_agent.consult_history(plan, scene, memory, source_kind="synthetic_survey",
                                                                 task_id="current", max_frames=6)
    assert rejected["candidates"] and not rejected["applied"] and unchanged == plan


def test_debugging_history_requires_confirmed_comparable_live_evidence(tmp_path):
    memory = ExperimentMemory(tmp_path / "memory.sqlite3")
    scene = _observations().model_dump()
    _store(memory, "repair", scene, status="needs_review")
    _store(memory, "repair", scene, number=1, output_artifact_id="repaired", parameters={"motion_threshold": 0.031})
    memory.record_repair("repair", 0, 1, "ghosting", 0.03, 0.007)
    proposed = PipelineConfig(mode="night", motion_threshold=0.045)
    common = dict(fault="ghosting", hypothesis="temporal mixing", confidence=0.9, confidence_basis="test",
                  recommended_action="protect reference")
    diagnosed = Diagnosis(**common, validation_result="confirmed",
                          evidence={"metric": {"value": 0.039}, "coherent_motion_support_pixels": 200})
    changed, reviews = camera_debugging_agent.consult_history(proposed, [diagnosed], scene, memory, mode="night",
                                                               task_id="new", source_kind="synthetic_survey")
    assert reviews[0]["applied"]
    assert changed.motion_threshold == pytest.approx(0.031 * (0.03 / 0.039)**0.25)
    unconfirmed = diagnosed.model_copy(update={"validation_result": "rejected"})
    unchanged, reviews = camera_debugging_agent.consult_history(proposed, [unconfirmed], scene, memory, mode="night",
                                                                 task_id="new", source_kind="synthetic_survey")
    assert unchanged == proposed and not reviews
