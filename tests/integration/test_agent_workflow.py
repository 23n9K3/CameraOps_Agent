from pathlib import Path

import numpy as np
import pytest

from scripts.run_agent_feedback_demo import make_shifted_raw_manifest
from src.camera.capture_plan import CapturePlan
from src.camera.capture_simulator import SimulationConfig
from src.memory.experiment_memory import ExperimentMemory
from src.memory.strategy_history import import_reports
from src.orchestration.graph import build_graph, run_workflow
from src.orchestration.state import WorkflowRequest
from src.storage.artifact_store import ArtifactStore
from src.tools.contracts import SourceSpec, SurveyInput
from src.tools.registry import ToolFailure, ToolRunner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def simulation():
    return SimulationConfig.from_file(ROOT / "configs/synthetic_burst.yaml")


def _report(root, state):
    return ArtifactStore(root / "reports").load_metadata(state["report_id"])


def test_langgraph_ghosting_validation_revises_algorithm_and_scores_real_outputs(tmp_path, simulation):
    graph = build_graph(tmp_path).get_graph()
    assert any(edge.source == "strategy_revision" and edge.target == "photography" for edge in graph.edges)
    request = WorkflowRequest(user_request="处理运动行人，减少鬼影和拖影。", source=SourceSpec(simulation=simulation),
                              initial_parameters={"deghost": False}, max_iterations=1, max_experiments=2)
    state = run_workflow(request, tmp_path)
    report = _report(tmp_path, state)
    assert report["status"] == "completed", report
    assert report["experiment_count"] == 2 and report["iteration_count"] == 1
    plan = CapturePlan.model_validate(report["capture_plan"])
    assert plan.frame_count == 4
    frames = ArtifactStore(tmp_path / "burst").load_metadata(report["input_artifacts"][0])["frames"]
    assert [row["exposure_s"] for row in frames] == list(plan.exposure_times)
    assert [row["analog_gain"] for row in frames] == list(plan.gains)
    assert plan.fusion_strategy == "motion_reference"
    before, after = report["attempts"]
    assert before["parameters"]["mode"] == after["parameters"]["mode"] == "motion"
    assert before["parameters"]["deghost"] is False and after["parameters"]["deghost"] is True
    assert before["artifact_id"] != after["artifact_id"] == report["final_artifact_id"]
    validation = next(row for row in report["debugging_results"][0]["diagnoses"] if row["fault"] == "ghosting")
    assert validation["validation_result"] == "confirmed" and validation["evidence"]["artifact_id"] == before["artifact_id"]
    assert validation["confidence"] == pytest.approx(min(1, validation["evidence"]["metric"]["value"] /
                                                          (2 * validation["evidence"]["limit"])))
    scores = report["repair_comparisons"][0]["metrics"]["ghosting_score"]
    assert scores["after"] < 0.5 * scores["before"]
    assert after["iq"]["metrics"]["psnr"]["valid"] and after["iq"]["metrics"]["ssim"]["valid"]
    assert not after["iq"]["metrics"]["mtf50"]["valid"]
    calls = [call for event in report["agent_trace"] for call in event["tool_calls"]]
    assert any(call["tool"] == "validate_diagnosis" and call["artifact_id"] == before["artifact_id"] for call in calls)
    memory = ExperimentMemory(tmp_path / "experiments.sqlite3")
    recorded = memory.history(group_id=report["task_id"])
    assert {row["status"] for row in recorded} == {"passed", "needs_review"}
    assert next(row for row in recorded if row["number"] == 0)["diagnoses"][0]["fault"] == "ghosting"
    assert memory.repairs(mode="motion", fault="ghosting", source_kind="synthetic_survey")[0]["after_value"] == scores["after"]
    recovered = ExperimentMemory(tmp_path / "recovered.sqlite3")
    assert import_reports(recovered, tmp_path)["inserted"] == 2
    assert import_reports(recovered, tmp_path)["inserted"] == 0
    assert recovered.repairs(mode="motion", fault="ghosting", source_kind="synthetic_survey")[0]["after_value"] == scores["after"]


def test_langgraph_detects_measured_misregistration_and_reprocesses_file_sequence(tmp_path):
    manifest = make_shifted_raw_manifest(tmp_path / "input")
    request = WorkflowRequest(user_request="Reduce motion blur on these RAW frames.", source=SourceSpec(raw_manifest=manifest),
                              initial_parameters={"registration": False}, max_iterations=1, max_experiments=2)
    report = _report(tmp_path, run_workflow(request, tmp_path))
    assert report["status"] == "completed", report
    before, after = report["attempts"]
    assert "registration_failure" in before["faults"]
    assert after["parameters"]["registration"] and after["parameters"]["registration_method"] == "phase_ecc"
    checks = after["iq"]["registration_checks"]
    assert checks[0]["alignment_residual"] < 0.012
    assert report["repair_comparisons"][0]["metrics"]["alignment_error"]["before"] > 0.035
    transform = ArtifactStore(tmp_path / "photography").load_array(after["artifact_id"], "transforms")[1]
    np.testing.assert_allclose(transform[:, 2], [12, -8], atol=0.35)
    assert after["iq"]["metrics"]["psnr"]["value"] is None
    assert after["iq"]["metrics"]["ssim"]["value"] is None


def test_experiment_limit_still_validates_fault_without_reprocessing(tmp_path, simulation):
    request = WorkflowRequest(user_request="motion and ghosting", source=SourceSpec(simulation=simulation),
                              initial_parameters={"deghost": False}, max_experiments=1)
    report = _report(tmp_path, run_workflow(request, tmp_path))
    assert report["status"] == "needs_review"
    assert report["experiment_count"] == 1 and report["iteration_count"] == 0
    assert report["debugging_results"][0]["proposed_changes"]["deghost"]
    assert len(report["attempts"]) == 1 and not report["repair_comparisons"]


def test_tool_timeout_retries_and_graph_records_terminal_error(tmp_path, simulation):
    request = WorkflowRequest(user_request="HDR highlights", source=SourceSpec(simulation=simulation),
                              tool_timeout_s=0.001, tool_retries=1)
    report = _report(tmp_path, run_workflow(request, tmp_path))
    assert report["status"] == "failed" and report["experiment_count"] == 0
    assert report["final_artifact_id"] is None
    assert report["errors"][0]["node"] == "survey"
    calls = next(event for event in report["agent_trace"] if event["node"] == "survey")["tool_calls"]
    assert [row["attempt"] for row in calls] == [1, 2]
    assert all(row["status"] == "error" and "timeout" in row["error"] for row in calls)


def test_tool_allowlist_rejects_unknown_tools(tmp_path, simulation):
    runner = ToolRunner(2, 0)
    with pytest.raises(ToolFailure, match="unregistered tool"):
        runner.call("eval_python", SurveyInput(artifact_root=tmp_path, source=SourceSpec(simulation=simulation)))
