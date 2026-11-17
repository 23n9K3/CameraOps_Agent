"""Execute capture, processing, evidence-based diagnosis and bounded reprocessing."""

from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from src.agents import (camera_debugging_agent, capture_planner_agent, computational_photography_agent,
                        deepseek, iq_agent, scene_motion_agent, supervisor)
from src.camera.capture_plan import CapturePlan
from src.computational.pipeline import PipelineConfig
from src.iq.evaluation import IQResult
from src.memory.experiment_memory import Experiment, ExperimentMemory
from src.memory.strategy_history import record_verified_repairs
from src.scene.analysis import SceneFeatures
from src.storage.artifact_store import ArtifactStore
from src.tools.contracts import ArtifactInput, CaptureInput, PhotographyInput, SurveyInput
from src.tools.registry import ToolRunner

from .state import CameraState, WorkflowRequest


def build_graph(artifact_root: str | Path, *, on_event: Callable[[dict], None] | None = None,
                runner_factory=ToolRunner, history_memory: ExperimentMemory | None = None):
    root = Path(artifact_root).resolve()
    memory = ExperimentMemory(root / "experiments.sqlite3")
    history = history_memory if history_memory is not None else memory

    def emit(event):
        if on_event is not None:
            on_event(event)

    def source_kind(request):
        return "synthetic_survey" if request.source.simulation else "existing_raw"

    def record_experiment(state, request, *, output_id, runtime_ms, metrics, status, error=None):
        input_id = state["input_artifacts"][0]
        source = ArtifactStore(root / "burst").load_metadata(input_id)
        simulation = source.get("simulation_config", {})
        memory.record(Experiment(
            group_id=state["task_id"], number=state["iteration_count"], kind="workflow", mode=state["task_type"],
            source_kind=source_kind(request), status=status, input_artifact_id=input_id, output_artifact_id=output_id,
            seed=simulation.get("seed"), runtime_ms=runtime_ms, survey=state["capture_survey"], scene=state["scene_features"],
            capture_plan=state["capture_plan"], parameters=state["pipeline_config"], metrics=metrics,
            configuration={"user_request": request.user_request, "quality_limits": request.quality_limits.model_dump(),
                           "simulation": simulation or None, "source": source.get("source")}, error=error,
        ))

    def node(name):
        def wrap(action):
            @wraps(action)
            def execute(state: CameraState):
                started = perf_counter()
                request = WorkflowRequest.model_validate(state["request"])
                event = {"node": name, "iteration": state.get("iteration_count", 0)}
                emit({"type": "node_started", **event, "status": "running"})
                runner = runner_factory(request.tool_timeout_s, request.tool_retries,
                                    on_event=(lambda message: emit({**event, **message})) if on_event is not None else None)
                try:
                    updates = action(state, request, runner)
                    event["status"] = "ok"
                except Exception as exc:
                    event.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    updates = {"status": "failed", "termination_reason": f"{name} failed",
                               "errors": [{"node": name, "error": event["error"]}]}
                    if name in ("photography", "iq"):
                        try:
                            record_experiment(state, request, output_id=state["photography_results"]["artifact_id"] if name == "iq" else None,
                                              runtime_ms=(perf_counter() - started) * 1000, metrics={},
                                              status="failed", error=event["error"])
                        except Exception as persistence_error:
                            updates["errors"].append({"node": "experiment_memory", "error": str(persistence_error)})
                if name == "photography":
                    updates["experiment_count"] = state.get("experiment_count", 0) + 1
                event.update(runtime_ms=(perf_counter() - started) * 1000, tool_calls=runner.calls)
                if updates.get("photography_results"):
                    event["artifact_id"] = updates["photography_results"]["artifact_id"]
                emit({"type": "node_finished", **event, "updates": updates})
                return updates | {"agent_trace": [event]}
            return execute
        return wrap

    @node("supervisor")
    def start(state, request, runner):
        return {"intent": supervisor.interpret_goal(request.user_request), "status": "running",
                "execution_plan": ["survey", *(["brain"] if request.brain == "deepseek" else []),
                                   "capture_planner", "capture", "scene_motion", "photography", "iq",
                                   "decision", "debugging", "strategy_revision", "report"]}

    @node("survey")
    def survey(state, request, runner):
        result = scene_motion_agent.survey(runner, SurveyInput(artifact_root=root, source=request.source))
        return {"capture_survey": result.features.model_dump(mode="json"),
                "input_artifacts": [result.input_artifact_id] if result.input_artifact_id else []}

    @node("brain")
    def choose_brain_strategy(state, request, runner):
        model, decision = deepseek.choose_strategy(request.user_request, SceneFeatures.model_validate(state["capture_survey"]))
        return {"brain_decision": {"model": model, **decision.model_dump(mode="json")},
                "intent": {"hdr": decision.mode == "hdr", "night": decision.mode == "night",
                           "motion": decision.mode == "motion" or decision.protect_motion}}

    @node("capture_planner")
    def plan(state, request, runner):
        template = request.source.simulation
        gap = template.capture_plan.inter_frame_gap_s if template else 0
        scene = SceneFeatures.model_validate(state["capture_survey"])
        capture_plan = capture_planner_agent.plan_capture(state["intent"], scene, simulated=template is not None, gap=gap)
        reviews = []
        if request.enable_memory:
            capture_plan, review = capture_planner_agent.consult_history(
                capture_plan, scene, history, source_kind=source_kind(request), task_id=state["task_id"],
                max_frames=4_194_304 // (template.scene.width * template.scene.height) if template else 16)
            reviews.append(review)
        if request.reference_frame is not None:
            capture_plan = CapturePlan.model_validate(capture_plan.model_dump(mode="json") |
                                                      {"reference_frame": request.reference_frame})
        parameters = request.initial_parameters
        if state.get("brain_decision", {}).get("protect_motion"):
            parameters = {"protect_motion_reference": True} | parameters
        config = computational_photography_agent.initialize_pipeline(capture_plan, parameters)
        return {"capture_plan": capture_plan.model_dump(mode="json"), "task_type": config.mode,
                "pipeline_config": config.model_dump(mode="json"), "memory_reviews": reviews}

    @node("capture")
    def capture(state, request, runner):
        if request.source.simulation is None:
            return {}
        result = runner.call("simulate_capture", CaptureInput(artifact_root=root, simulation=request.source.simulation,
                                                             plan=CapturePlan.model_validate(state["capture_plan"])))
        return {"input_artifacts": [result.artifact_id]}

    @node("scene_motion")
    def analyze(state, request, runner):
        result = scene_motion_agent.inspect_burst(runner, ArtifactInput(artifact_root=root, artifact_id=state["input_artifacts"][0]))
        return {"scene_features": result.model_dump(mode="json"),
                "motion_results": {"motion_fraction": result.motion_fraction, "moving_flow_p90_px": result.moving_flow_p90_px}}

    @node("photography")
    def photograph(state, request, runner):
        config = PipelineConfig.model_validate(state["pipeline_config"])
        result = computational_photography_agent.process(runner, PhotographyInput(
            artifact_root=root, artifact_id=state["input_artifacts"][0],
            plan=CapturePlan.model_validate(state["capture_plan"]), config=config))
        return {"photography_results": {"artifact_id": result.artifact_id, "parameters": config.model_dump(mode="json")}}

    @node("iq")
    def iq(state, request, runner):
        result = iq_agent.evaluate(runner, ArtifactInput(artifact_root=root, artifact_id=state["photography_results"]["artifact_id"]))
        return {"iq_results": result.model_dump(mode="json")}

    @node("decision")
    def decide(state, request, runner):
        measurements = IQResult.model_validate(state["iq_results"])
        assessment = supervisor.assess_result(measurements, request.quality_limits)
        record = {**state["photography_results"], "iteration": state["iteration_count"],
                  "iq": state["iq_results"], **assessment}
        record_experiment(state, request, output_id=measurements.artifact_id, runtime_ms=measurements.metrics["runtime"].value,
                          metrics=state["iq_results"], status="needs_review" if assessment["faults"] else "passed")
        if state.get("attempts") and state.get("debugging_results"):
            record_verified_repairs(memory, state["task_id"], state["attempts"][-1], record,
                                    state["debugging_results"][-1]["diagnoses"])
        updates = {"faults": assessment["faults"], "attempts": [record]}
        if "final_rank" not in state or assessment["rank"] < state["final_rank"]:
            updates.update(final_rank=assessment["rank"], final_artifact_id=measurements.artifact_id)
        if not assessment["faults"]:
            updates.update(status="completed", termination_reason="no measured registration or ghosting violation")
        elif not request.enable_feedback:
            updates.update(status="needs_review", termination_reason="feedback disabled for this run")
        return updates

    @node("debugging")
    def debug(state, request, runner):
        diagnoses, revised = camera_debugging_agent.debug(
            runner, root, state["photography_results"]["artifact_id"], state["faults"], request.quality_limits,
            PipelineConfig.model_validate(state["pipeline_config"]))
        reviews = []
        if request.enable_memory:
            revised, reviews = camera_debugging_agent.consult_history(
                revised, diagnoses, state["scene_features"], history, mode=state["task_type"],
                source_kind=source_kind(request), task_id=state["task_id"])
        evidence = [record.model_dump(mode="json") for record in diagnoses]
        memory.set_diagnoses(state["task_id"], state["iteration_count"], evidence)
        proposed = revised.model_dump(mode="json") if revised else None
        return {"proposed_config": proposed, "memory_reviews": reviews, "debugging_results": [{
            "iteration": state["iteration_count"], "artifact_id": state["photography_results"]["artifact_id"],
            "diagnoses": evidence,
            "proposed_changes": {key: value for key, value in (proposed or {}).items()
                                 if value != state["pipeline_config"][key]},
        }]}

    @node("strategy_revision")
    def revise(state, request, runner):
        if state["iteration_count"] >= request.max_iterations or state["experiment_count"] >= request.max_experiments:
            return {"status": "needs_review", "termination_reason": "revision or experiment budget exhausted"}
        candidate = state.get("proposed_config")
        if not candidate or any(candidate == record["parameters"] for record in state.get("attempts", [])):
            return {"status": "needs_review", "termination_reason": "no new validated repair strategy"}
        return {"pipeline_config": candidate, "iteration_count": state["iteration_count"] + 1}

    def report(state: CameraState):
        started = perf_counter()
        coordinates = {"node": "report", "iteration": state.get("iteration_count", 0)}
        emit({"type": "node_started", **coordinates, "status": "running"})
        trace = {**coordinates, "status": "ok", "tool_calls": []}
        attempts = state.get("attempts", [])
        comparisons = [
            {"before_artifact_id": before["artifact_id"], "after_artifact_id": after["artifact_id"],
             "before_faults": before["faults"], "after_faults": after["faults"],
             "metrics": {name: {"before": before["iq"]["metrics"][name]["value"],
                                 "after": after["iq"]["metrics"][name]["value"]}
                         for name in ("alignment_error", "ghosting_score", "psnr")}}
            for before, after in zip(attempts, attempts[1:])
        ]
        identifier = ArtifactStore(root / "reports").save({}, {
            "artifact_type": "camera_workflow_report", "schema_version": 1, "llm_mode": state["request"]["brain"],
            "brain_decision": state.get("brain_decision"),
            "task_id": state["task_id"], "user_request": state["user_request"], "task_type": state.get("task_type"),
            "status": state["status"], "termination_reason": state.get("termination_reason"),
            "input_artifacts": state.get("input_artifacts", []), "capture_plan": state.get("capture_plan"),
            "capture_survey": state.get("capture_survey"), "scene_features": state.get("scene_features"),
            "final_artifact_id": state.get("final_artifact_id"), "experiment_count": state.get("experiment_count", 0),
            "iteration_count": state.get("iteration_count", 0), "attempts": attempts,
            "debugging_results": state.get("debugging_results", []), "repair_comparisons": comparisons,
            "agent_trace": state.get("agent_trace", []) + [trace], "errors": state.get("errors", []),
            "quality_limits": state["request"]["quality_limits"],
            "memory_reviews": state.get("memory_reviews", []), "experiment_memory": str(memory.path),
            "workflow_policy": {name: state["request"].get(name) for name in
                                ("enable_feedback", "enable_memory", "reference_frame")},
            "execution_limits": {name: state["request"][name] for name in
                                 ("max_iterations", "max_experiments", "tool_timeout_s", "tool_retries")},
        })
        emit({"type": "node_finished", **trace, "runtime_ms": (perf_counter() - started) * 1000,
              "updates": {"report_id": identifier, "final_artifact_id": state.get("final_artifact_id"),
                          "status": state["status"], "termination_reason": state.get("termination_reason")}})
        return {"report_id": identifier, "agent_trace": [trace]}

    graph = StateGraph(CameraState)
    for name, action in (("supervisor", start), ("survey", survey), ("brain", choose_brain_strategy),
                         ("capture_planner", plan), ("capture", capture),
                         ("scene_motion", analyze), ("photography", photograph), ("iq", iq), ("decision", decide),
                         ("debugging", debug), ("strategy_revision", revise), ("report", report)):
        graph.add_node(name, action)
    graph.add_edge(START, "supervisor")
    path = ("brain", "capture_planner", "capture", "scene_motion", "photography", "iq", "decision")
    for current, following in [("supervisor", "survey"), *zip(path, path[1:])]:
        graph.add_conditional_edges(current, lambda state: "report" if state["status"] == "failed" else "continue",
                                    {"report": "report", "continue": following})
    graph.add_conditional_edges("survey", lambda state: "report" if state["status"] == "failed" else (
        "brain" if state["request"]["brain"] == "deepseek" else "capture_planner"),
        {"report": "report", "brain": "brain", "capture_planner": "capture_planner"})
    graph.add_conditional_edges("decision", lambda state: "debugging" if state["status"] == "running" else "report",
                                {"debugging": "debugging", "report": "report"})
    graph.add_conditional_edges("debugging", lambda state: "strategy_revision" if state["status"] == "running" else "report",
                                {"strategy_revision": "strategy_revision", "report": "report"})
    graph.add_conditional_edges("strategy_revision", lambda state: "photography" if state["status"] == "running" else "report",
                                {"photography": "photography", "report": "report"})
    graph.add_edge("report", END)
    return graph.compile()


def run_workflow(request: WorkflowRequest, artifact_root: str | Path, *, on_event: Callable[[dict], None] | None = None,
                 runner_factory=ToolRunner, history_memory: ExperimentMemory | None = None) -> CameraState:
    graph = build_graph(artifact_root, on_event=on_event, runner_factory=runner_factory, history_memory=history_memory)
    return graph.invoke({"task_id": uuid4().hex, "user_request": request.user_request,
                         "request": request.model_dump(mode="json"), "experiment_count": 0, "iteration_count": 0},
                        config={"recursion_limit": 24 + 6 * request.max_iterations})
