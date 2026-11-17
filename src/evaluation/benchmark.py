"""Execute five policies against identical datasets with independent success criteria."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.agents import supervisor
from src.camera.bayer import demosaic, normalize_mosaic
from src.camera.capture_plan import CapturePlan
from src.computational.comparison import ground_truth_regions
from src.computational.pipeline import PipelineConfig
from src.computational.registration import luminance
from src.computational.sequence import read_burst
from src.memory.experiment_memory import ExperimentMemory
from src.orchestration.graph import run_workflow
from src.orchestration.state import WorkflowRequest
from src.storage.artifact_store import ArtifactStore
from src.tools.contracts import ArtifactInput, PhotographyInput, SourceSpec, SurveyInput
from src.tools.registry import REGISTRY

from .cases import Case


POLICIES = {
    "fixed": "Fixed Pipeline",
    "single": "Single Agent",
    "multi": "Multi-Agent without Feedback",
    "feedback": "Multi-Agent with Debugging Feedback",
    "memory": "Multi-Agent with Experiment Memory",
}


class LocalBenchmarkRunner:
    """Use registered schemas and implementations in-process throughout the study."""

    def __init__(self, timeout=45, retries=0, *, on_event=None):
        self.calls = []

    def call(self, name, arguments):
        spec = REGISTRY[name]
        started = perf_counter()
        entry = {"tool": name, "attempt": 1}
        try:
            result = spec.output_schema.model_validate(spec.execute(spec.input_schema.model_validate(arguments.model_dump())))
            entry.update(status="ok", artifact_id=getattr(result, "artifact_id", None),
                         input_artifact_id=getattr(arguments, "artifact_id", None))
            return result
        except Exception as exc:
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.calls.append(entry | {"runtime_ms": (perf_counter() - started) * 1000})


def _single_pass(policy: str, case: Case, root: Path) -> dict:
    runner = LocalBenchmarkRunner()
    try:
        sequence = read_burst(ArtifactStore(root / "burst"), case.input_id)
        values = sequence.exposures
        if policy == "single":
            survey = runner.call("survey_capture_conditions", SurveyInput(artifact_root=root, source=SourceSpec(burst_id=case.input_id)))
            goal = supervisor.interpret_goal(case.goal)
            mode = "hdr" if goal["hdr"] else "night" if goal["night"] else (
                "motion" if goal["motion"] or (survey.features.motion_fraction or 0) >= 0.035 else "night")
        else:
            mode = "night"
        plan = CapturePlan(exposure_times=[entry.exposure_s for entry in values], gains=[entry.analog_gain for entry in values],
                           reference_frame=case.reference, capture_mode={"night": "burst", "hdr": "bracket", "motion": "motion_aware"}[mode])
        config = PipelineConfig.model_validate(case.initial | {"mode": mode, "reference_frame": case.reference})
        artifact = runner.call("run_computational_pipeline", PhotographyInput(artifact_root=root, artifact_id=case.input_id,
                                                                                 plan=plan, config=config)).artifact_id
        iq = runner.call("evaluate_final_image", ArtifactInput(artifact_root=root, artifact_id=artifact)).model_dump(mode="json")
        return {"artifact_id": artifact, "attempts": [{"artifact_id": artifact, "iq": iq, "iteration": 0}],
                "trace": runner.calls, "status": "completed", "capture_plan": plan.model_dump(mode="json")}
    except Exception as exc:
        return {"status": "failed", "errors": [f"{type(exc).__name__}: {exc}"], "trace": runner.calls}


def run_policy(policy: str, case: Case, root: Path, history: ExperimentMemory | None = None) -> dict:
    started = perf_counter()
    if policy in ("fixed", "single"):
        evidence = _single_pass(policy, case, root)
    else:
        try:
            request = WorkflowRequest(user_request=case.goal, source=SourceSpec(burst_id=case.input_id),
                                      initial_parameters=case.initial, reference_frame=case.reference,
                                      enable_feedback=policy != "multi", enable_memory=policy == "memory")
            state = run_workflow(request, root, runner_factory=LocalBenchmarkRunner, history_memory=history)
            evidence = {"artifact_id": state.get("final_artifact_id"), "report_id": state["report_id"],
                        "attempts": state.get("attempts", []), "diagnoses": state.get("debugging_results", []),
                        "trace": [tool for event in state["agent_trace"] for tool in event["tool_calls"]],
                        "memory_reviews": state.get("memory_reviews", []), "status": state["status"],
                        "errors": state.get("errors", []), "capture_plan": state.get("capture_plan")}
        except Exception as exc:
            evidence = {"status": "failed", "errors": [f"{type(exc).__name__}: {exc}"]}
    evidence["wall_ms"] = (perf_counter() - started) * 1000
    return evidence


def _measure_pixels(image: np.ndarray, target: np.ndarray, masks: dict, smooth: np.ndarray) -> dict:
    error = image - target
    noise = float(np.std(luminance(error)[smooth], dtype=np.float64)) if smooth.sum() >= 32 else None
    def rmse(region):
        mask = masks[region]
        return float(np.sqrt(np.mean(error[mask] ** 2, dtype=np.float64))) if mask.sum() >= 32 else None
    mse = float(np.mean(error.astype(np.float64) ** 2))
    peak = max(1, float(target.max()))
    return {"noise_std": noise, "ghost_rmse": rmse("motion_region"), "highlight_rmse": rmse("reference_clipped_highlights"),
            "shadow_rmse": rmse("static_shadows"), "psnr": 10 * float(np.log10(peak**2 / mse)) if mse else None}


def assess(case: Case, root: Path, evidence: dict) -> dict:
    identifier = evidence.get("artifact_id")
    if identifier is None:
        return {"success": False, "tool_correct": False, "goal_pass": False, "failure_reason": "no scored output"}
    store = ArtifactStore(root / "photography")
    bursts = ArtifactStore(root / "burst")
    source = bursts.load_metadata(case.input_id)
    photo = store.load_metadata(identifier)
    if photo["source"]["input_artifact_id"] != case.input_id or photo["source_reference_index"] != case.reference:
        raise ValueError("benchmark output did not preserve the shared input and reference frame")
    entry = next(attempt for attempt in evidence["attempts"] if attempt["artifact_id"] == identifier)
    iq = entry["iq"]["metrics"]
    chosen = photo["pipeline"]["mode"]
    tool_ok = chosen == case.expected_mode and any(row["tool"] == "run_computational_pipeline" and row["status"] == "ok"
                                                    and row.get("artifact_id") == identifier for row in evidence["trace"])
    factor = photo["evaluation"]["scene_to_reference_scale"]
    target, masks = ground_truth_regions(bursts, case.input_id, case.reference, factor)
    edges = np.abs(cv2.Laplacian(luminance(target), cv2.CV_32F))
    smooth = cv2.erode((masks["static_shadows"] & (edges < 0.008)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    sequence = read_burst(bursts, case.input_id)
    layout = sequence.layouts[case.reference]
    linear, _ = normalize_mosaic(sequence.mosaics[case.reference], layout)
    base = _measure_pixels(demosaic(linear, layout.bayer_pattern) / factor, target, masks, smooth)
    measured = _measure_pixels(store.load_array(identifier, "final_linear", mmap_mode="r") / factor, target, masks, smooth)
    measured.update(ssim=iq["ssim"]["value"], alignment_residual=iq["alignment_error"]["value"],
                    pipeline_ms=iq["runtime"]["value"], pipeline_psnr=iq["psnr"]["value"], blind_ghosting=iq["ghosting_score"]["value"])
    def below(metric, multiplier):
        return measured[metric] is not None and base[metric] is not None and measured[metric] <= base[metric] * multiplier

    transform_error = None
    if case.shift:
        transforms = store.load_array(identifier, "transforms", mmap_mode="r")
        transform_error = float(np.linalg.norm(transforms[1, :, 2] - case.shift))
        passed, target_name = transform_error <= 1, "known translation error ≤ 1 px"
    elif case.family == "bayer":
        passed = photo["raw_layouts"][case.reference]["bayer_pattern"] == source["simulation_config"]["sensor"]["bayer_pattern"]
        target_name = "restore known Bayer pattern"
    elif case.family in ("hdr", "overexposure"):
        passed, target_name = below("highlight_rmse", 0.93), "highlight RMSE ≤ 93% single RAW"
    elif case.family in ("motion", "ghosting"):
        passed, target_name = below("ghost_rmse", 1.10), "motion-region RMSE ≤ 110% single RAW"
    elif case.family == "underexposure":
        passed, target_name = below("shadow_rmse", 0.90), "shadow RMSE ≤ 90% single RAW"
    else:
        psnr_preserved = measured["psnr"] is not None and base["psnr"] is not None and measured["psnr"] >= base["psnr"] - 0.25
        threshold = {"noise": 0.93, "night": 0.97, "control": 1.05}[case.family]
        passed, target_name = below("noise_std", threshold) and psnr_preserved, f"noise STD ≤ {threshold:.0%} single RAW and PSNR drop ≤ 0.25 dB"
    diagnoses = sorted({rec["fault"] for event in evidence.get("diagnoses", []) for rec in event["diagnoses"]
                        if rec["validation_result"] == "confirmed"})
    first_id = evidence["attempts"][0]["artifact_id"]
    repair = None
    if first_id != identifier and case.fault in diagnoses:
        if case.shift:
            first_transform = store.load_array(first_id, "transforms", mmap_mode="r")[1, :, 2]
            before = float(np.linalg.norm(first_transform - case.shift))
            repair = {"metric": "translation_error_px", "before": before, "after": transform_error, "passed": before > 1 and passed}
        elif case.fault == "ghosting":
            first = store.load_array(first_id, "final_linear", mmap_mode="r") / factor
            before = _measure_pixels(first, target, masks, smooth)["ghost_rmse"]
            repair = {"metric": "ghost_rmse", "before": before, "after": measured["ghost_rmse"],
                      "passed": before is not None and base["ghost_rmse"] is not None and
                      before > 1.10 * base["ghost_rmse"] and passed}
    scored = [metric for metric in iq.values() if metric["valid"]]
    grounded = sum(metric["value"] is not None and bool(metric["method"]) and identifier in metric["input_artifacts"]
                   for metric in scored)
    return {"success": bool(tool_ok and passed and evidence["status"] != "failed"), "goal_pass": bool(passed),
            "tool_correct": tool_ok, "target": target_name, "mode": chosen, "measurements": measured,
            "single_reference": base, "translation_error_px": transform_error, "confirmed_faults": diagnoses,
            "repair": repair, "grounded": grounded, "valid_metrics": len(scored)}


def run_case(case: Case, policy: str, root: Path, history: ExperimentMemory | None = None) -> dict:
    execution = run_policy(policy, case, root, history)
    result = assess(case, root, execution)
    return {"case": asdict(case), "policy": policy, "execution": execution, **result}
