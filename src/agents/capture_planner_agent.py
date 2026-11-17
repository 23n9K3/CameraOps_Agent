"""Turn intent and measured scene conditions into an executable capture plan."""

from src.camera.capture_plan import CapturePlan
from src.memory.experiment_memory import ExperimentMemory
from src.memory.retrieval import similar_strategies
from src.memory.strategy_history import citation
from src.scene.analysis import SceneFeatures

from .supervisor import Mode

STRATEGIES = {"night": "night_temporal", "hdr": "hdr_exposure", "motion": "motion_reference"}
CAPTURE_MODES = {"night": "burst", "hdr": "bracket", "motion": "motion_aware"}


def _choose_mode(intent: dict, scene: SceneFeatures) -> Mode:
    if intent["hdr"]:
        return "hdr"
    if intent["night"]:
        return "night"
    if intent["motion"] or (scene.motion_fraction or 0) >= 0.035:
        return "motion"
    return "hdr" if scene.highlight_ratio > 0.025 and scene.shadow_ratio > 0.18 else "night"


def plan_capture(intent: dict, scene: SceneFeatures, *, simulated: bool, gap: float) -> CapturePlan:
    mode = _choose_mode(intent, scene)
    if not simulated:
        times = [frame.exposure_s for frame in scene.exposures]
        gains = [frame.analog_gain for frame in scene.exposures]
        if mode == "motion" or (mode == "night" and intent["motion"]):
            reference = min(range(len(times)), key=lambda i: (times[i], scene.saturation_by_frame[i]))
        elif mode == "hdr":
            order = sorted(range(len(times)), key=lambda i: times[i] * gains[i])
            reference = order[(len(order) - 1) // 2]
        else:
            reference = min(range(len(times)), key=lambda i: (scene.saturation_by_frame[i], -scene.frame_sharpness[i]))
        rationale = "Reuse measured exposures; select reference by motion, bracket position or sharpness and clipping."
    else:
        # Survey measurements are projected at 20 ms and 1x. Keep shot time and gain in physical units.
        moving = intent["motion"] or (scene.motion_fraction or 0) >= 0.035
        target_time = max(0.004, min(0.030, 0.020 * 0.68 / max(scene.luminance_p99, 1e-4)))
        if mode == "hdr":
            times = [max(0.001, target_time / 4), target_time, min(0.090, target_time * 3)]
            gains, reference = [1.0] * 3, 1
            rationale = "Bracket around the measured highlight percentile and retain a short highlight exposure."
        else:
            exposure = min(target_time, 0.005 if mode == "motion" else 0.009) if moving else target_time
            exposure = round(exposure, 6)
            count = 4 if mode == "motion" else 6
            gain = round(min(8.0, max(1.0, 0.11 / max(scene.brightness_median * exposure / 0.02, 1e-4))), 3)
            times, gains, reference = [exposure] * count, [gain] * count, 0
            rationale = "Cap motion exposure when needed and use repeated samples to reduce noise."
    if len(times) > 16:
        raise ValueError("the current CapturePlan supports up to 16 input frames")
    return CapturePlan(capture_mode=CAPTURE_MODES[mode], exposure_times=times, gains=gains,
                       inter_frame_gap_s=gap, reference_frame=reference, selected_frames=tuple(range(len(times))),
                       fusion_strategy=STRATEGIES[mode], constraints=("offline simulation" if simulated else "existing RAW; fixed exposures",),
                       rationale=rationale)


def consult_history(plan: CapturePlan, scene: SceneFeatures, memory: ExperimentMemory, *,
                    source_kind: str, task_id: str, max_frames: int) -> tuple[CapturePlan, dict]:
    mode = next(mode for mode, strategy in STRATEGIES.items() if strategy == plan.fusion_strategy)
    matches = similar_strategies(memory, scene.model_dump(mode="json"), mode=mode,
                                 source_kind=source_kind, exclude_group=task_id)
    evidence = {"agent": "capture_planner", "candidates": [citation(row) for row in matches], "applied": False}
    if not matches or source_kind != "synthetic_survey":
        return plan, evidence | {"reason": "no compatible history" if not matches else "reuse fixed RAW exposures"}
    prior = matches[0]["capture_plan"]
    scenario = {"motion_fraction": scene.motion_fraction, "median": scene.brightness_median,
                "highlight_p99": scene.luminance_p99}
    changes = {}
    if mode == "hdr":
        previous_times = prior.get("exposure_times", [])
        if len(previous_times) >= 3:
            relative_span = max(previous_times) / previous_times[prior["reference_frame"]]
            long = min(0.09, plan.exposure_times[plan.reference_frame] * min(4, max(2, relative_span)))
            current_short_peak = scene.luminance_p99 * plan.exposure_times[0] / 0.02
            current_long_median = scene.brightness_median * long / 0.02
            scenario.update(short_predicted_peak=current_short_peak, long_predicted_median=current_long_median)
            if current_short_peak < 0.85 and current_long_median >= 0.06 and (scene.motion_fraction or 0) < 0.035:
                changes["exposure_times"] = [*plan.exposure_times[:-1], long]
    else:
        historical_count = len(prior.get("exposure_times", []))
        median = scene.brightness_median * plan.exposure_times[0] * plan.gains[0] / 0.02
        peak = scene.luminance_p99 * plan.exposure_times[0] * plan.gains[0] / 0.02
        duration = historical_count * sum((plan.exposure_times[0], plan.inter_frame_gap_s)) - plan.inter_frame_gap_s
        moving = (scene.motion_fraction or 0) >= 0.035 or mode == "motion"
        minimum = 4 if median < 0.15 else 3
        scenario.update(predicted_peak=peak, predicted_median=median, candidate_duration_s=duration,
                        minimum_frames=minimum, max_frames=max_frames)
        if minimum <= historical_count <= min(8, max_frames) and peak < 0.95 and (not moving or duration <= 0.14):
            changes = {"exposure_times": [plan.exposure_times[0]] * historical_count,
                       "gains": [plan.gains[0]] * historical_count, "selected_frames": list(range(historical_count))}
    payload = plan.model_dump(mode="json", exclude={"frame_count"})
    proposed = CapturePlan.model_validate(payload | changes)
    applied = proposed != plan
    if applied:
        proposed = proposed.model_copy(update={"rationale": plan.rationale + " Historical span or count passed live-condition checks."})
    return proposed, evidence | {"applied": applied, "current_checks": scenario,
                                 "reason": "current-condition check accepted a derived change" if applied else
                                 "history agreed with the live plan or did not pass its admissibility checks"}
