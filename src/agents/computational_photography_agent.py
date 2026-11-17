"""Dispatch the concrete photography pipeline selected by the capture plan."""

from src.camera.capture_plan import CapturePlan
from src.computational.pipeline import PipelineConfig
from src.tools.contracts import PhotographyInput
from src.tools.registry import ToolRunner


def initialize_pipeline(plan: CapturePlan, initial_parameters: dict) -> PipelineConfig:
    strategy = {"night_temporal": "night", "hdr_exposure": "hdr", "motion_reference": "motion"}
    default = {"burst": "night", "bracket": "hdr", "motion_aware": "motion"}
    mode = strategy[plan.fusion_strategy] if plan.fusion_strategy else default[plan.capture_mode]
    local_reference = (plan.selected_frames or tuple(range(plan.frame_count))).index(plan.reference_frame)
    return PipelineConfig.model_validate(initial_parameters | {"mode": mode, "reference_frame": local_reference})


def process(runner: ToolRunner, args: PhotographyInput):
    return runner.call("run_computational_pipeline", args)
