"""Collect scene and motion measurements from the survey and captured frames."""

from src.tools.contracts import ArtifactInput, SurveyInput
from src.tools.registry import ToolRunner


def survey(runner: ToolRunner, args: SurveyInput):
    return runner.call("survey_capture_conditions", args)


def inspect_burst(runner: ToolRunner, args: ArtifactInput):
    return runner.call("analyze_scene_motion", args)
