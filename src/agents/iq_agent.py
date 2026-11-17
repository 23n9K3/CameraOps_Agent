"""Request measured IQ values for a published photography artifact."""

from src.tools.contracts import ArtifactInput
from src.tools.registry import ToolRunner


def evaluate(runner: ToolRunner, args: ArtifactInput):
    return runner.call("evaluate_final_image", args)
