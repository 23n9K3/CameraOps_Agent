"""Run the offline Lab agents on a simulation or a local RAW Burst."""

import argparse
import json
from pathlib import Path

import yaml

from src.camera.capture_simulator import SimulationConfig
from src.orchestration.graph import run_workflow
from src.orchestration.state import WorkflowRequest
from src.storage.artifact_store import ArtifactStore
from src.tools.contracts import SourceSpec

DEFAULT_SIMULATION = Path(__file__).resolve().parents[1] / "configs" / "synthetic_burst.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--brain", choices=("offline_mock", "deepseek"), default="offline_mock")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--simulation", type=Path, help="P2 scene/sensor YAML template")
    inputs.add_argument("--burst", help="existing Burst artifact ID")
    inputs.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--pipeline-parameters", type=Path, help="optional first-pass PipelineConfig YAML overrides")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-experiments", type=int, default=3)
    parser.add_argument("--tool-timeout", type=float, default=45)
    parser.add_argument("--tool-retries", type=int, default=1)
    args = parser.parse_args()
    try:
        simulation = None if args.burst or args.raw_manifest else SimulationConfig.from_file(args.simulation or DEFAULT_SIMULATION)
        parameters = yaml.safe_load(args.pipeline_parameters.read_text(encoding="utf-8-sig")) if args.pipeline_parameters else {}
        request = WorkflowRequest(user_request=args.goal, source=SourceSpec(simulation=simulation, burst_id=args.burst,
                                                                            raw_manifest=args.raw_manifest),
                                  brain=args.brain, initial_parameters=parameters, max_iterations=args.max_iterations,
                                  max_experiments=args.max_experiments, tool_timeout_s=args.tool_timeout, tool_retries=args.tool_retries)
        result = run_workflow(request, args.artifacts)
        report = ArtifactStore(args.artifacts / "reports").load_metadata(result["report_id"])
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
