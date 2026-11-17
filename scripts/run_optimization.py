"""Run measured Optuna search and publish a Pareto report."""

import argparse
import json
from pathlib import Path

import yaml

from src.camera.capture_simulator import SimulationConfig, simulate_capture
from src.computational.pipeline import PipelineConfig
from src.optimization.optimizer import OptimizationConfig, TrialFailed, optimize_burst
from src.storage.artifact_store import ArtifactStore

DEFAULT_SIMULATION = Path(__file__).resolve().parents[1] / "configs" / "synthetic_burst.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--burst", help="existing synthetic Burst artifact ID")
    inputs.add_argument("--simulation", type=Path, help="P2 YAML used to generate a fixed Burst once")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--mode", default="motion", choices=("night", "hdr", "motion"))
    parser.add_argument("--baseline-parameters", type=Path, help="optional PipelineConfig YAML overrides")
    parser.add_argument("--trials", type=int, default=18, help="search trials in addition to the baseline")
    parser.add_argument("--seed", type=int, default=71, help="Optuna sampler seed")
    args = parser.parse_args()
    try:
        config = OptimizationConfig(seed=args.seed, trials=args.trials)
        parameters = yaml.safe_load(args.baseline_parameters.read_text(encoding="utf-8-sig")) if args.baseline_parameters else {}
        if not isinstance(parameters, dict):
            raise ValueError("baseline parameters must be a YAML mapping")
        baseline = PipelineConfig.model_validate(parameters | {"mode": args.mode})
        burst = args.burst or simulate_capture(SimulationConfig.from_file(args.simulation or DEFAULT_SIMULATION),
                                               ArtifactStore(args.artifacts / "burst"))
        report_id = optimize_burst(args.artifacts, burst, baseline, config)
        report = ArtifactStore(args.artifacts / "optimization").load_metadata(report_id)
    except (OSError, ValueError, TrialFailed, yaml.YAMLError) as exc:
        parser.exit(1, f"{exc}\n")
    summary = {key: report[key] for key in ("artifact_id", "study_name", "input_artifact_id", "completed_trials", "trial_count",
                                          "objective_comparison")}
    summary["pareto_trials"] = [row["number"] for row in report["pareto_front"]]
    summary["baseline_artifact_id"] = report["baseline"]["output_artifact_id"]
    summary["selected_artifact_id"] = report["selected"]["output_artifact_id"]
    summary["selected_trial"] = report["selected"]["number"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
