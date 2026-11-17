"""Run a computational photography pipeline on an existing RAW sequence."""

import argparse
import json
import logging
from pathlib import Path

import cv2
import yaml

from src.computational.pipeline import PipelineConfig, run_pipeline
from src.computational.sequence import read_burst, read_file_sequence
from src.storage.artifact_store import ArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--burst", help="ID of a P2 synthetic Burst with Ground Truth")
    inputs.add_argument("--raw-manifest", type=Path, help="YAML describing ordered RAW files and exposures")
    parser.add_argument("--burst-root", type=Path, default=Path("artifacts/burst"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/photography"))
    parser.add_argument("--mode", required=True, choices=("night", "hdr", "motion"))
    parser.add_argument("--reference-frame", type=int, help="Zero-based reference index override")
    parser.add_argument("--parameters", type=Path, help="YAML with optional PipelineConfig fields")
    args = parser.parse_args()
    results = ArtifactStore(args.output)
    burst_store = ArtifactStore(args.burst_root)
    try:
        parameters = {}
        if args.parameters:
            with args.parameters.open(encoding="utf-8-sig") as source:
                parameters = yaml.safe_load(source)
            if not isinstance(parameters, dict):
                raise ValueError("pipeline parameters must be a YAML mapping")
        parameters["mode"] = args.mode
        if args.reference_frame is not None:
            parameters["reference_frame"] = args.reference_frame
        config = PipelineConfig.model_validate(parameters)
        sequence = read_burst(burst_store, args.burst) if args.burst else read_file_sequence(args.raw_manifest)
        identifier = run_pipeline(sequence, config, results, truth_store=burst_store if args.burst else None)
        print(json.dumps(results.load_metadata(identifier), indent=2))
    except (OSError, ValueError, KeyError, yaml.YAMLError, cv2.error) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
