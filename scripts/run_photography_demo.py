"""Generate and evaluate three reproducible synthetic RAW photography experiments."""

import argparse
import json
from pathlib import Path

from src.camera.capture_simulator import SimulationConfig, simulate_capture
from src.computational.pipeline import PipelineConfig, run_pipeline
from src.computational.sequence import read_burst
from src.storage.artifact_store import ArtifactStore


def demo_config(base: SimulationConfig, mode: str) -> SimulationConfig:
    settings = base.model_dump(mode="json", exclude={"capture_plan": {"frame_count"}})
    if mode == "night":
        settings["scene"]["velocity_px_s"] = 0
        settings["sensor"].update(photon_rate_e_s=65000, read_noise_e=6.0)
        settings["capture_plan"].update(capture_mode="burst", exposure_times=[0.009] * 8,
                                        gains=[4.0] * 8, reference_frame=0)
    elif mode == "hdr":
        settings["scene"]["velocity_px_s"] = 0
        settings["capture_plan"].update(exposure_times=[0.004, 0.035, 0.070, 0.014],
                                        gains=[1.0] * 4, reference_frame=1)
    return SimulationConfig.model_validate(settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    base = SimulationConfig.from_file(Path(__file__).resolve().parents[1] / "configs/synthetic_burst.yaml")
    raw = ArtifactStore(args.artifacts / "burst")
    results = ArtifactStore(args.artifacts / "photography")
    summary = {}
    for mode in ("night", "hdr", "motion"):
        input_id = simulate_capture(demo_config(base, mode), raw)
        sequence = read_burst(raw, input_id)
        output_id = run_pipeline(sequence, PipelineConfig(mode=mode), results, truth_store=raw)
        metadata = results.load_metadata(output_id)
        summary[mode] = {"input_artifact_id": input_id, "output_artifact_id": output_id,
                         "runtime_ms": metadata["runtime_ms"], "scores": metadata["evaluation"]["scores"]}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
