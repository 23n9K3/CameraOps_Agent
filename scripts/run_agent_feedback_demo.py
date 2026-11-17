"""Exercise the live feedback loop with known translation and coherent scene motion."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import yaml

from src.camera.capture_simulator import SimulationConfig
from src.orchestration.graph import run_workflow
from src.orchestration.state import WorkflowRequest
from src.storage.artifact_store import ArtifactStore
from src.tools.contracts import SourceSpec

ROOT = Path(__file__).resolve().parents[1]


def make_shifted_raw_manifest(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(115)
    texture = rng.uniform(0.04, 0.87, (112, 144)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (5, 5), 0.85)
    shift = np.array([[1, 0, 12], [0, 1, -8]], dtype=np.float32)
    shifted = cv2.warpAffine(texture, shift, (144, 112), borderMode=cv2.BORDER_REFLECT)
    entries = []
    for index, frame in enumerate((texture, shifted)):
        name = f"shift_{index}.raw"
        np.rint(64 + frame * (4095 - 64)).astype("<u2").tofile(directory / name)
        entries.append({"path": name, "exposure_s": 0.012, "analog_gain": 1})
    manifest = {"frames": entries, "raw_config": {"width": 144, "height": 112, "bit_depth": 12,
                "storage_format": "unpacked16", "bayer_pattern": "RGGB", "black_level": 64, "white_level": 4095}}
    path = directory / "shifted_sequence.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    raw_manifest = make_shifted_raw_manifest(args.artifacts / "agent_demo_inputs" / uuid4().hex)
    template = SimulationConfig.from_file(ROOT / "configs/synthetic_burst.yaml")
    cases = {
        "registration": WorkflowRequest(user_request="Reduce motion blur on these RAW frames.",
                                        source=SourceSpec(raw_manifest=raw_manifest),
                                        initial_parameters={"registration": False}),
        "ghosting": WorkflowRequest(user_request="处理运动行人，减少鬼影和拖影。",
                                    source=SourceSpec(simulation=template), initial_parameters={"deghost": False}),
    }
    reports = ArtifactStore(args.artifacts / "reports")
    summary = {}
    for name, request in cases.items():
        result = run_workflow(request, args.artifacts)
        report = reports.load_metadata(result["report_id"])
        summary[name] = {key: report[key] for key in ("artifact_id", "status", "termination_reason", "iteration_count",
                         "final_artifact_id", "repair_comparisons", "errors")}
    print(json.dumps(summary, indent=2))
    return 0 if all(item["repair_comparisons"] and item["status"] == "completed" for item in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
