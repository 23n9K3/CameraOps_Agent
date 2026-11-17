"""Generate a reproducible synthetic RAW Burst and print its artifact manifest."""

import argparse
import json
import logging
from pathlib import Path

import cv2
import yaml

from src.camera.capture_simulator import SimulationConfig, simulate_capture
from src.storage.artifact_store import ArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_burst.yaml"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/burst"))
    args = parser.parse_args()
    store = ArtifactStore(args.artifacts)
    try:
        artifact_id = simulate_capture(SimulationConfig.from_file(args.config), store)
    except (OSError, ValueError, yaml.YAMLError, cv2.error) as exc:
        logging.error("%s: %s", args.config, exc)
        return 1
    print(json.dumps(store.load_metadata(artifact_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
