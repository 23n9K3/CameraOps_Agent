"""Decode and inspect one or more offline RAW frames."""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import cv2
import yaml

from src.storage.artifact_store import ArtifactStore
from src.visualization.preview import display_rgb, encode_png

from .bayer import demosaic, normalize_mosaic
from .raw_config import RawConfig
from .raw_reader import read_raw


def inspect_raw(path: Path, config: RawConfig | None, store: ArtifactStore) -> dict:
    started = perf_counter()
    frame = read_raw(path, config)
    linear, saturated = normalize_mosaic(frame.mosaic, frame.config)
    rgb = demosaic(linear, frame.config.bayer_pattern)

    png = encode_png(display_rgb(rgb))

    metadata = {
        "source_file": str(path.resolve()),
        "source_format": frame.source_format,
        "raw_config": frame.config.model_dump(mode="json"),
        "color_space": "sensor-linear RGB",
        "demosaicing": "bilinear",
        "saturated_site_ratio": float(saturated.mean()),
        "decode_and_preview_ms": (perf_counter() - started) * 1000,
    }
    artifact_id = store.save(
        {"raw_mosaic": frame.mosaic, "linear_mosaic": linear, "linear_rgb": rgb, "saturation_mask": saturated},
        metadata, preview_png=png,
    )
    return {**metadata, "artifact_id": artifact_id, "artifact_path": str(store.path(artifact_id).resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, help="Shared YAML or JSON layout for headerless frames")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/raw"))
    args = parser.parse_args(argv)

    try:
        config = RawConfig.from_file(args.config) if args.config else None
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logging.error("%s: %s", args.config, exc)
        return 1

    status = 0
    store = ArtifactStore(args.artifacts)
    for path in args.inputs:
        try:
            print(json.dumps(inspect_raw(path, config, store), allow_nan=False))
        except (OSError, ValueError, cv2.error) as exc:
            logging.error("%s: %s", path, exc)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
