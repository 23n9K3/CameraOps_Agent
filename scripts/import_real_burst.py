"""Decode a camera RAW burst using measured shutter and relative EXIF ISO."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import rawpy
from tifffile import TiffFile

from src.camera.raw_config import site_levels
from src.camera.raw_reader import RawFrame, read_raw
from src.computational.sequence import Exposure, RawSequence, publish_raw_sequence
from src.storage.artifact_store import ArtifactStore


def camera_metadata(path: Path) -> tuple[float, float, int, list[float]]:
    with path.open("rb") as source, rawpy.imread(source) as camera:
        seconds, iso = camera.other.shutter_speed, camera.other.iso_speed
        orientation = camera.sizes.flip
        balance = camera.camera_whitebalance[:3]
    # Some DNG writers put these EXIF tags directly in the main TIFF directory.
    if path.suffix.lower() == ".dng" and (seconds <= 0 or iso <= 0):
        with TiffFile(path) as tiff:
            tags = tiff.pages[0].tags
            if seconds <= 0 and (shutter := tags.valueof(33434)):
                numerator, denominator = shutter
                if denominator:
                    seconds = numerator / denominator
            if iso <= 0 and isinstance(iso_tag := tags.valueof(34855), int):
                iso = float(iso_tag)
    return seconds, iso, orientation, balance


def prepare_full_frame(frame: RawFrame, factor: int, rotation: int) -> RawFrame:
    """Average each CFA phase independently and keep the sensor layout in sync."""
    height = frame.config.height // (2 * factor)
    width = frame.config.width // (2 * factor)
    if min(width, height) < 1:
        raise ValueError("binning factor is too large for the frame")
    mosaic = np.empty((2 * height, 2 * width), dtype=np.uint16)
    for y, x in np.ndindex(2, 2):
        plane = frame.mosaic[y:2 * height * factor:2, x:2 * width * factor:2]
        mosaic[y::2, x::2] = cv2.resize(plane, (width, height), interpolation=cv2.INTER_AREA)
    mosaic = np.ascontiguousarray(np.rot90(mosaic, rotation))
    config = frame.config
    pattern = np.rot90(np.array(list(config.bayer_pattern)).reshape(2, 2), rotation)
    blacks = np.rot90(np.array(site_levels(config.black_level)).reshape(2, 2), rotation)
    whites = np.rot90(np.array(site_levels(config.white_level)).reshape(2, 2), rotation)
    return replace(frame, mosaic=mosaic, config=config.model_copy(update={
        "width": mosaic.shape[1], "height": mosaic.shape[0],
        "bayer_pattern": "".join(pattern.ravel()),
        "black_level": tuple(blacks.ravel().tolist()), "white_level": tuple(whites.ravel().tolist()),
    }))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, nargs="+", help="RAW files in capture order from the same camera and scene")
    resize = parser.add_mutually_exclusive_group()
    resize.add_argument("--crop", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="Optional Bayer-aligned center crop")
    resize.add_argument("--binning", type=int, help="Bayer-aware full-frame downsampling with camera orientation")
    parser.add_argument("--scene-label", help="Display label for this scene")
    parser.add_argument("--color-matrix", type=Path, help="Row-major sensor RGB to linear sRGB matrix, applied after white balance")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not 2 <= len(args.frames) <= 16:
        parser.error("provide 2 through 16 RAW frames")
    if args.binning is not None and args.binning < 1:
        parser.error("binning must be positive")
    try:
        metadata = [camera_metadata(path) for path in args.frames]
        if any(not (seconds > 0 and iso > 0) for seconds, iso, _, _ in metadata):
            raise ValueError("each frame must contain a positive EXIF shutter speed and ISO")
        base_iso = min(iso for _, iso, _, _ in metadata)
        exposures = tuple(Exposure(exposure_s=seconds, analog_gain=iso / base_iso)
                          for seconds, iso, _, _ in metadata)
        source = {"files": [str(path.resolve()) for path in args.frames],
                  "exif_iso": [iso for _, iso, _, _ in metadata],
                  "gain_convention": "EXIF ISO divided by minimum burst ISO"}
        if args.scene_label:
            source["scene_label"] = args.scene_label
        if args.color_matrix:
            matrix = np.loadtxt(args.color_matrix).reshape(3, 3)
            balance = np.array(metadata[0][3])
            if not np.isfinite(matrix).all() or not (np.isfinite(balance) & (balance > 0)).all():
                raise ValueError("color rendering requires a finite matrix and positive camera white balance")
            source["display_color_matrix"] = (matrix * (balance / balance[1])).tolist()
        frames, crop = [], None
        if args.binning:
            orientation = metadata[0][2]
            if orientation not in (0, 3, 5, 6) or any(item[2] != orientation for item in metadata):
                raise ValueError("binning requires a common supported camera orientation")
            source.update(bayer_binning=args.binning, camera_orientation=orientation)
            rotation = {0: 0, 3: 2, 5: 1, 6: -1}[orientation]
        for path in args.frames:
            frame = read_raw(path)
            if args.crop:
                width, height = args.crop
                if not (2 <= width <= frame.config.width and 2 <= height <= frame.config.height):
                    raise ValueError("crop must fit inside each frame and be at least 2 by 2 pixels")
                x = (frame.config.width - width) // 4 * 2
                y = (frame.config.height - height) // 4 * 2
                region = [x, y, width, height]
                if crop is not None and region != crop:
                    raise ValueError("frames do not share a common Bayer-aligned crop")
                crop = region
                frame = replace(frame, mosaic=frame.mosaic[y:y + height, x:x + width].copy(),
                                config=frame.config.model_copy(update={"width": width, "height": height}))
            elif args.binning:
                frame = prepare_full_frame(frame, args.binning, rotation)
            frames.append(frame)
        source.update(formats=[frame.source_format for frame in frames], center_crop_xywh=crop)
        sequence = RawSequence(tuple(frame.mosaic for frame in frames), tuple(frame.config for frame in frames), exposures,
                               source)
        artifact_id = publish_raw_sequence(ArtifactStore(args.artifacts / "burst"), sequence)
    except (OSError, ValueError, rawpy.LibRawError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({"artifact_id": artifact_id, "frame_count": len(frames), "center_crop_xywh": crop,
                      "exposures": [item.model_dump() for item in exposures]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
