"""Inputs from a P2 Burst artifact or a sequence of offline camera RAW files."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.camera.raw_config import RawConfig
from src.camera.raw_reader import read_raw
from src.storage.artifact_store import ArtifactStore


class Exposure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    exposure_s: float = Field(gt=0)
    analog_gain: float = Field(gt=0)

    @property
    def dose(self) -> float:
        return self.exposure_s * self.analog_gain


class FileExposure(Exposure):
    path: Path


class FileSequence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frames: list[FileExposure] = Field(min_length=1)
    raw_config: RawConfig | None = None
    reference_frame: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class RawSequence:
    mosaics: tuple[np.ndarray, ...]
    layouts: tuple[RawConfig, ...]
    exposures: tuple[Exposure, ...]
    source: dict
    reference_frame: int | None = None

    def __post_init__(self):
        if not self.mosaics or len(self.mosaics) != len(self.layouts) or len(self.mosaics) != len(self.exposures):
            raise ValueError("each RAW frame requires a layout and exposure metadata")
        shape = self.mosaics[0].shape
        if any(raw.shape != shape for raw in self.mosaics):
            raise ValueError("all RAW frames must share dimensions")
        doses = np.array([frame.dose for frame in self.exposures])
        if not np.isfinite(doses).all() or (doses <= 0).any():
            raise ValueError("exposure times and gains must yield finite, positive doses")
        if self.reference_frame is not None and not 0 <= self.reference_frame < len(self.mosaics):
            raise ValueError("reference_frame must refer to a frame in the sequence")


def read_file_sequence(path: str | Path) -> RawSequence:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as source:
        manifest = FileSequence.model_validate(yaml.safe_load(source))
    files = [(path.parent / frame.path).resolve() for frame in manifest.frames]
    decoded = [read_raw(filename, manifest.raw_config) for filename in files]
    exposures = tuple(Exposure(exposure_s=entry.exposure_s, analog_gain=entry.analog_gain) for entry in manifest.frames)
    return RawSequence(tuple(frame.mosaic for frame in decoded), tuple(frame.config for frame in decoded), exposures,
                       {"input_manifest": str(path), "files": [str(file) for file in files],
                        "formats": [frame.source_format for frame in decoded]}, manifest.reference_frame)


def read_burst(store: ArtifactStore, artifact_id: str) -> RawSequence:
    metadata = store.load_metadata(artifact_id)
    kind = metadata.get("artifact_type")
    if kind not in ("synthetic_raw_burst", "imported_raw_burst"):
        raise ValueError("input artifact is not a supported RAW Burst")
    raw = store.load_array(artifact_id, "raw_burst", mmap_mode="r")
    if kind == "synthetic_raw_burst":
        layouts = (RawConfig.model_validate(metadata["raw_config"]),) * len(raw)
        reference = metadata["simulation_config"]["capture_plan"]["reference_frame"]
    else:
        layouts = tuple(RawConfig.model_validate(layout) for layout in metadata["raw_layouts"])
        reference = metadata["reference_frame"]
    exposures = tuple(Exposure(exposure_s=frame["exposure_s"], analog_gain=frame["analog_gain"])
                      for frame in metadata["frames"])
    source = {"input_artifact_id": artifact_id, "source_format": metadata["source_format"]}
    if color_matrix := metadata.get("source", {}).get("display_color_matrix"):
        source["display_color_matrix"] = color_matrix
    return RawSequence(tuple(raw), layouts, exposures, source, reference)


def select_frames(sequence: RawSequence, indices: tuple[int, ...], reference: int) -> RawSequence:
    return replace(sequence, mosaics=tuple(sequence.mosaics[i] for i in indices),
                   layouts=tuple(sequence.layouts[i] for i in indices),
                   exposures=tuple(sequence.exposures[i] for i in indices), reference_frame=indices.index(reference),
                   source={**sequence.source, "frame_indices": list(indices)})


def import_file_sequence(store: ArtifactStore, path: str | Path) -> str:
    return publish_raw_sequence(store, read_file_sequence(path))


def publish_raw_sequence(store: ArtifactStore, sequence: RawSequence) -> str:
    return store.save({"raw_burst": np.stack(sequence.mosaics)}, {
        "artifact_type": "imported_raw_burst", "source_format": "offline_raw_files", "source": sequence.source,
        "reference_frame": sequence.reference_frame,
        "raw_layouts": [layout.model_dump(mode="json") for layout in sequence.layouts],
        "frames": [entry.model_dump(mode="json") for entry in sequence.exposures],
    })
