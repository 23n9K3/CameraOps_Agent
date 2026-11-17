"""Integrate moving scene radiance, simulate sensor readout, and persist a Burst."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.scene.synthetic import SceneConfig, SyntheticScene
from src.storage.artifact_store import ArtifactStore
from src.visualization.preview import display_rgb, encode_png

from .bayer import demosaic, normalize_mosaic
from .capture_plan import CapturePlan
from .sensor_noise import SensorConfig, SensorNoiseModel, mosaic_rgb


class SimulationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(ge=0, le=2**32 - 1)
    motion_samples: int = Field(default=12, ge=1, le=48)
    scene: SceneConfig
    sensor: SensorConfig
    capture_plan: CapturePlan

    @model_validator(mode="after")
    def validate_size(self) -> "SimulationConfig":
        if self.scene.width * self.scene.height * self.capture_plan.frame_count > 4_194_304:
            raise ValueError("Burst exceeds 4,194,304 total sensor sites")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "SimulationConfig":
        with Path(path).open(encoding="utf-8-sig") as source:
            return cls.model_validate(yaml.safe_load(source))


def _contact_sheet(previews: list[np.ndarray], frames: list[dict]) -> bytes:
    cards = []
    for image, frame in zip(previews, frames):
        height, width = image.shape[:2]
        scale = 256 / max(height, width)
        thumb = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
        padding = 256 - thumb.shape[1]
        card = cv2.copyMakeBorder(thumb, 0, 26, padding // 2, padding - padding // 2,
                                 cv2.BORDER_CONSTANT, value=(22, 27, 34))
        label = f'{frame["index"] + 1:02d}   {frame["exposure_s"] * 1000:g} ms   x{frame["analog_gain"]:g}'
        cv2.putText(card, label, (9, len(thumb) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (244, 245, 247), 1)
        cards.append(card)
    columns = min(4, len(cards))
    cards.extend(np.zeros_like(cards[0]) for _ in range(-len(cards) % columns))
    return encode_png(np.concatenate([
        np.concatenate(cards[start:start + columns], axis=1) for start in range(0, len(cards), columns)
    ], axis=0))


def simulate_capture(config: SimulationConfig, store: ArtifactStore) -> str:
    """Return the Artifact ID of a batch with RAW, radiance truth, masks and configuration."""
    started = perf_counter()
    plan = config.capture_plan
    shape = config.scene.height, config.scene.width
    layout = config.sensor.raw_layout(config.scene.width, config.scene.height)
    scene = SyntheticScene(config.scene)
    calibration_seed, *frame_seeds = np.random.SeedSequence(config.seed).spawn(plan.frame_count + 1)
    sensor = SensorNoiseModel(config.sensor, shape, np.random.default_rng(calibration_seed))
    raw = np.empty((plan.frame_count, *shape), dtype=np.uint16)
    midpoint_rgb = np.empty((plan.frame_count, *shape, 3), dtype=np.float32)
    integrated_rgb = np.empty_like(midpoint_rgb)
    subject_mask = np.empty(raw.shape, dtype=bool)
    previews, frames = [], []
    start_s = 0.0

    for index, (exposure_s, gain, seed) in enumerate(zip(plan.exposure_times, plan.gains, frame_seeds)):
        midpoint_s = start_s + exposure_s / 2
        midpoint_rgb[index], subject_mask[index] = scene.render(midpoint_s)
        if config.motion_samples == 1:
            integrated_rgb[index] = midpoint_rgb[index]
        else:
            accumulated = np.zeros((*shape, 3), dtype=np.float32)
            for sample in range(config.motion_samples):
                timestamp = start_s + (sample + 0.5) * exposure_s / config.motion_samples
                radiance, _ = scene.render(timestamp)
                accumulated += radiance
            integrated_rgb[index] = accumulated / config.motion_samples
        mosaic = mosaic_rgb(integrated_rgb[index], layout.bayer_pattern)
        raw[index] = sensor.expose(mosaic, exposure_s, gain, np.random.default_rng(seed))
        linear, saturated = normalize_mosaic(raw[index], layout)
        previews.append(display_rgb(demosaic(linear, layout.bayer_pattern)))
        frames.append({
            "index": index, "start_s": start_s, "midpoint_s": midpoint_s, "end_s": start_s + exposure_s,
            "exposure_s": exposure_s, "analog_gain": gain, "mean_raw_dn": float(raw[index].mean()),
            "saturated_site_ratio": float(saturated.mean()),
            "subject_displacement_during_exposure_px": config.scene.velocity_px_s * exposure_s,
            "reference": index == plan.reference_frame,
        })
        start_s += exposure_s + plan.inter_frame_gap_s

    preview = _contact_sheet(previews, frames)
    metadata = {
        "artifact_type": "synthetic_raw_burst", "source_format": "simulated_sensor", "schema_version": 1,
        "frame_count": plan.frame_count, "raw_config": layout.model_dump(mode="json"),
        "simulation_config": config.model_dump(mode="json"), "frames": frames,
        "hot_pixel_count": int(sensor.maps.hot.sum()), "dead_pixel_count": int(sensor.maps.dead.sum()),
        "truth_color_space": "scene-linear RGB radiance", "numpy_version": np.__version__,
        "simulation_and_preview_ms": (perf_counter() - started) * 1000,
    }
    return store.save({
        "raw_burst": raw, "ground_truth_rgb": midpoint_rgb, "exposure_average_rgb": integrated_rgb,
        "subject_mask": subject_mask, "hot_pixel_mask": sensor.maps.hot, "dead_pixel_mask": sensor.maps.dead,
        "prnu_map": sensor.maps.prnu, "dsnu_e_s_map": sensor.maps.dsnu_e_s,
    }, metadata, preview_png=preview)
