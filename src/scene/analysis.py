"""Image-derived luminance, exposure, detail and motion observations."""

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from src.camera.bayer import demosaic, normalize_mosaic
from src.computational.motion import optical_flow
from src.computational.registration import luminance, matching_guides, register_ecc, warp
from src.computational.sequence import Exposure, RawSequence


class SceneFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    brightness_median: float
    shadow_ratio: float
    highlight_ratio: float
    luminance_p99: float
    laplacian_variance: float
    histogram_16: list[int]
    motion_fraction: float | None
    moving_flow_p90_px: float | None
    exposures: list[Exposure]
    saturation_by_frame: list[float]
    frame_sharpness: list[float]


def _preview_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1, 384 / max(height, width))
    return cv2.resize(image, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)


def describe_scene(reference: np.ndarray, peer: np.ndarray | None, *, exposures: list[Exposure],
                   saturated: list[float], sharpness: list[float], align: bool) -> SceneFeatures:
    gray = luminance(reference)
    motion, p90 = None, None
    if peer is not None:
        left, right = matching_guides(reference, peer, 1.0)
        if align:
            registration = register_ecc(left, right, luminance(peer) < 0.95)
            right = warp(right, registration.matrix)
            comparable = warp(np.ones(left.shape, np.float32), registration.matrix) > 0.999
        else:
            comparable = np.ones(left.shape, bool)
        displacement = optical_flow(left, np.where(comparable, right, left))
        magnitude = np.linalg.norm(displacement, axis=-1)
        moving = comparable & (np.abs(left - right) > 0.085)
        moving = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        motion = float(moving.mean())
        p90 = float(np.percentile(magnitude[moving], 90)) if moving.any() else 0.0
    return SceneFeatures(brightness_median=float(np.median(gray)), shadow_ratio=float((gray < 0.08).mean()),
                         highlight_ratio=float((gray > 0.9).mean()), luminance_p99=float(np.percentile(gray, 99)),
                         laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                         histogram_16=np.histogram(np.clip(gray, 0, 1), bins=16, range=(0, 1))[0].tolist(),
                         motion_fraction=motion, moving_flow_p90_px=p90,
                         exposures=exposures, saturation_by_frame=saturated, frame_sharpness=sharpness)


def describe_burst(sequence: RawSequence) -> SceneFeatures:
    previews, ratios = [], []
    for raw, layout in zip(sequence.mosaics, sequence.layouts):
        linear, saturation = normalize_mosaic(raw, layout)
        previews.append(_preview_size(demosaic(linear, layout.bayer_pattern)))
        ratios.append(float(saturation.mean()))
    reference = sequence.reference_frame if sequence.reference_frame is not None else 0
    sharpness = [float(cv2.Laplacian(luminance(image), cv2.CV_32F).var()) for image in previews]
    peer = None
    if len(previews) > 1:
        index = max(range(len(previews)), key=lambda i: abs(i - reference))
        peer = previews[index] * (sequence.exposures[reference].dose / sequence.exposures[index].dose)
    return describe_scene(previews[reference], peer, exposures=list(sequence.exposures), saturated=ratios,
                          sharpness=sharpness, align=True)
