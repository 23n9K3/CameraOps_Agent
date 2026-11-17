"""Define fixed Ground Truth regions and measurable optimization objectives."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.computational.comparison import ground_truth_regions
from src.computational.fusion import tone_map
from src.computational.registration import luminance
from src.storage.artifact_store import ArtifactStore

OBJECTIVES = {
    "noise_residual_std": "Standard deviation of luminance error in smooth static shadows, in scene-linear radiance",
    "detail_error": "Mean absolute Laplacian error on strong static reference edges, divided by reference edge magnitude",
    "ghosting_rmse": "Scene-linear RGB RMSE in the known moving subject sweep",
    "color_delta_e": "Mean CIE76 delta E against Ground Truth rendered with the fixed baseline tone curve",
}


def _lab(display_linear: np.ndarray) -> np.ndarray:
    linear = np.clip(display_linear, 0, 1)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return cv2.cvtColor(srgb.astype(np.float32, copy=False), cv2.COLOR_RGB2Lab)


@dataclass(frozen=True)
class ObjectiveTarget:
    rgb: np.ndarray
    grayscale: np.ndarray
    edges: np.ndarray
    lab: np.ndarray
    factor: float
    masks: dict[str, np.ndarray]

    @classmethod
    def prepare(cls, bursts: ArtifactStore, artifact_id: str, reference: int, factor: float,
                tone_exposure: float, tone_strength: float):
        target, regions = ground_truth_regions(bursts, artifact_id, reference, factor)
        grayscale = luminance(target)
        edges = cv2.Laplacian(grayscale, cv2.CV_32F)
        static = ~regions["motion_region"]
        # Erode support so high-contrast neighbors cannot dominate the shadow residual.
        shadows = cv2.erode((regions["static_shadows"] & (np.abs(edges) < 0.008)).astype(np.uint8),
                             np.ones((3, 3), np.uint8)).astype(bool)
        target_edges = np.abs(edges)
        detail = static & (target_edges >= max(0.01, float(np.percentile(target_edges, 90))))
        masks = {"noise_residual_std": shadows, "detail_error": detail, "ghosting_rmse": regions["motion_region"],
                 "color_delta_e": np.ones_like(shadows)}
        return cls(target, grayscale, edges, _lab(tone_map(target * factor, tone_exposure, tone_strength)), factor, masks)

    @property
    def active(self) -> list[str]:
        return [name for name, mask in self.masks.items() if mask.sum() >= 32]

    def score(self, final: np.ndarray, tone_mapped: np.ndarray) -> dict[str, float]:
        estimate = final / self.factor
        gray = luminance(estimate)
        active = self.active
        scores = {}
        if "noise_residual_std" in active:
            mask = self.masks["noise_residual_std"]
            scores["noise_residual_std"] = float(np.std((gray - self.grayscale)[mask], dtype=np.float64))
        if "detail_error" in active:
            mask = self.masks["detail_error"]
            edges = cv2.Laplacian(gray, cv2.CV_32F)
            scores["detail_error"] = float(np.abs((edges - self.edges)[mask]).mean() / np.abs(self.edges[mask]).mean())
        if "ghosting_rmse" in active:
            error = (estimate - self.rgb)[self.masks["ghosting_rmse"]]
            scores["ghosting_rmse"] = float(np.sqrt(np.mean(error**2, dtype=np.float64)))
        if "color_delta_e" in active:
            scores["color_delta_e"] = float(np.linalg.norm(_lab(tone_mapped) - self.lab, axis=-1).mean(dtype=np.float64))
        return scores
