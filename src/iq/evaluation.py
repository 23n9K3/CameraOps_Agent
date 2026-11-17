"""Reference and blind IQ measurements with explicit validity and provenance."""

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from skimage.metrics import structural_similarity

from src.computational.registration import luminance
from src.storage.artifact_store import ArtifactStore


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    value: float | None
    unit: str
    valid: bool
    requires_ground_truth: bool = False
    method: str
    input_artifacts: list[str]
    limitations: str

    @model_validator(mode="after")
    def validate_availability(self):
        if self.valid != (self.value is not None):
            raise ValueError("metric validity must agree with the presence of a numeric value")
        return self


class IQResult(BaseModel):
    artifact_id: str
    metrics: dict[str, Metric]
    registration_checks: list[dict]
    ghosting_support_pixels: int


def _motion_support(aligned: np.ndarray, usable: np.ndarray, geometry: np.ndarray, reference: int) -> np.ndarray:
    """Identify coherent temporal outliers independently of the pipeline's motion mask."""
    anchor = aligned[reference]
    region = np.zeros(anchor.shape[:2], bool)
    kernel = np.ones((3, 3), np.uint8)
    for i, peer in enumerate(aligned):
        if i == reference:
            continue
        comparable = geometry[i] & usable[i].all(axis=-1) & usable[reference].all(axis=-1)
        difference = cv2.GaussianBlur(np.max(np.abs(peer - anchor), axis=-1), (5, 5), 0)
        signal = np.maximum(anchor.max(axis=-1), peer.max(axis=-1))
        candidate = comparable & (difference > (0.018 + 0.16 * signal))
        coherent = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        region |= coherent.astype(bool)
    return region


def evaluate_image(output: ArtifactStore, burst: ArtifactStore, artifact_id: str) -> IQResult:
    meta = output.load_metadata(artifact_id)
    final = output.load_array(artifact_id, "final_linear", mmap_mode="r")
    aligned = output.load_array(artifact_id, "aligned_rgb", mmap_mode="r")
    geometry = output.load_array(artifact_id, "aligned_geometry", mmap_mode="r")
    usable = output.load_array(artifact_id, "fusion_valid_channels", mmap_mode="r")
    mapped = output.load_array(artifact_id, "tone_mapped_rgb", mmap_mode="r")
    weights = output.load_array(artifact_id, "fusion_weights", mmap_mode="r")
    reference = meta["reference_index"]
    source_id = meta["source"].get("input_artifact_id")
    metrics = {}

    def metric(name, value, unit, method, limitations, *, ground_truth=False):
        metrics[name] = Metric(value=value, unit=unit, valid=value is not None, method=method,
                               limitations=limitations, requires_ground_truth=ground_truth,
                               input_artifacts=[artifact_id, source_id] if ground_truth and source_id else [artifact_id])

    evaluations = meta["evaluation"]
    if evaluations["valid"] and source_id:
        target = burst.load_array(source_id, "ground_truth_rgb", mmap_mode="r")[meta["source_reference_index"]]
        estimate = final / evaluations["scene_to_reference_scale"]
        peak = evaluations["psnr_peak"]
        error = float(np.mean((estimate.astype(np.float64) - target) ** 2))
        metric("psnr", float(10 * np.log10(peak**2 / error)) if error else None, "dB", "reference scene-linear MSE",
               "fixed synthetic radiometric calibration" if error else "zero error gives infinite PSNR", ground_truth=True)
        window = min(7, *(target.shape[:2]))
        window -= int(window % 2 == 0)
        ssim = float(structural_similarity(target, estimate, data_range=peak, channel_axis=-1, win_size=window)) if window >= 3 else None
        metric("ssim", ssim, "1", "reference scene-linear RGB SSIM",
               "ground truth at reference exposure midpoint" if ssim is not None else "image too small for SSIM", ground_truth=True)
        target_edges = cv2.Laplacian(luminance(target), cv2.CV_32F)
        detail = np.abs(target_edges) >= max(float(np.percentile(np.abs(target_edges), 90)), 1e-4)
        residual = cv2.Laplacian(luminance(estimate), cv2.CV_32F) - target_edges
        agreement = 1 / (1 + float(np.mean(np.abs(residual[detail])) / np.mean(np.abs(target_edges[detail])))) if detail.any() else None
        metric("detail_preservation", agreement, "1", "Laplacian edge agreement against Ground Truth",
               "evaluated on strong reference edges" if detail.any() else "reference has no usable edges", ground_truth=True)
    else:
        for name, unit in (("psnr", "dB"), ("ssim", "1"), ("detail_preservation", "1")):
            metric(name, None, unit, "calibrated reference comparison",
                   evaluations.get("reason", "no calibrated input artifact ID"), ground_truth=True)

    gray = luminance(final)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = np.abs(cv2.Laplacian(smooth, cv2.CV_32F))
    flat = (gradient < 0.006) & (gray > 0.005) & (gray < 0.85)
    residual = (gray - cv2.GaussianBlur(gray, (3, 3), 0))[flat]
    noise = float(1.4826 * np.median(np.abs(residual - np.median(residual)))) if len(residual) >= 32 else None
    metric("noise_std", noise, "reference-dose signal", "MAD of high-pass luminance in smooth regions",
           "blind residual proxy, sensitive to detail" if noise is not None else "insufficient smooth support")
    motion = _motion_support(aligned, usable, geometry, reference)
    support = int(motion.sum())
    contamination = float(np.abs(final[motion] - aligned[reference][motion]).mean()) if support >= 32 else None
    metric("ghosting_score", contamination, "reference-dose signal", "output deviation from reference in independent temporal-outlier regions",
           "proxy may include valid subject changes or misregistration" if contamination is not None else "insufficient coherent motion support")

    checks = []
    for row in meta["registration"]:
        index = row["frame"]
        if index != reference:
            checks.append({**row, "mean_fusion_weight": float(weights[index].mean())})
    residuals = [row["alignment_residual"] for row in checks if row["alignment_residual"] is not None]
    metric("alignment_error", max(residuals) if residuals else None, "guide intensity",
           "worst median residual over co-visible unsaturated pixels",
           "photometric proxy, includes exposure mismatch" if residuals else "no comparable frame pairs")
    metric("clipping_ratio", float((mapped.max(axis=-1) >= 0.999).mean()), "fraction", "pixels at final linear-display white",
           "tone mapping compresses sensor highlights")
    unresolved = (~usable.any(axis=0)) & ~usable[reference]
    metric("unrecoverable_clipping", float(unresolved.any(axis=-1).mean()), "fraction",
           "any channel clipped in reference with no valid exposure", "saturation flags propagate through demosaicing")
    metric("runtime", meta["runtime_ms"]["total_before_artifact_write"], "ms", "recorded photography tool runtime",
           "excludes input I/O and artifact publication")
    metric("output_median_luminance", float(np.median(luminance(mapped))), "display-linear signal", "median output luminance",
           "scene intent affects the appropriate brightness")
    metric("mtf50", None, "cycles/pixel", "slanted-edge MTF50", "no qualified slanted-edge target supplied", ground_truth=True)
    return IQResult(artifact_id=artifact_id, metrics=metrics, registration_checks=checks, ghosting_support_pixels=support)
