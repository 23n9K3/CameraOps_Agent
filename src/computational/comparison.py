"""Evaluate on calibrated synthetic reference coordinates, before tone mapping."""

import cv2
import numpy as np

from src.storage.artifact_store import ArtifactStore


def ground_truth_regions(store: ArtifactStore, artifact_id: str, reference: int, factor: float):
    target = store.load_array(artifact_id, "ground_truth_rgb", mmap_mode="r")[reference]
    masks_by_frame = store.load_array(artifact_id, "subject_mask", mmap_mode="r")
    union = masks_by_frame.any(axis=0)
    subject = union.copy() if (union != masks_by_frame.all(axis=0)).any() else np.zeros_like(union)
    # The swept exposure reveals motion outside the midpoint masks, including visible trails.
    mid = store.load_array(artifact_id, "ground_truth_rgb", mmap_mode="r")
    integrated = store.load_array(artifact_id, "exposure_average_rgb", mmap_mode="r")
    for sharp, blurred in zip(mid, integrated):
        subject |= (np.max(np.abs(sharp - blurred), axis=-1) > 0.02)
    if subject.any():
        subject = cv2.dilate(subject.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    masks = {
        "full_frame": np.ones(target.shape[:2], dtype=bool),
        "static_shadows": (~subject) & (target.mean(axis=-1) < 0.09),
        "motion_region": subject,
        "reference_clipped_highlights": (~subject) & (target.max(axis=-1) * factor >= 1),
    }
    return target, masks


def compare_ground_truth(store: ArtifactStore, artifact_id: str, reference: int, reference_dose: float,
                         candidates: dict[str, np.ndarray]) -> dict:
    metadata = store.load_metadata(artifact_id)
    sensor = metadata["simulation_config"]["sensor"]
    factor = sensor["photon_rate_e_s"] * reference_dose / sensor["full_well_e"]
    target, masks = ground_truth_regions(store, artifact_id, reference, factor)
    peak = max(1.0, float(target.max()))
    scores = {}
    for label, image in candidates.items():
        squared = (image.astype(np.float64) / factor - target) ** 2
        scores[label] = {}
        for region, mask in masks.items():
            count = int(mask.sum())
            if not count:
                scores[label][region] = {"pixels": 0, "rmse": None, "psnr_db": None, "reason": "region is empty"}
                continue
            mse = float(squared[mask].mean())
            scores[label][region] = {"pixels": count, "rmse": mse ** 0.5,
                                     "psnr_db": 10 * np.log10(peak**2 / mse) if mse else None,
                                     "reason": None if mse else "zero error; PSNR is positive infinity"}
    return {"valid": True, "input_artifact_id": artifact_id, "ground_truth": "ground_truth_rgb",
            "reference_index": reference, "reference_time_s": metadata["frames"][reference]["midpoint_s"],
            "evaluation_domain": "scene-linear RGB radiance before tone mapping", "psnr_peak": peak,
            "scene_to_reference_scale": factor, "scores": scores}
