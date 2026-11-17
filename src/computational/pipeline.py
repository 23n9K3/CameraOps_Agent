"""Night, HDR, and Motion-aware processing on a shared sensor-linear pipeline."""

from time import perf_counter
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.camera.bayer import demosaic, normalize_mosaic
from src.storage.artifact_store import ArtifactStore
from src.visualization.preview import display_rgb, encode_png

from .comparison import compare_ground_truth
from .fusion import (measurement_weights, multi_frame_average, normalized_average, repair_isolated_cfa_sites,
                     suppress_ghosting, tone_map)
from .motion import motion_mask, optical_flow
from .preview import diagnostic_preview
from .registration import matching_guides, register_ecc, warp
from .sequence import RawSequence


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    mode: Literal["night", "hdr", "motion"]
    reference_frame: int | None = Field(default=None, ge=0)
    motion_threshold: float = Field(default=0.085, gt=0, le=1)
    flow_threshold_px: float = Field(default=1.3, gt=0, le=100)
    tone_exposure: float | None = Field(default=None, gt=0, le=64)
    denoise_strength: float = Field(default=1, ge=0, le=1)
    fusion_weight: float = Field(default=1, ge=0, le=1)
    exposure_weight: float | None = Field(default=None, ge=0, le=1)
    tone_mapping_strength: float = Field(default=1, ge=0.1, le=3)
    spatial_sigma: float = Field(default=0, ge=0, le=1)
    registration: bool = True
    registration_method: Literal["ecc", "phase_ecc"] = "ecc"
    deghost: bool = True
    protect_motion_reference: bool = False
    repair_isolated_sites: bool | None = None


def _reference(sequence: RawSequence, config: PipelineConfig) -> int:
    if config.reference_frame is not None:
        return config.reference_frame
    if sequence.reference_frame is not None:
        return sequence.reference_frame
    if config.mode == "motion":
        return min(range(len(sequence.exposures)), key=lambda i: sequence.exposures[i].exposure_s)
    if config.mode == "hdr":
        order = sorted(range(len(sequence.exposures)), key=lambda i: sequence.exposures[i].dose)
        return order[(len(order) - 1) // 2]
    return 0


def _preprocess(sequence: RawSequence, repair: bool):
    linear, rgb, saturated_sites, clipped_channels, repairs = [], [], [], [], []
    for raw, layout in zip(sequence.mosaics, sequence.layouts):
        mosaic, saturation = normalize_mosaic(raw, layout)
        linear.append(mosaic)
        corrected, repaired = repair_isolated_cfa_sites(mosaic) if repair else (mosaic, np.zeros_like(saturation))
        rgb.append(demosaic(corrected, layout.bayer_pattern))
        repairs.append(repaired)
        saturated_sites.append(saturation)
        # Interpolation propagates saturation only to the channel influenced by a clipped CFA site.
        clipped_channels.append(demosaic((saturation & ~repaired).astype(np.float32), layout.bayer_pattern) > 0)
    return np.stack(linear), np.stack(rgb), np.stack(saturated_sites), np.stack(clipped_channels), np.stack(repairs)


def _align(rgb: np.ndarray, clipped: np.ndarray, scales: np.ndarray, reference: int, config: PipelineConfig):
    aligned = np.empty_like(rgb)
    usable = np.empty_like(clipped)
    geometry = np.ones(rgb.shape[:3], dtype=bool)
    masks = np.zeros(rgb.shape[:3], dtype=bool)
    flows = np.zeros((*rgb.shape[:3], 2), dtype=np.float32)
    transforms = np.repeat(np.eye(2, 3, dtype=np.float32)[None], len(rgb), axis=0)
    records = []
    reference_rgb = rgb[reference]
    reference_valid = ~clipped[reference].any(axis=-1)
    for i, source in enumerate(rgb):
        if i == reference:
            aligned[i], usable[i] = source, ~clipped[i]
            records.append({"frame": i, "method": "reference", "ecc": None, "note": None,
                            "overlap_ratio": 1.0, "alignment_residual": 0.0})
            continue
        ceiling = min(1, 1 / scales[i])
        ref_guide, source_guide = matching_guides(reference_rgb, source, ceiling)
        reg = (register_ecc(ref_guide, source_guide, ~clipped[i].any(axis=-1),
                            phase_seed=config.registration_method == "phase_ecc") if config.registration else None)
        if reg:
            transforms[i] = reg.matrix
        matrix = transforms[i]
        aligned[i] = warp(source, matrix)
        geometry[i] = warp(np.ones(source.shape[:2], dtype=np.float32), matrix) > 0.999
        usable[i] = (warp((~clipped[i]).astype(np.float32), matrix) > 0.999) & geometry[i][..., None]
        warped_guide = warp(source_guide, matrix)
        comparable = reference_valid & usable[i].all(axis=-1)
        # Use identical values where clipping or shifted borders prevent a fair flow match.
        flows[i] = optical_flow(ref_guide, np.where(comparable, warped_guide, ref_guide))
        color_outliers = None
        if config.mode == "motion" or config.protect_motion_reference:
            difference_rgb = np.max(np.abs(reference_rgb - aligned[i]), axis=-1)
            color_outliers = cv2.GaussianBlur(difference_rgb, (5, 5), 0) > (0.32 * config.motion_threshold * ceiling)
        masks[i] = motion_mask(ref_guide, warped_guide, flows[i], comparable,
                               difference_threshold=config.motion_threshold, flow_threshold=config.flow_threshold_px,
                               padding=4 if config.mode == "motion" or config.protect_motion_reference else 2,
                               color_outliers=color_outliers)
        difference = np.abs(ref_guide - warped_guide)
        # Measure on common unsaturated support, independent of the estimated motion mask.
        records.append({"frame": i, "method": reg.method if reg else "disabled", "ecc": reg.ecc if reg else None,
                        "note": reg.note if reg else None, "overlap_ratio": float(geometry[i].mean()),
                        "alignment_residual": float(np.median(difference[comparable])) if comparable.any() else None})
    return aligned, usable, geometry, masks, flows, transforms, records


def run_pipeline(sequence: RawSequence, config: PipelineConfig, store: ArtifactStore,
                 *, truth_store: ArtifactStore | None = None) -> str:
    started = perf_counter()
    reference = _reference(sequence, config)
    if reference >= len(sequence.mosaics):
        raise ValueError("reference_frame must refer to a frame in the sequence")
    exposures = sequence.exposures
    dose = exposures[reference].dose
    ratios = [entry.dose / dose for entry in exposures]
    if any(not 1e-8 <= value <= 1e8 for value in ratios):
        raise ValueError("relative exposure doses exceed the supported floating-point range")
    if any(not 1e-8 <= entry.analog_gain <= 1e8 for entry in exposures):
        raise ValueError("analog gains exceed the supported floating-point range")
    scales = np.array(ratios, dtype=np.float32)
    gains = np.array([entry.analog_gain for entry in exposures], dtype=np.float32)
    repair = config.repair_isolated_sites if config.repair_isolated_sites is not None else config.mode == "night"
    linear, sensor_rgb, saturated, clipped, repairs = _preprocess(sequence, repair)
    rgb = sensor_rgb / scales[:, None, None, None]
    preprocess_done = perf_counter()

    aligned, usable, geometry, masks, flows, transforms, records = _align(rgb, clipped, scales, reference, config)
    align_done = perf_counter()

    signals = aligned * scales[:, None, None, None]
    weight_settings = {"hdr": config.mode == "hdr", "denoise_strength": config.denoise_strength,
                       "fusion_weight": config.fusion_weight, "exposure_weight": config.exposure_weight,
                       "reference": reference}
    weights = measurement_weights(signals, usable, scales, gains, **weight_settings)
    can_reconstruct = (weights > 1e-12).any(axis=0)
    without_deghost, _ = normalized_average(aligned, weights, reference)
    if config.deghost:
        weights = suppress_ghosting(weights, masks, reference,
                                    motion_aware=config.mode == "motion" or config.protect_motion_reference)
    # Restore the original exposure weights where motion rejection would leave only a clipped reference.
    clipped_fallback = ((weights <= 1e-12).all(axis=0) & can_reconstruct & clipped[reference])
    if clipped_fallback.any():
        original_weights = measurement_weights(signals, usable, scales, gains, **weight_settings)
        weights[:, clipped_fallback] = original_weights[:, clipped_fallback]
    fused, normalized_weights = normalized_average(aligned, weights, reference)
    if config.spatial_sigma > 0:
        final_linear = cv2.bilateralFilter(fused, 5, config.spatial_sigma, 1.2)
    else:
        final_linear = fused
    average = multi_frame_average(rgb)
    fusion_done = perf_counter()

    tone_exposure = config.tone_exposure or {"night": 4.0, "hdr": 2.0, "motion": 2.5}[config.mode]
    color_matrix = sequence.source.get("display_color_matrix")
    display_linear = final_linear
    if color_matrix is not None:
        color_matrix = np.asarray(color_matrix, dtype=np.float32)
        display_linear = cv2.transform(final_linear, color_matrix)
    mapped = tone_map(display_linear, tone_exposure, config.tone_mapping_strength)
    preview = encode_png(display_rgb(mapped))
    candidates = np.flatnonzero(np.arange(len(masks)) != reference)
    inspected = int(candidates[np.argmax(masks[candidates].sum(axis=(1, 2)))]) if len(candidates) else reference
    diagnostic_inputs = (rgb[reference], aligned[inspected], without_deghost)
    if color_matrix is not None:
        diagnostic_inputs = tuple(cv2.transform(image, color_matrix) for image in diagnostic_inputs)
    diagnostic = diagnostic_preview(*diagnostic_inputs, mapped,
                                    masks[inspected], normalized_weights[reference], exposure=tone_exposure,
                                    frame_index=inspected, tone_strength=config.tone_mapping_strength)
    tone_done = perf_counter()
    source_id = sequence.source.get("input_artifact_id")
    frame_indices = sequence.source.get("frame_indices", list(range(len(sequence.mosaics))))
    source_reference = frame_indices[reference]
    evaluation = (
        compare_ground_truth(truth_store, source_id, source_reference, dose, {
            "single_reference": demosaic(linear[reference], sequence.layouts[reference].bayer_pattern) if repair else rgb[reference],
            "single_preprocessed": rgb[reference], "unregistered_average": average,
            "fusion_without_deghosting": without_deghost, "final_fusion": final_linear,
        }) if truth_store is not None and source_id and sequence.source.get("source_format") == "simulated_sensor" else
        {"valid": False, "reason": "no calibrated, reference-aligned Ground Truth supplied"}
    )
    finished = perf_counter()
    metadata = {
        "schema_version": 1, "artifact_type": "computational_photography", "source": sequence.source,
        "pipeline": config.model_dump(mode="json"), "tone_exposure": tone_exposure,
        "reference_index": reference, "source_reference_index": source_reference, "dose_scales": scales.tolist(),
        "diagnostic_frame_index": inspected,
        "raw_layouts": [layout.model_dump(mode="json") for layout in sequence.layouts],
        "exposures": [entry.model_dump(mode="json") for entry in exposures], "registration": records,
        "transform_direction": "reference grid to input frame; OpenCV WARP_INVERSE_MAP",
        "flow_direction": "reference grid to globally aligned input, (dx, dy) in pixels",
        "fusion_color_space": "sensor-linear RGB normalized to reference exposure dose",
        "isolated_site_repair_enabled": repair, "isolated_sites_repaired": repairs.sum(axis=(1, 2)).tolist(),
        "motion_rejection_fallback_channels": int(clipped_fallback.sum()),
        "unrecoverable_reference_clipped_channels": int((~can_reconstruct & clipped[reference]).sum()),
        "runtime_ms": {
            "raw_preprocessing": (preprocess_done - started) * 1000,
            "registration_and_motion": (align_done - preprocess_done) * 1000,
            "fusion_and_baselines": (fusion_done - align_done) * 1000, "tone_and_preview": (tone_done - fusion_done) * 1000,
            "ground_truth_comparison": (finished - tone_done) * 1000, "total_before_artifact_write": (finished - started) * 1000,
        }, "evaluation": evaluation,
    }
    return store.save({
        "linear_bayer": linear, "saturated_cfa_sites": saturated, "repaired_cfa_sites": repairs,
        "demosaiced_sensor_rgb": sensor_rgb,
        "aligned_rgb": aligned, "aligned_geometry": geometry, "fusion_valid_channels": usable,
        "transforms": transforms, "optical_flow": flows, "motion_masks": masks,
        "fusion_weights": normalized_weights, "unregistered_average": average,
        "fusion_without_deghosting": without_deghost, "fused_linear": fused,
        "final_linear": final_linear, "tone_mapped_rgb": mapped,
    }, metadata, preview_png=preview, previews={"diagnostics": diagnostic})
