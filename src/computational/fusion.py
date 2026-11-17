"""Linear RGB fusion, noise weighting, deghosting and display tone mapping."""

import cv2
import numpy as np


def normalized_average(frames: np.ndarray, weights: np.ndarray, reference: int) -> tuple[np.ndarray, np.ndarray]:
    """Fuse independently by channel, preserving the reference where no samples qualify."""
    weights = weights.astype(np.float32, copy=True)
    total = weights.sum(axis=0)
    missing = total <= 1e-12
    weights[reference][missing] = 1
    total[missing] = 1
    weights /= total
    return (frames * weights).sum(axis=0), weights


def multi_frame_average(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=0, dtype=np.float32)


def repair_isolated_cfa_sites(mosaic: np.ndarray, margin: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Replace extreme points outside their same-color neighborhood range (Night Mode)."""
    corrected = mosaic.copy()
    detected = np.zeros(mosaic.shape, dtype=bool)
    neighbors = np.ones((3, 3), dtype=np.uint8)
    neighbors[1, 1] = 0
    for row in range(2):
        for column in range(2):
            sites = np.s_[row::2, column::2]
            plane = np.ascontiguousarray(mosaic[sites])
            if min(plane.shape) < 3:
                continue
            low = cv2.erode(plane, neighbors)
            high = cv2.dilate(plane, neighbors)
            outliers = (plane > high + margin) | (plane < low - margin)
            outliers[[0, -1], :] = False
            outliers[:, [0, -1]] = False
            detected[sites] = outliers
            corrected[sites] = np.where(outliers, cv2.medianBlur(plane, 3), plane)
    return corrected, detected


def measurement_weights(signals: np.ndarray, usable: np.ndarray, doses: np.ndarray, gains: np.ndarray,
                        *, hdr: bool, denoise_strength: float, reference: int,
                        fusion_weight: float = 1.0, exposure_weight: float | None = None) -> np.ndarray:
    """Relative signal confidence, using a Poisson/read-noise proxy in reference units."""
    scale = doses[:, None, None, None]
    gain = gains[:, None, None, None]
    # Variance proxy for linear sensor signal, then scaled to reference-dose units.
    variance = (np.maximum(signals, 0.01) * gain + 0.004 * gain**2) / scale**2
    weights = usable * (1 - denoise_strength + denoise_strength / (variance + 0.015))
    preference = float(hdr) if exposure_weight is None else exposure_weight
    if preference:
        exposed = np.clip(signals, 0, 1)
        # Taper the top of the response and the weakest shadow measurements.
        suitability = np.clip((1 - exposed) / 0.2, 0, 1) * np.clip((exposed + 0.002) / 0.04, 0, 1)
        weights *= 1 - preference + preference * suitability
    if not hdr and denoise_strength < 1:
        weights[reference] += (1 - denoise_strength) * len(weights) * usable[reference]
    if fusion_weight < 1:
        reference_weights = weights[reference].copy()
        weights *= np.where(reference_weights > 1e-12, fusion_weight, 1.0)
        weights[reference] = reference_weights
    return weights


def suppress_ghosting(weights: np.ndarray, masks: np.ndarray, reference: int,
                      *, motion_aware: bool) -> np.ndarray:
    """Reject mismatched source regions, with extra reference protection in Motion-aware mode."""
    stable = ~masks
    if motion_aware:
        # A region moving in any pair is reconstructed from the sharp reference when it is valid.
        safe_reference = (weights[reference] > 1e-12) & masks.any(axis=0)[..., None]
        stable_weights = weights * stable[..., None]
        stable_weights = np.where(safe_reference[None], 0, stable_weights)
        stable_weights[reference] = weights[reference]
        return stable_weights
    return weights * stable[..., None]


def tone_map(linear: np.ndarray, exposure: float, strength: float = 1.0) -> np.ndarray:
    """Apply a smooth highlight shoulder and keep the RGB channel ratios."""
    bright = np.maximum(linear, 0) * exposure
    # A max-channel shoulder also guarantees in-gamut RGB without channel-wise clipping.
    peak = bright.max(axis=-1, keepdims=True)
    return bright / np.maximum(1 + strength * peak, peak)
