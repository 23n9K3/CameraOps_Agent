"""Estimate a global translation, then sample source frames on the reference grid."""

from dataclasses import dataclass

import cv2
import numpy as np

_IDENTITY = np.eye(2, 3, dtype=np.float32)


@dataclass(frozen=True)
class Registration:
    matrix: np.ndarray
    method: str
    ecc: float | None = None
    note: str | None = None


def luminance(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def matching_guides(reference: np.ndarray, source: np.ndarray, ceiling: float) -> tuple[np.ndarray, np.ndarray]:
    """Compress exposure-matched luminance identically, with a common clipping ceiling."""
    strength = 12.0 / ceiling
    return tuple(
        cv2.GaussianBlur(np.log1p(np.minimum(luminance(rgb), ceiling) * strength) / np.float32(np.log(13)),
                         (3, 3), 0)
        for rgb in (reference, source)
    )


def _feature_translation(reference: np.ndarray, source: np.ndarray) -> np.ndarray | None:
    detector = cv2.ORB_create(nfeatures=600, edgeThreshold=12, fastThreshold=8)
    left, left_desc = detector.detectAndCompute(np.rint(reference * 255).astype(np.uint8), None)
    right, right_desc = detector.detectAndCompute(np.rint(source * 255).astype(np.uint8), None)
    if left_desc is None or right_desc is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(left_desc, right_desc)
    if len(matches) < 6:
        return None
    matches = sorted(matches, key=lambda match: match.distance)[:80]
    offsets = np.array([np.subtract(right[m.trainIdx].pt, left[m.queryIdx].pt) for m in matches])
    translation = np.median(offsets, axis=0)
    inliers = np.linalg.norm(offsets - translation, axis=1) < 2
    if inliers.sum() < max(5, len(offsets) * 0.45):
        return None
    matrix = _IDENTITY.copy()
    matrix[:, 2] = np.median(offsets[inliers], axis=0)
    return matrix


def warp(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)


def _residual(reference: np.ndarray, source: np.ndarray, matrix: np.ndarray) -> float:
    covered = warp(np.ones(reference.shape, dtype=np.float32), matrix) > 0.999
    if covered.mean() < 0.75:
        return float("inf")
    return float(np.median(np.abs(reference[covered] - warp(source, matrix)[covered])))


def register_ecc(reference: np.ndarray, source: np.ndarray, source_valid: np.ndarray,
                 *, phase_seed: bool = False) -> Registration:
    """Return the reference-to-source transform expected by WARP_INVERSE_MAP."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 70, 1e-5)
    identity_error = _residual(reference, source, _IDENTITY)
    seed = _IDENTITY.copy()
    seeded = False
    if phase_seed and min(reference.shape) >= 4:
        window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
        # phaseCorrelate may window its inputs in-place on some OpenCV builds.
        offset, response = cv2.phaseCorrelate(reference.copy(), source.copy(), window)
        if response > 0.08 and np.isfinite(offset).all():
            seed[:, 2] = offset
            seeded = _residual(reference, source, seed) < identity_error
        if not seeded:
            seed = _IDENTITY.copy()
    try:
        score, matrix = cv2.findTransformECC(reference, source, seed.copy(), cv2.MOTION_TRANSLATION,
                                              criteria, source_valid.astype(np.uint8) * 255, 5)
        if np.isfinite(matrix).all() and np.isfinite(score) and score >= 0.65:
            if _residual(reference, source, matrix) <= _residual(reference, source, seed) + 0.002:
                return Registration(matrix, "phase_seeded_ecc" if seeded else "ecc_translation", float(score))
        note = "ECC candidate failed correlation, coverage or photometric checks"
    except cv2.error as exc:
        note = f"ECC failed ({exc.code}): {exc.err}"
    if seeded:
        return Registration(seed, "phase_translation", note=note)
    candidate = _feature_translation(reference, source)
    if candidate is not None and _residual(reference, source, candidate) < identity_error:
        return Registration(candidate, "orb_translation", note=note)
    return Registration(_IDENTITY.copy(), "identity_fallback", note=note)
