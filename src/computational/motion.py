"""Residual motion masks after global background alignment."""

import cv2
import numpy as np


def optical_flow(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    """Farneback displacement from reference pixels to globally aligned source pixels."""
    first = np.rint(np.clip(reference, 0, 1) * 255).astype(np.uint8)
    second = np.rint(np.clip(aligned, 0, 1) * 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 4, 19, 4, 7, 1.5, 0)


def motion_mask(reference: np.ndarray, aligned: np.ndarray, flow: np.ndarray, comparable: np.ndarray,
                *, difference_threshold: float, flow_threshold: float, padding: int,
                color_outliers: np.ndarray | None = None) -> np.ndarray:
    difference = np.abs(reference - aligned)
    magnitude = np.linalg.norm(flow, axis=-1)
    moving = (difference > difference_threshold) | (
        (difference > difference_threshold * 0.45) & (magnitude > flow_threshold)
    )
    if color_outliers is not None:
        moving |= color_outliers
    moving &= comparable
    mask = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if padding:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * padding + 1, 2 * padding + 1))
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)
