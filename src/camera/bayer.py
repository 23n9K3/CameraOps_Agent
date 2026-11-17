"""Black-level correction and bilinear interpolation in sensor-linear RGB."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .raw_config import BayerPattern, RawConfig, site_levels

_BILINEAR_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)


def normalize_mosaic(
    mosaic: NDArray[np.uint16], config: RawConfig
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Return normalized Bayer values and a saturation mask measured before clipping."""
    if mosaic.shape != (config.height, config.width) or mosaic.dtype != np.uint16:
        raise ValueError("mosaic must be a uint16 array matching the configured dimensions")
    linear = np.empty(mosaic.shape, dtype=np.float32)
    saturated = np.empty(mosaic.shape, dtype=bool)
    levels = zip(site_levels(config.black_level), site_levels(config.white_level))
    for index, (black, white) in enumerate(levels):
        sites = np.s_[index // 2::2, index % 2::2]
        values = mosaic[sites]
        saturated[sites] = values >= white
        linear[sites] = (values.astype(np.float32) - black) / (white - black)
    return np.clip(linear, 0, 1, out=linear), saturated


def demosaic(mosaic: NDArray[np.float32], pattern: BayerPattern) -> NDArray[np.float32]:
    """Interpolate only missing samples, retaining the measured values at CFA sites."""
    if mosaic.ndim != 2 or min(mosaic.shape) < 2 or mosaic.dtype != np.float32:
        raise ValueError("demosaic expects a float32 Bayer mosaic of at least 2x2")
    if pattern not in ("RGGB", "BGGR", "GRBG", "GBRG"):
        raise ValueError(f"unsupported Bayer pattern {pattern!r}")
    rgb = np.empty((*mosaic.shape, 3), dtype=np.float32)
    for channel, color in enumerate("RGB"):
        mask = np.zeros(mosaic.shape, dtype=np.float32)
        for index, site_color in enumerate(pattern):
            if site_color == color:
                mask[index // 2::2, index % 2::2] = 1
        numerator = cv2.filter2D(mosaic * mask, -1, _BILINEAR_KERNEL, borderType=cv2.BORDER_CONSTANT)
        denominator = cv2.filter2D(mask, -1, _BILINEAR_KERNEL, borderType=cv2.BORDER_CONSTANT)
        rgb[..., channel] = np.where(mask, mosaic, numerator / denominator)
    return rgb
