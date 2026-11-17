"""Display-only encoding shared by the CLI, simulator and viewer."""

import cv2
import numpy as np
from numpy.typing import NDArray


def display_rgb(linear: NDArray[np.float32]) -> NDArray[np.uint8]:
    rgb = np.clip(linear, 0, 1)
    display = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.rint(display * 255).astype(np.uint8)


def encode_png(rgb: NDArray[np.uint8]) -> bytes:
    encoded, png = cv2.imencode(".png", rgb[..., ::-1])
    if not encoded:
        raise ValueError("could not encode RGB preview")
    return png.tobytes()
