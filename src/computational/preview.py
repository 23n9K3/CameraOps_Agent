"""Create a viewable contact sheet from intermediate processing results."""

import cv2
import numpy as np

from src.visualization.preview import display_rgb, encode_png

from .fusion import tone_map


def diagnostic_preview(reference, aligned, before_deghost, mapped, mask, reference_weight,
                       *, exposure: float, frame_index: int, tone_strength: float = 1.0) -> bytes:
    height, width = mask.shape
    scale = min(1, 320 / max(height, width))
    # A readable label row can be wider than very small RAW inputs.
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb_images = [reference, aligned, before_deghost]
    tiles = [display_rgb(tone_map(cv2.resize(image, size, interpolation=cv2.INTER_AREA), exposure, tone_strength))
             for image in rgb_images]
    tiles.append(display_rgb(cv2.resize(mapped, size, interpolation=cv2.INTER_AREA)))
    for gray, interp in ((mask, cv2.INTER_NEAREST), (reference_weight.mean(axis=-1), cv2.INTER_AREA)):
        image = np.rint(gray * 255).astype(np.uint8)
        tiles.append(cv2.cvtColor(cv2.resize(image, size, interpolation=interp), cv2.COLOR_GRAY2RGB))
    labels = ["Reference", f"Aligned frame {frame_index + 1}", "Before deghosting", "Final",
              f"Motion mask, frame {frame_index + 1}", "Reference weight"]
    cards = []
    for image, label in zip(tiles, labels):
        card = cv2.copyMakeBorder(image, 0, 22, 0, max(0, 160 - size[0]), cv2.BORDER_CONSTANT, value=(23, 34, 34))
        cv2.putText(card, label, (6, size[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (244, 247, 245), 1)
        cards.append(card)
    return encode_png(np.concatenate((np.concatenate(cards[:3], axis=1), np.concatenate(cards[3:], axis=1)), axis=0))
