"""An analytic, textured scene with a moving foreground and known visibility."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class SceneConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    width: int = Field(default=320, ge=32, le=2048)
    height: int = Field(default=192, ge=32, le=2048)
    velocity_px_s: float = Field(default=540, ge=-4000, le=4000)


class SyntheticScene:
    """Render radiance in linear RGB at a timestamp in seconds."""

    def __init__(self, config: SceneConfig):
        self.config = config
        y, x = np.indices((config.height, config.width), dtype=np.float32)
        self.x, self.y = x, y
        u, v = x / config.width, y / config.height
        brick = ((x + (y // 14 % 2) * 11) % 22 < 1.5) | (y % 14 < 1.5)
        wall = (0.035 + 0.32 * u + 0.08 * v) * np.where(brick, 0.57, 1)
        background = wall[..., None] * np.array([1.12, 0.93, 0.75])
        window = (u > 0.70) & (u < 0.91) & (v > 0.14) & (v < 0.46)
        panes = window & (np.abs(u - 0.805) > 0.008) & (np.abs(v - 0.30) > 0.010)
        background[window] = [0.18, 0.14, 0.095]
        background[panes] = [1.35, 1.02, 0.65]
        shadow = (u < 0.23) & (v > 0.62)
        background[shadow] *= 0.17
        for left, color in [(0.30, [0.43, 0.07, 0.06]), (0.42, [0.06, 0.34, 0.12]),
                            (0.54, [0.06, 0.13, 0.50])]:
            background[(u > left) & (u < left + 0.08) & (v > 0.10) & (v < 0.26)] = color
        self.background = background.astype(np.float32)

    def render(self, time_s: float) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        h, w = self.config.height, self.config.width
        center = 0.25 * w + self.config.velocity_px_s * time_s
        u = (self.x - center) / w
        v = self.y / h
        head = (u / 0.028) ** 2 + ((v - 0.45) / 0.051) ** 2 < 1
        torso = (np.abs(u) < 0.045) & (v >= 0.50) & (v < 0.70)
        stride = 0.013 + 0.012 * np.sin(16 * time_s * abs(self.config.velocity_px_s) / 540)
        legs = (np.abs(np.abs(u) - stride) < 0.012) & (v >= 0.69) & (v < 0.86)
        mask = head | torso | legs
        rgb = self.background.copy()
        rgb[head] = [0.53, 0.31, 0.21]
        rgb[torso] = [0.05, 0.32, 0.43]
        rgb[torso & (self.y % 7 < 2)] *= 0.65
        rgb[legs] = [0.055, 0.067, 0.096]
        return rgb, mask
