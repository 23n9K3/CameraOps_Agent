"""Electron-domain sensor model and reproducible fixed-pattern calibration."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .raw_config import BayerPattern, RawConfig


class SensorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    bit_depth: Literal[8, 10, 12, 14, 16] = 12
    bayer_pattern: BayerPattern = "RGGB"
    black_level: int = Field(default=64, ge=0)
    full_well_e: float = Field(default=12000, ge=1, le=1e9)
    photon_rate_e_s: float = Field(default=300000, gt=0, le=1e9)
    shot_noise: bool = True
    read_noise_e: float = Field(default=3.5, ge=0, le=1e6)
    hot_pixel_fraction: float = Field(default=0.001, ge=0, le=1)
    dead_pixel_fraction: float = Field(default=0.001, ge=0, le=1)
    hot_pixel_rate_e_s: float = Field(default=220000, ge=0, le=1e9)
    dsnu_e_s: float = Field(default=45, ge=0, le=1e9)
    prnu_fraction: float = Field(default=0.014, ge=0, le=1)
    row_noise_e: float = Field(default=1.4, ge=0, le=1e6)
    column_noise_e: float = Field(default=0.8, ge=0, le=1e6)

    @model_validator(mode="after")
    def validate_sensor(self) -> "SensorConfig":
        if self.black_level >= (1 << self.bit_depth) - 1:
            raise ValueError("black_level must be below the ADC white code")
        if self.hot_pixel_fraction + self.dead_pixel_fraction > 1:
            raise ValueError("the combined defective pixel fraction cannot exceed one")
        return self

    def raw_layout(self, width: int, height: int) -> RawConfig:
        return RawConfig(width=width, height=height, bit_depth=self.bit_depth,
                         storage_format="unpacked16", bayer_pattern=self.bayer_pattern,
                         black_level=self.black_level, white_level=(1 << self.bit_depth) - 1)


def mosaic_rgb(rgb: NDArray[np.float32], pattern: BayerPattern) -> NDArray[np.float32]:
    mosaic = np.empty(rgb.shape[:2], dtype=np.float32)
    for index, color in enumerate(pattern):
        row, column = index // 2, index % 2
        mosaic[row::2, column::2] = rgb[row::2, column::2, "RGB".index(color)]
    return mosaic


@dataclass(frozen=True)
class SensorMaps:
    hot: NDArray[np.bool_]
    dead: NDArray[np.bool_]
    prnu: NDArray[np.float32]
    dsnu_e_s: NDArray[np.float32]


class SensorNoiseModel:
    def __init__(self, config: SensorConfig, shape: tuple[int, int], rng: np.random.Generator):
        self.config = config
        hot_count = int(config.hot_pixel_fraction * np.prod(shape))
        dead_count = int(config.dead_pixel_fraction * np.prod(shape))
        defects = rng.choice(int(np.prod(shape)), size=hot_count + dead_count, replace=False)
        hot, dead = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
        hot.flat[defects[:hot_count]] = True
        dead.flat[defects[hot_count:]] = True
        prnu = np.maximum(0, rng.normal(1, config.prnu_fraction, shape)).astype(np.float32)
        dsnu = rng.normal(0, config.dsnu_e_s, shape).astype(np.float32)
        self.maps = SensorMaps(hot, dead, prnu, dsnu)

    def expose(self, radiance: NDArray[np.float32], exposure_s: float, gain: float,
               rng: np.random.Generator) -> NDArray[np.uint16]:
        cfg, maps = self.config, self.maps
        photo_rate = radiance * maps.prnu * cfg.photon_rate_e_s
        mean_e = np.maximum(0, photo_rate + maps.dsnu_e_s + maps.hot * cfg.hot_pixel_rate_e_s) * exposure_s
        mean_e[maps.dead] = 0
        sampled = rng.poisson(mean_e).astype(np.float64) if cfg.shot_noise else mean_e
        electrons = np.minimum(sampled, cfg.full_well_e)
        height, width = radiance.shape
        electrons += rng.normal(0, cfg.read_noise_e, (height, width))
        electrons += rng.normal(0, cfg.row_noise_e, (height, 1))
        electrons += rng.normal(0, cfg.column_noise_e, (1, width))

        white = (1 << cfg.bit_depth) - 1
        codes = cfg.black_level + electrons * gain * (white - cfg.black_level) / cfg.full_well_e
        return np.rint(np.clip(codes, 0, white)).astype(np.uint16)
