"""Layout and radiometric metadata for Bayer RAW frames."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

BayerPattern = Literal["RGGB", "BGGR", "GRBG", "GBRG"]
SiteLevels = StrictInt | tuple[StrictInt, StrictInt, StrictInt, StrictInt]
PACKED_GROUPS = {10: (4, 5), 12: (2, 3), 14: (4, 7)}


def site_levels(level: SiteLevels) -> tuple[int, int, int, int]:
    return (level, level, level, level) if isinstance(level, int) else level


class RawConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: StrictInt = Field(ge=2)
    height: StrictInt = Field(ge=2)
    bit_depth: Literal[8, 10, 12, 14, 16]
    storage_format: Literal["unpacked8", "unpacked16", "mipi_packed"]
    endianness: Literal["little", "big"] = "little"
    bayer_pattern: BayerPattern
    black_level: SiteLevels
    white_level: SiteLevels
    row_stride_bytes: StrictInt | None = Field(default=None, gt=0)

    @property
    def max_code(self) -> int:
        return (1 << self.bit_depth) - 1

    @property
    def row_payload_bytes(self) -> int:
        if self.storage_format == "mipi_packed":
            pixels, byte_count = PACKED_GROUPS[self.bit_depth]
            return self.width // pixels * byte_count
        return self.width * (2 if self.storage_format == "unpacked16" else 1)

    @property
    def row_stride(self) -> int:
        return self.row_stride_bytes or self.row_payload_bytes

    @model_validator(mode="after")
    def validate_layout(self) -> "RawConfig":
        if self.storage_format == "unpacked8" and self.bit_depth != 8:
            raise ValueError("unpacked8 requires bit_depth=8")
        if self.storage_format == "mipi_packed":
            if self.bit_depth not in PACKED_GROUPS:
                raise ValueError("mipi_packed supports RAW10, RAW12 and RAW14")
            pixels, _ = PACKED_GROUPS[self.bit_depth]
            if self.width % pixels:
                raise ValueError(f"MIPI RAW{self.bit_depth} requires a width divisible by {pixels}")
        if self.storage_format != "unpacked16" and self.endianness == "big":
            raise ValueError("endianness only applies to unpacked16; MIPI byte order is fixed")
        if self.row_stride < self.row_payload_bytes:
            raise ValueError(f"row stride must be at least {self.row_payload_bytes} bytes")
        for black, white in zip(site_levels(self.black_level), site_levels(self.white_level)):
            if not 0 <= black < white <= self.max_code:
                raise ValueError(f"each site requires 0 <= black < white <= {self.max_code}")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "RawConfig":
        with Path(path).open(encoding="utf-8-sig") as source:
            return cls.model_validate(yaml.safe_load(source))
