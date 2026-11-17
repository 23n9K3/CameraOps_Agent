"""Decode headerless Bayer buffers and LibRaw-supported Bayer containers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy
from numpy.typing import NDArray

from .raw_config import PACKED_GROUPS, RawConfig


class RawFormatError(ValueError):
    """Input does not match a supported RAW layout."""


@dataclass(frozen=True)
class RawFrame:
    mosaic: NDArray[np.uint16]
    config: RawConfig
    source_format: str


def _unpack_mipi(rows: NDArray[np.uint8], config: RawConfig) -> NDArray[np.uint16]:
    pixels, byte_count = PACKED_GROUPS[config.bit_depth]
    groups = rows.reshape(config.height, -1, byte_count).astype(np.uint32)
    low_bits_per_pixel = config.bit_depth - 8
    # Lower bits form a little-endian field after the high bytes in each group.
    low_field = groups[..., pixels].copy()
    for index in range(1, byte_count - pixels):
        low_field |= groups[..., pixels + index] << (8 * index)
    offsets = np.arange(pixels, dtype=np.uint32) * low_bits_per_pixel
    low_mask = (1 << low_bits_per_pixel) - 1
    samples = (groups[..., :pixels] << low_bits_per_pixel) | (
        (low_field[..., None] >> offsets) & low_mask
    )
    return samples.reshape(config.height, config.width).astype(np.uint16)


def decode_raw(payload: bytes, config: RawConfig) -> NDArray[np.uint16]:
    """Return native-endian, right-aligned sensor codes with padding removed."""
    expected_bytes = config.height * config.row_stride
    if len(payload) != expected_bytes:
        raise RawFormatError(f"expected {expected_bytes} bytes, got {len(payload)}")
    rows = np.frombuffer(payload, dtype=np.uint8).reshape(config.height, config.row_stride)
    rows = rows[:, :config.row_payload_bytes]
    if config.storage_format == "mipi_packed":
        return _unpack_mipi(rows, config)
    if config.storage_format == "unpacked8":
        return rows.astype(np.uint16)

    byte_order = "<u2" if config.endianness == "little" else ">u2"
    samples = rows.view(byte_order).astype(np.uint16)
    if samples.max() > config.max_code:
        raise RawFormatError(f"sample exceeds {config.max_code}; expected right-aligned RAW codes")
    return samples


def _read_camera_container(path: Path) -> RawFrame:
    try:
        with path.open("rb") as source, rawpy.imread(source) as camera:
            if (camera.raw_type != rawpy.RawType.Flat or camera.raw_pattern is None
                    or camera.raw_pattern.shape != (2, 2)):
                raise RawFormatError("only 2x2 Bayer camera RAW containers are supported")
            mosaic = camera.raw_image_visible.copy()
            # The visible crop can shift the phase relative to the full sensor tile.
            sizes = camera.sizes
            indices = [
                int(camera.raw_color(sizes.top_margin + row, sizes.left_margin + column))
                for row in range(2) for column in range(2)
            ]
            pattern = "".join(chr(camera.color_desc[index]) for index in indices)
            if pattern not in ("RGGB", "BGGR", "GRBG", "GBRG"):
                raise RawFormatError(f"unsupported visible CFA {pattern!r}")
            blacks = camera.black_level_per_channel
            whites = camera.camera_white_level_per_channel
            white_fallback = int(camera.white_level)
            config = RawConfig(
                width=mosaic.shape[1], height=mosaic.shape[0], bit_depth=16,
                storage_format="unpacked16", bayer_pattern=pattern,
                black_level=tuple(int(blacks[index]) for index in indices),
                white_level=tuple(int(whites[index] or white_fallback) if whites else white_fallback
                                  for index in indices),
            )
    except rawpy.LibRawError as exc:
        raise RawFormatError(f"LibRaw could not decode {path.name}: {exc}") from exc
    except NotImplementedError as exc:
        raise RawFormatError(f"unsupported LibRaw layout in {path.name}: {exc}") from exc
    return RawFrame(mosaic, config, "libraw")


def read_raw(path: str | Path, config: RawConfig | None = None) -> RawFrame:
    """Use an explicit layout for a headerless file, otherwise use LibRaw."""
    path = Path(path)
    if config is None:
        return _read_camera_container(path)
    with path.open("rb") as source:
        # The extra byte lets us detect trailing bytes without reading an unbounded file.
        payload = source.read(config.height * config.row_stride + 1)
    return RawFrame(decode_raw(payload, config), config, "headerless")
