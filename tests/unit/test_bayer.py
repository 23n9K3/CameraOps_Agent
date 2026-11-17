import numpy as np
import pytest

from src.camera.bayer import demosaic, normalize_mosaic
from src.camera.raw_config import RawConfig


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_demosaic_recovers_correct_channel_order_at_odd_size_and_borders(pattern):
    values = {"R": 0.8, "G": 0.4, "B": 0.1}
    mosaic = np.empty((5, 7), dtype=np.float32)
    for index, color in enumerate(pattern):
        mosaic[index // 2::2, index % 2::2] = values[color]
    expected = np.broadcast_to(list(values.values()), (5, 7, 3))
    np.testing.assert_allclose(demosaic(mosaic, pattern), expected, atol=1e-7)


def test_measured_green_sites_are_preserved():
    mosaic = np.array([[0, 0.25], [0.75, 0]], dtype=np.float32)
    rgb = demosaic(mosaic, "RGGB")
    np.testing.assert_array_equal(rgb[..., 1], [[0.5, 0.25], [0.75, 0.5]])


def test_normalization_uses_per_site_levels_and_preserves_original():
    config = RawConfig(width=2, height=2, bit_depth=8, storage_format="unpacked8",
                       bayer_pattern="BGGR", black_level=[16, 30, 20, 40],
                       white_level=[216, 230, 220, 240])
    mosaic = np.array([[0, 130], [220, 250]], dtype=np.uint16)
    original = mosaic.copy()
    linear, saturated = normalize_mosaic(mosaic, config)
    assert linear.dtype == np.float32
    np.testing.assert_array_equal(linear, [[0, 0.5], [1, 1]])
    np.testing.assert_array_equal(saturated, [[False, False], [True, True]])
    np.testing.assert_array_equal(mosaic, original)
