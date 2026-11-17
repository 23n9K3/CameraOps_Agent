import numpy as np
import pytest

from src.camera.sensor_noise import SensorConfig, SensorNoiseModel, mosaic_rgb


def clean_config(**overrides):
    defaults = dict(full_well_e=10000, photon_rate_e_s=20000, shot_noise=False, read_noise_e=0,
                    hot_pixel_fraction=0, dead_pixel_fraction=0, dsnu_e_s=0, prnu_fraction=0,
                    row_noise_e=0, column_noise_e=0)
    return SensorConfig(**(defaults | overrides))


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_rgb_is_sampled_according_to_bayer_pattern(pattern):
    rgb = np.broadcast_to(np.array([0.3, 0.2, 0.1], dtype=np.float32), (2, 2, 3))
    expected = np.array([rgb[0, 0, "RGB".index(site)] for site in pattern]).reshape(2, 2)
    np.testing.assert_array_equal(mosaic_rgb(rgb, pattern), expected)


def test_read_and_shot_noise_match_electron_statistics():
    shape = (256, 256)
    for config, expected_mean, expected_std in [
        (clean_config(shot_noise=True), 200, np.sqrt(200)),
        (clean_config(read_noise_e=8), 200, 8),
    ]:
        sensor = SensorNoiseModel(config, shape, np.random.default_rng(1))
        raw = sensor.expose(np.ones(shape, dtype=np.float32), 0.01, 1, np.random.default_rng(2))
        electrons = (raw.astype(float) - config.black_level) * config.full_well_e / (4095 - config.black_level)
        assert electrons.mean() == pytest.approx(expected_mean, abs=0.5)
        assert electrons.std() == pytest.approx(expected_std, rel=0.04)


def test_hot_and_dead_pixels_are_disjoint_stable_and_change_raw():
    config = clean_config(hot_pixel_fraction=0.1, dead_pixel_fraction=0.1, hot_pixel_rate_e_s=50000)
    sensor = SensorNoiseModel(config, (10, 10), np.random.default_rng(4))
    hot, dead = sensor.maps.hot, sensor.maps.dead
    assert hot.sum() == dead.sum() == 10
    assert not (hot & dead).any()
    for exposure_s in [0.01, 0.03]:
        raw = sensor.expose(np.full((10, 10), 0.4, dtype=np.float32), exposure_s, 1, np.random.default_rng(5))
        assert raw[hot].min() > raw[~(hot | dead)].max()
        assert (raw[dead] == config.black_level).all()


def test_saturation_caps_at_adc_maximum_even_at_high_gain():
    config = clean_config(bit_depth=10)
    sensor = SensorNoiseModel(config, (4, 4), np.random.default_rng(1))
    raw = sensor.expose(np.ones((4, 4), dtype=np.float32), 1, 8, np.random.default_rng(2))
    assert (raw == 1023).all()
