import cv2
import numpy as np
import pytest

from src.computational.fusion import (measurement_weights, multi_frame_average, normalized_average,
                                      repair_isolated_cfa_sites, suppress_ghosting, tone_map)
from src.computational.motion import motion_mask, optical_flow
from src.computational.registration import register_ecc, warp


def test_ecc_recovers_translation_and_warp_direction():
    rng = np.random.default_rng(21)
    reference = cv2.GaussianBlur(rng.random((96, 128), dtype=np.float32), (5, 5), 0)
    shift = np.array([[1, 0, 4], [0, 1, -3]], dtype=np.float32)
    source = cv2.warpAffine(reference, shift, (128, 96))
    mask = cv2.warpAffine(np.ones(reference.shape, dtype=np.uint8), shift, (128, 96)).astype(bool)
    reg = register_ecc(reference, source, mask)
    assert reg.method == "ecc_translation"
    np.testing.assert_allclose(reg.matrix, shift, atol=0.2)
    region = np.s_[8:-8, 8:-8]
    assert np.mean(np.abs(warp(source, reg.matrix)[region] - reference[region])) < 0.01
    assert not (warp(mask.astype(np.float32), reg.matrix) > 0.999).all()


def test_flat_ecc_input_reports_fallback():
    frame = np.ones((32, 32), np.float32)
    reg = register_ecc(frame, frame, frame.astype(bool))
    assert reg.method == "identity_fallback"
    assert reg.note
    np.testing.assert_array_equal(reg.matrix, np.eye(2, 3))


def test_farneback_flow_recovers_local_shift():
    rng = np.random.default_rng(2)
    background = np.full((104, 128), 0.1, dtype=np.float32)
    patch = rng.uniform(0.45, 0.95, size=(40, 40)).astype(np.float32)
    first, second = background.copy(), background.copy()
    first[30:70, 35:75] = patch
    second[32:72, 40:80] = patch
    flow = optical_flow(first, second)
    np.testing.assert_allclose(np.median(flow[40:60, 45:65], axis=(0, 1)), [5, 2], atol=0.5)
    mask = motion_mask(first, second, flow, np.ones(first.shape, bool), difference_threshold=0.1,
                       flow_threshold=1.0, padding=3)
    assert mask[36:64, 35:40].mean() > 0.9
    assert not mask[:10, :10].any()


def test_hdr_recovers_clipped_channel_with_short_exposure():
    radiance = np.array([1.6, 0.35, 0.17], dtype=np.float32).reshape(1, 1, 3)
    dose = np.array([1, 0.25], dtype=np.float32)
    exposed = np.minimum(dose[:, None, None, None] * radiance, 1)
    aligned = exposed / dose[:, None, None, None]
    weights = measurement_weights(exposed, exposed < 1, dose, np.ones_like(dose), hdr=True,
                                  denoise_strength=1.0, reference=0)
    result, normalized = normalized_average(aligned, weights, 0)
    np.testing.assert_allclose(result, radiance, atol=1e-6)
    assert normalized[0, 0, 0, 0] == 0
    np.testing.assert_allclose(normalized.sum(axis=0), 1)


def test_average_and_motion_weights_preserve_reference_where_subject_moves():
    frames = np.zeros((2, 2, 2, 3), dtype=np.float32)
    frames[0, 0, 0] = 0.8
    frames[1, 0, 1] = 0.8
    np.testing.assert_allclose(multi_frame_average(frames)[0, :, 0], [0.4, 0.4])
    masks = np.zeros((2, 2, 2), dtype=bool)
    masks[1, 0] = True
    weights = suppress_ghosting(np.ones_like(frames), masks, 0, motion_aware=True)
    fused, normalized = normalized_average(frames, weights, 0)
    np.testing.assert_array_equal(fused, frames[0])
    np.testing.assert_array_equal(normalized[0, 0], 1)
    np.testing.assert_array_equal(normalized[0, 1], 0.5)


def test_night_repair_uses_same_color_neighbors_and_leaves_sensor_input_unchanged():
    raw = np.full((14, 14), 0.4, np.float32)
    raw[::2, ::2] = 0.05
    raw[6, 6] = 0.95
    corrected, repaired = repair_isolated_cfa_sites(raw)
    assert repaired.sum() == 1
    assert corrected[6, 6] == pytest.approx(0.05)
    assert raw[6, 6] == pytest.approx(0.95)


def test_tone_mapping_preserves_channel_ratios_and_range():
    raw = np.array([[[0, 0, 0], [20, 10, 4]]], dtype=np.float32)
    mapped = tone_map(raw, 2)
    assert (0 <= mapped).all() and (mapped < 1).all()
    assert mapped[0, 1, 0] / mapped[0, 1, 1] == pytest.approx(2)
