from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.camera.capture_plan import CapturePlan
from src.camera.capture_simulator import SimulationConfig, simulate_capture
from src.camera.raw_config import RawConfig
from src.camera.raw_reader import read_raw
from src.camera.sensor_noise import mosaic_rgb
from src.storage.artifact_store import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("overrides", [
    {"exposure_times": []}, {"exposure_times": [0]}, {"gains": [0.5]},
    {"gains": [1, 2]}, {"reference_frame": 1},
])
def test_capture_plan_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        CapturePlan.model_validate({"exposure_times": [0.01], "gains": [1]} | overrides)


def test_capture_plan_drives_raw_codes_exposure_and_gain_independently(tmp_path):
    cfg = SimulationConfig.model_validate({
        "seed": 42, "motion_samples": 3,
        "scene": {"width": 32, "height": 32, "velocity_px_s": 0},
        "sensor": {"full_well_e": 10000, "photon_rate_e_s": 20000, "shot_noise": False,
                   "read_noise_e": 0, "hot_pixel_fraction": 0, "dead_pixel_fraction": 0,
                   "dsnu_e_s": 0, "prnu_fraction": 0, "row_noise_e": 0, "column_noise_e": 0},
        "capture_plan": {"exposure_times": [0.02, 0.04, 0.02], "gains": [1, 1, 2],
                         "reference_frame": 1, "inter_frame_gap_s": 0.005},
    })
    store = ArtifactStore(tmp_path)
    artifact_id = simulate_capture(cfg, store)
    raw = store.load_array(artifact_id, "raw_burst")
    gt = store.load_array(artifact_id, "ground_truth_rgb")
    np.testing.assert_array_equal(gt[0], gt[2])
    np.testing.assert_array_equal(raw[1], raw[2])
    assert (raw[1] > raw[0]).any()
    expected = np.rint(64 + mosaic_rgb(gt[0], "RGGB") * (20000 * 0.02 / 10000) * (4095 - 64))
    np.testing.assert_allclose(raw[0], expected, atol=1)
    layout = RawConfig.model_validate(store.load_metadata(artifact_id)["raw_config"])
    exported = tmp_path / "frame.raw"
    exported.write_bytes(raw[1].astype("<u2").tobytes())
    np.testing.assert_array_equal(read_raw(exported, layout).mosaic, raw[1])
    frames = store.load_metadata(artifact_id)["frames"]
    assert [f["start_s"] for f in frames] == pytest.approx([0, 0.025, 0.07])
    assert [f["reference"] for f in frames] == [False, True, False]


def test_demo_burst_persists_reproducible_ground_truth_noise_and_metadata(tmp_path):
    store = ArtifactStore(tmp_path)
    cfg = SimulationConfig.from_file(ROOT / "configs/synthetic_burst.yaml")
    first, second = [simulate_capture(cfg, store) for _ in range(2)]
    manifest = store.load_metadata(first)
    assert SimulationConfig.model_validate(manifest["simulation_config"]) == cfg
    assert manifest["source_format"] == "simulated_sensor"
    assert manifest["frame_count"] == 4
    for name in manifest["arrays"]:
        np.testing.assert_array_equal(store.load_array(first, name), store.load_array(second, name))
    other = simulate_capture(cfg.model_copy(update={"seed": cfg.seed + 1}), store)
    raw = store.load_array(first, "raw_burst")
    truth = store.load_array(first, "ground_truth_rgb")
    assert not np.array_equal(raw, store.load_array(other, "raw_burst"))
    np.testing.assert_array_equal(truth, store.load_array(other, "ground_truth_rgb"))
    masks = store.load_array(first, "subject_mask")
    assert np.where(masks[3])[1].mean() > np.where(masks[0])[1].mean() + 40
    assert np.abs(truth[3] - store.load_array(first, "exposure_average_rgb")[3]).max() > 0.1
    assert manifest["frames"][-1]["saturated_site_ratio"] == (raw[-1] == 4095).mean()
    assert (store.path(first) / manifest["preview"]).exists()
