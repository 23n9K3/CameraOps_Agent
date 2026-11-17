"""Frozen benchmark inputs, targets and known injected failure labels."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.camera.capture_plan import CapturePlan
from src.camera.capture_simulator import SimulationConfig, simulate_capture
from src.storage.artifact_store import ArtifactStore
from src.visualization.preview import display_rgb, encode_png


FAMILIES = {
    "night": ("night", "夜景降噪，保留暗部细节"),
    "hdr": ("hdr", "HDR 恢复高光和暗部"),
    "motion": ("motion", "减少运动行人的拖影"),
    "noise": ("night", "低照度高噪声，请降噪"),
    "ghosting": ("motion", "减少运动行人鬼影"),
    "registration": ("motion", "减少手抖产生的运动拖影"),
    "overexposure": ("hdr", "修复曝光过度的高光"),
    "underexposure": ("night", "改善欠曝光夜景的暗部细节"),
    "bayer": ("night", "夜间 Bayer RAW 偏色，请恢复正确彩色"),
    "control": ("night", "处理夜景 RAW，保留现有细节"),
}


@dataclass(frozen=True)
class Case:
    key: str
    family: str
    goal: str
    expected_mode: str
    seed: int
    input_id: str
    reference: int
    initial: dict
    fault: str | None = None
    shift: tuple[int, int] | None = None


def simulation_for(family: str, variant: int, seed: int) -> SimulationConfig:
    scene = {"width": 160, "height": 96, "velocity_px_s": 0}
    sensor = {"bit_depth": 12, "bayer_pattern": ("RGGB", "BGGR", "GRBG")[variant % 3]}
    plan = {"inter_frame_gap_s": 0.015, "reference_frame": 0, "capture_mode": "burst"}
    if family in ("hdr", "overexposure"):
        ref = (0.039 if family == "hdr" else 0.053) + 0.003 * variant
        plan.update(capture_mode="bracket", exposure_times=[0.005, ref, 0.075, 0.018], gains=[1] * 4, reference_frame=1)
    elif family in ("motion", "ghosting"):
        scene["velocity_px_s"] = 310 + 36 * variant
        plan.update(capture_mode="motion_aware", exposure_times=[0.004, 0.010, 0.022, 0.038], gains=[4, 2, 1, 1])
    else:
        time = 0.008 + 0.001 * variant
        if family in ("night", "noise", "underexposure"):
            sensor.update(photon_rate_e_s=65000 + 4500 * variant, read_noise_e=8 if family == "noise" else 4,
                          hot_pixel_fraction=0.0007)
        times = [0.002, 0.025, 0.032, 0.018] if family == "underexposure" else [time] * 4
        plan.update(exposure_times=times, gains=[4] * 4 if family != "control" else [2] * 4)
    return SimulationConfig.model_validate({"seed": seed, "motion_samples": 10, "scene": scene, "sensor": sensor,
                                            "capture_plan": plan})


def _registration_input(store: ArtifactStore, variant: int, seed: int) -> tuple[str, tuple[int, int]]:
    h, w = 112, 144
    settings = SimulationConfig.model_validate({"seed": seed, "scene": {"width": w, "height": h, "velocity_px_s": 0},
        "sensor": {"shot_noise": False, "read_noise_e": 0, "hot_pixel_fraction": 0, "dead_pixel_fraction": 0,
                   "dsnu_e_s": 0, "prnu_fraction": 0, "row_noise_e": 0, "column_noise_e": 0},
        "capture_plan": {"exposure_times": [0.012, 0.012], "gains": [1, 1]}})
    rng = np.random.default_rng(seed)
    texture = cv2.GaussianBlur(rng.uniform(0.025, 0.92, (h, w)).astype(np.float32), (5, 5), 0.8)
    shift = ((12, -8), (-10, 8), (14, 6))[variant % 3]
    matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
    shifted = cv2.warpAffine(texture, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    raw = np.rint(64 + np.stack((texture, shifted)) * (4095 - 64)).astype(np.uint16)
    exposure = settings.capture_plan.exposure_times[0]
    factor = settings.sensor.photon_rate_e_s * exposure / settings.sensor.full_well_e
    reference = np.repeat(texture[..., None] / np.float32(factor), 3, axis=-1)
    truth = np.stack((reference, reference))
    frames = [{"index": index, "exposure_s": exposure, "analog_gain": 1.0, "start_s": start,
               "midpoint_s": start + exposure / 2, "end_s": start + exposure, "mean_raw_dn": float(raw[index].mean()),
               "saturated_site_ratio": float((raw[index] >= 4095).mean()), "subject_displacement_during_exposure_px": 0.0,
               "reference": index == 0} for index, start in enumerate((0.0, exposure + settings.capture_plan.inter_frame_gap_s))]
    preview = np.repeat(np.concatenate((texture, shifted), axis=1)[..., None], 3, axis=-1)
    identifier = store.save({"raw_burst": raw, "ground_truth_rgb": truth, "exposure_average_rgb": truth,
                             "subject_mask": np.zeros(raw.shape, dtype=bool), "hot_pixel_mask": np.zeros((h, w), bool),
                             "dead_pixel_mask": np.zeros((h, w), bool), "prnu_map": np.ones((h, w), np.float32),
                             "dsnu_e_s_map": np.zeros((h, w), np.float32)}, {
        "artifact_type": "synthetic_raw_burst", "source_format": "simulated_sensor", "frames": frames,
        "hot_pixel_count": 0, "dead_pixel_count": 0, "frame_count": len(frames),
        "raw_config": settings.sensor.raw_layout(w, h).model_dump(mode="json"),
        "simulation_config": settings.model_dump(mode="json"), "fixture": "seeded calibrated texture with known CFA-preserving shift",
    }, preview_png=encode_png(display_rgb(preview)))
    return identifier, shift


def make_case(store: ArtifactStore, family: str, variant: int, seed: int, *, training: bool = False) -> Case:
    expected, goal = FAMILIES[family]
    initial, fault, shift = {}, None, None
    if family == "registration":
        identifier, shift = _registration_input(store, variant, seed)
        reference, initial, fault = 0, {"registration": False}, "registration_failure"
    else:
        config = simulation_for(family, variant, seed)
        reference = config.capture_plan.reference_frame
        identifier = simulate_capture(config, store)
        if family == "ghosting":
            initial, fault = {"deghost": False}, "ghosting"
            if training:
                initial["motion_threshold"] = 0.023
        if family == "bayer":
            meta = store.load_metadata(identifier)
            arrays = {name: store.load_array(identifier, name, mmap_mode="r") for name in meta.pop("arrays")}
            meta.pop("artifact_id")
            meta.pop("preview")
            meta["raw_config"]["bayer_pattern"] = {"RGGB": "BGGR", "BGGR": "RGGB", "GRBG": "GBRG"}[config.sensor.bayer_pattern]
            meta["fixture"] = {"type": "deliberate Bayer metadata mismatch", "original_burst": identifier}
            identifier = store.save(arrays, meta)
            fault = "bayer_mismatch"
    return Case(f"{family}-{variant + 1:02d}", family, goal, expected, seed, identifier, reference, initial, fault, shift)
