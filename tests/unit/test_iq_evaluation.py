import numpy as np
import pytest
from pydantic import ValidationError

from src.iq.evaluation import Metric, evaluate_image
from src.storage.artifact_store import ArtifactStore


def _fixture(tmp_path, *, with_truth):
    bursts, photos = ArtifactStore(tmp_path / "burst"), ArtifactStore(tmp_path / "photography")
    target = np.full((16, 16, 3), 0.25, np.float32)
    estimate = np.full(target.shape, 0.3, np.float32)
    output = estimate * np.float32(0.5)
    mapped = output.copy()
    mapped[0, :2] = 1
    calibration = {}
    if with_truth:
        source = bursts.save({"ground_truth_rgb": target[None]}, {})
        calibration = {"input_artifact_id": source}
    identifier = photos.save({"final_linear": output, "aligned_rgb": output[None],
                             "aligned_geometry": np.ones((1, 16, 16), bool),
                             "fusion_valid_channels": np.ones((1, *output.shape), bool),
                             "fusion_weights": np.ones((1, *output.shape), np.float32), "tone_mapped_rgb": mapped},
                            {"source": calibration, "reference_index": 0, "source_reference_index": 0,
                             "evaluation": {"valid": with_truth, "scene_to_reference_scale": 0.5, "psnr_peak": 1.0,
                                            "reason": "no paired ground truth"},
                             "registration": [{"frame": 0}], "runtime_ms": {"total_before_artifact_write": 17.2}})
    return photos, bursts, identifier


def test_reference_metrics_match_analytic_constant_image_values(tmp_path):
    photos, bursts, identifier = _fixture(tmp_path, with_truth=True)
    metrics = evaluate_image(photos, bursts, identifier).metrics
    assert metrics["psnr"].value == pytest.approx(20 * np.log10(1 / 0.05), abs=1e-4)
    assert metrics["ssim"].value == pytest.approx((2 * 0.25 * 0.3 + 0.01**2) / (0.25**2 + 0.3**2 + 0.01**2), abs=2e-4)
    assert metrics["clipping_ratio"].value == 2 / 256
    assert metrics["noise_std"].value == 0
    assert metrics["runtime"].value == 17.2
    assert metrics["detail_preservation"].value is None
    assert len(metrics["psnr"].input_artifacts) == 2


def test_external_path_only_artifact_still_has_blind_measurements(tmp_path):
    photos, bursts, identifier = _fixture(tmp_path, with_truth=False)
    metrics = evaluate_image(photos, bursts, identifier).metrics
    assert metrics["clipping_ratio"].valid and metrics["runtime"].valid
    for name in ("psnr", "ssim", "alignment_error", "ghosting_score", "mtf50"):
        assert not metrics[name].valid and metrics[name].value is None and metrics[name].limitations
        assert metrics[name].input_artifacts == [identifier]


def test_metric_schema_rejects_contradictory_availability():
    with pytest.raises(ValidationError, match="validity"):
        Metric(value=None, unit="dB", valid=True, method="fixture", input_artifacts=[], limitations="fixture")
