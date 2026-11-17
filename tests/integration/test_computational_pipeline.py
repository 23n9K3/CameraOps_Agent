import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

from scripts.run_photography_demo import demo_config
from src.camera.capture_simulator import SimulationConfig, simulate_capture
from src.computational.pipeline import PipelineConfig, run_pipeline
from src.computational.sequence import read_burst
from src.storage.artifact_store import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["night", "hdr", "motion"])
def test_each_pipeline_processes_raw_and_improves_its_target_against_ground_truth(mode, tmp_path):
    base = SimulationConfig.from_file(ROOT / "configs/synthetic_burst.yaml")
    raw = ArtifactStore(tmp_path / "burst")
    results = ArtifactStore(tmp_path / "processed")
    input_id = simulate_capture(demo_config(base, mode), raw)
    batch = read_burst(raw, input_id)
    output_id = run_pipeline(batch, PipelineConfig(mode=mode), results, truth_store=raw)
    manifest = results.load_metadata(output_id)
    scores = manifest["evaluation"]["scores"]
    assert manifest["evaluation"]["valid"]
    assert manifest["source"]["input_artifact_id"] == input_id
    assert all(runtime > 0 for runtime in manifest["runtime_ms"].values())
    assert "ecc_translation" in {record["method"] for record in manifest["registration"]}
    weights = results.load_array(output_id, "fusion_weights")
    np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-6)
    assert weights.shape == (len(batch.mosaics), 192, 320, 3)
    assert results.load_array(output_id, "optical_flow").shape == (*weights.shape[:3], 2)
    assert results.load_array(output_id, "transforms").shape == (len(weights), 2, 3)
    assert results.load_array(output_id, "motion_masks").dtype == bool
    np.testing.assert_allclose(
        (weights * results.load_array(output_id, "aligned_rgb")).sum(axis=0),
        results.load_array(output_id, "fused_linear"), atol=1e-6,
    )
    png = (results.path(output_id) / manifest["preview"]).read_bytes()
    assert cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR).shape == (192, 320, 3)
    assert (results.path(output_id) / manifest["previews"]["diagnostics"]).is_file()

    final = scores["final_fusion"]
    if mode == "night":
        assert final["static_shadows"]["rmse"] < 0.85 * scores["single_preprocessed"]["static_shadows"]["rmse"]
        assert all(count > 0 for count in manifest["isolated_sites_repaired"])
    elif mode == "hdr":
        highlights = "reference_clipped_highlights"
        assert final[highlights]["rmse"] < 0.83 * scores["single_reference"][highlights]["rmse"]
    else:
        assert final["motion_region"]["rmse"] < 0.5 * scores["fusion_without_deghosting"]["motion_region"]["rmse"]
        assert results.load_array(output_id, "motion_masks")[1:].any()


def test_cli_reads_external_raw_sequence_and_reports_absent_ground_truth(tmp_path):
    config = SimulationConfig.model_validate({
        "seed": 78, "scene": {"width": 48, "height": 32}, "sensor": {"bayer_pattern": "BGGR"},
        "capture_plan": {"exposure_times": [0.009, 0.003], "gains": [2, 4]},
    })
    store = ArtifactStore(tmp_path / "burst")
    source_id = simulate_capture(config, store)
    raw = store.load_array(source_id, "raw_burst")
    manifest = {"raw_config": store.load_metadata(source_id)["raw_config"], "frames": []}
    for index, raw_frame in enumerate(raw):
        filename = f"frame_{index}.raw"
        (tmp_path / filename).write_bytes(raw_frame.astype("<u2").tobytes())
        manifest["frames"].append({"path": filename, "exposure_s": config.capture_plan.exposure_times[index],
                                   "analog_gain": config.capture_plan.gains[index]})
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_photography.py"), "--mode", "motion",
                             "--raw-manifest", str(path), "--output", str(tmp_path / "results")],
                            capture_output=True, text=True, check=True, cwd=ROOT)
    summary = json.loads(result.stdout)
    assert not summary["evaluation"]["valid"]
    assert summary["reference_index"] == 1
    assert summary["arrays"]["tone_mapped_rgb"]["shape"] == [32, 48, 3]
