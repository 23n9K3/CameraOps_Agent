import json

import cv2
import numpy as np

from src.camera.cli import main
from src.storage.artifact_store import ArtifactStore


def test_cli_processes_sequence_and_publishes_reopenable_artifacts(tmp_path, capsys):
    raw = tmp_path / "测试.raw"
    second = tmp_path / "frame_2.raw"
    config = tmp_path / "layout.yaml"
    artifacts = tmp_path / "结果"
    config.write_text("""width: 2
height: 2
bit_depth: 8
storage_format: unpacked8
bayer_pattern: RGGB
black_level: 15
white_level: 215
""", encoding="utf-8")
    raw.write_bytes(bytes([215, 115, 115, 35]))
    second.write_bytes(bytes([115, 115, 115, 115]))
    assert main([str(raw), str(second), "--config", str(config), "--artifacts", str(artifacts)]) == 0
    first, last = map(json.loads, capsys.readouterr().out.splitlines())
    assert first["artifact_id"] != last["artifact_id"]
    assert first["saturated_site_ratio"] == 0.25
    assert last["saturated_site_ratio"] == 0
    store = ArtifactStore(artifacts)
    artifact_id = first["artifact_id"]
    rgb = store.load_array(artifact_id, "linear_rgb")
    np.testing.assert_allclose(rgb[0, 0], [1, 0.5, 0.1])
    np.testing.assert_array_equal(store.load_array(artifact_id, "raw_mosaic"), [[215, 115], [115, 35]])
    np.testing.assert_array_equal(store.load_array(artifact_id, "saturation_mask"), [[True, False], [False, False]])
    bundle = store.path(artifact_id)
    manifest = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    assert manifest["artifact_id"] == artifact_id
    assert manifest["arrays"]["linear_rgb"]["shape"] == [2, 2, 3]
    png = np.frombuffer((bundle / manifest["preview"]).read_bytes(), dtype=np.uint8)
    assert cv2.imdecode(png, cv2.IMREAD_COLOR).shape == (2, 2, 3)
    assert raw.read_bytes() == bytes([215, 115, 115, 35])


def test_cli_rejects_incompatible_input_without_publishing_bundle(tmp_path, capsys):
    raw = tmp_path / "bad.dng"
    raw.write_bytes(b"invalid")
    artifacts = tmp_path / "artifacts"
    assert main([str(raw), "--artifacts", str(artifacts)]) == 1
    assert capsys.readouterr().out == ""
    assert not artifacts.exists()
