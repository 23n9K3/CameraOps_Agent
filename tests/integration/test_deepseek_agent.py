import io
import json
from pathlib import Path

import pytest

from src.agents import deepseek
from src.camera.capture_simulator import SimulationConfig
from src.evaluation.benchmark import LocalBenchmarkRunner
from src.orchestration.graph import run_workflow
from src.orchestration.state import WorkflowRequest
from src.scene.analysis import SceneFeatures
from src.storage.artifact_store import ArtifactStore
from src.tools.contracts import SourceSpec


def test_deepseek_response_selects_actual_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    calls = []

    def local_http_fixture(request, *, timeout):
        body = json.loads(request.data)
        calls.append(body)
        assert request.full_url == deepseek.ENDPOINT
        assert request.get_header("Authorization") == "Bearer fixture-key"
        assert timeout == 30
        result = {"mode": "hdr", "protect_motion": True, "reason": "Fixture decision"}
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}).encode())

    monkeypatch.setattr(deepseek, "urlopen", local_http_fixture)
    config = SimulationConfig.from_file(Path(__file__).resolve().parents[2] / "configs/synthetic_burst.yaml")
    request = WorkflowRequest(user_request="夜景去噪", source=SourceSpec(simulation=config), brain="deepseek")
    state = run_workflow(request, tmp_path, runner_factory=LocalBenchmarkRunner)
    report = ArtifactStore(tmp_path / "reports").load_metadata(state["report_id"])
    assert len(calls) == 1
    sent = json.loads(calls[0]["messages"][1]["content"])
    assert set(sent) == {"goal", "measured_scene"} and sent["measured_scene"]["brightness_median"] > 0
    assert report["llm_mode"] == "deepseek" and report["brain_decision"]["mode"] == "hdr"
    assert report["capture_plan"]["fusion_strategy"] == "hdr_exposure"
    output = ArtifactStore(tmp_path / "photography").load_metadata(report["final_artifact_id"])
    assert output["pipeline"]["mode"] == "hdr" and output["pipeline"]["protect_motion_reference"]
    assert "fixture-key" not in json.dumps(report)


def test_deepseek_rejects_malformed_output_and_missing_credentials(monkeypatch):
    scene = SceneFeatures(brightness_median=0.1, shadow_ratio=0.2, highlight_ratio=0.1, luminance_p99=0.5,
                          laplacian_variance=0.1, histogram_16=[], motion_fraction=None, moving_flow_p90_px=None,
                          exposures=[], saturation_by_frame=[], frame_sharpness=[])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        deepseek.choose_strategy("noise", scene)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")
    monkeypatch.setattr(deepseek, "urlopen", lambda *args, **kwargs: io.BytesIO(
        b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"mode\\":\\"external_tool\\"}"}}]}'))
    with pytest.raises(RuntimeError, match="invalid planning decision"):
        deepseek.choose_strategy("noise", scene)
