from pathlib import Path
from time import monotonic, sleep

from streamlit.testing.v1 import AppTest

from src.camera.capture_simulator import SimulationConfig, simulate_capture
from src.computational.sequence import publish_raw_sequence, read_burst
from src.memory.experiment_memory import ExperimentMemory
from src.storage.artifact_store import ArtifactStore
from src.visualization.inspection import crop_box
from src.visualization.report_exports import export_html

ROOT = Path(__file__).resolve().parents[2]


def _app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()


def _page(at, name):
    at.switch_page(f"app_pages/{name}.py").run()
    assert not at.exception


def _wait_for(predicate, seconds=35):
    deadline = monotonic() + seconds
    while not predicate() and monotonic() < deadline:
        sleep(0.05)
    assert predicate()


def test_live_workspace_reveals_active_tool_and_repair_then_exports_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    store = ArtifactStore(tmp_path / "artifacts" / "burst")
    config = SimulationConfig.from_file(ROOT / "configs/synthetic_burst.yaml")
    synthetic = simulate_capture(config, store)
    at = _app()
    _page(at, "agent_workspace")
    assert at.button[0].disabled  # Synthetic fixtures are never selected by the workbench.
    imported = publish_raw_sequence(store, read_burst(store, synthetic))  # Exercise imported input with a test-only fixture.
    at.run()
    assert at.selectbox[0].value == imported
    at.text_area[0].set_value("处理运动行人并减少鬼影")
    at.checkbox[0].set_value(True)
    at.button[0].click().run()
    assert not at.exception
    job = at.session_state["workflow_job"]
    _wait_for(lambda: any(event["type"] == "tool_started" for event in job.snapshot()["events"]))
    assert not job.done
    at.run()
    assert not at.exception
    assert at.button[0].disabled
    _wait_for(lambda: job.done)
    at.run()
    assert not at.exception and not at.error
    state = job.snapshot()["result"]
    assert state["input_artifacts"] == [imported]
    assert state["status"] == "completed" and state["iteration_count"] == 1
    assert not at.button[0].disabled
    assert [m.value for m in at.metric[:2]] == ["完成", "1"]
    events = job.snapshot()["events"]
    assert len([e for e in events if e["type"] == "tool_started"]) == len([e for e in events if e["type"] == "tool_finished"])
    assert any(e.get("tool") == "validate_diagnosis" and e.get("result", {}).get("validation_result") == "confirmed"
               for e in events if e["type"] == "tool_finished" and e.get("result"))
    assert at.get("download_button")
    at.toggle[0].set_value(True).run()
    next(widget for widget in at.selectbox if widget.label == "图像对比").select("运动掩码").run()
    assert not at.exception
    assert any("运动区覆盖率" in text.value for text in at.caption)

    _page(at, "debugging")
    assert at.get("plotly_chart")
    next(widget for widget in at.selectbox if widget.label == "故障假设").select("ghosting")
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert any("confirmed" in str(item.value) for item in at.json)
    _page(at, "reports")
    assert len(at.get("download_button")) == 4
    _page(at, "dashboard")
    assert [m.value for m in at.metric[:2]] == ["1", "100%"]


def test_dataset_import_photography_roi_and_optuna_actions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = _app()
    _page(at, "raw_dataset")
    at.segmented_control[0].set_value("导入序列").run()
    at.segmented_control[1].set_value("无头 Bayer RAW").run()
    at.text_area[0].set_value("""width: 2
height: 2
bit_depth: 8
storage_format: unpacked8
bayer_pattern: RGGB
black_level: 0
white_level: 255
""")
    at.text_input[0].set_value("4, 10")
    at.text_input[1].set_value("1, 1")
    at.file_uploader[0].upload("01.raw", bytes([70, 40, 40, 30])).upload("02.raw", bytes([175, 100, 100, 75]))
    at.button[0].click().run()
    assert not at.exception and not at.error and at.success
    imported = at.session_state["dataset_imported"]
    assert ArtifactStore(tmp_path / "artifacts/burst").load_metadata(imported)["frames"][1]["exposure_s"] == 0.010

    _page(at, "photography")
    at.button[0].click().run()
    assert not at.exception and not at.error
    output = at.session_state["photo_output"]
    assert ExperimentMemory(tmp_path / "artifacts/experiments.sqlite3").history(artifact_id=output)
    at.select_slider[0].set_value(2).run()
    assert not at.exception and at.get("plotly_chart")
    next(widget for widget in at.selectbox if widget.label == "图像对比").select("融合权重").run()
    assert not at.exception and at.get("download_button")

    _page(at, "capture_planner")
    at.text_input[0].set_value("HDR 高光")
    at.button[0].click().run()
    at.button[1].click().run()
    assert not at.exception and at.success

    _page(at, "experiments")
    assert at.dataframe
    at.segmented_control[0].set_value("Optuna 与 Pareto").run()
    at.number_input[0].set_value(1)
    at.button[0].click().run()
    job = at.session_state["optimization_job"]
    _wait_for(lambda: job.done)
    at.run()
    assert not at.exception and not at.error
    assert at.metric[0].value == "2/2"
    assert at.get("plotly_chart") and len(at.get("download_button")) == 2
    _page(at, "reports")
    at.segmented_control[0].set_value("多目标优化").run()
    assert not at.exception and len(at.get("download_button")) == 4


def test_comparison_crop_uses_reference_coordinates_and_html_escapes_goal():
    rows, cols = crop_box((192, 320), 4, 100, 0)
    assert rows == slice(0, 48) and cols == slice(240, 320)
    html = export_html({"user_request": "<script>test</script>"}, []).decode("utf-8")
    assert "&lt;script&gt;test&lt;/script&gt;" in html


def test_artifact_publish_retries_transient_windows_sharing_conflict(tmp_path, monkeypatch):
    original = Path.rename
    renamed = []

    def rename(path, target):
        if path.name == "bundle" and not renamed:
            renamed.append(target)
            error = PermissionError("sharing conflict")
            error.winerror = 32
            raise error
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    store = ArtifactStore(tmp_path)
    identifier = store.save({}, {"artifact_type": "test"})
    assert renamed == [store.path(identifier)]
    assert store.load_metadata(identifier)["artifact_type"] == "test"
    assert [path.name for path in tmp_path.iterdir()] == [identifier]
