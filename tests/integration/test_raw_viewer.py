from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def test_burst_viewer_generates_and_switches_to_ground_truth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    at.switch_page("app_pages/capture_planner.py").run()
    assert not at.exception
    at.button[0].click().run()
    at.button[1].click().run()
    assert at.success
    at.switch_page("app_pages/raw_dataset.py").run()
    assert not at.exception
    assert not at.error
    assert at.metric[1].value == "6"
    assert len(at.get("plotly_chart")) == 1
    frame = next(widget for widget in at.selectbox if widget.label == "查看帧")
    frame.select(3).run()
    at.selectbox[-1].select("Ground Truth · 曝光平均").run()
    assert not at.exception
    assert at.selectbox[-1].value == "Ground Truth · 曝光平均"
    assert len(at.get("download_button")) == 3


def test_raw_viewer_uploads_and_displays_headerless_bayer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    at.switch_page("app_pages/raw_dataset.py").run()
    at.segmented_control[0].set_value("RAW 文件").run()
    at.segmented_control[1].set_value("无头 Bayer RAW").run()
    at.text_area[0].set_value("""width: 2
height: 2
bit_depth: 8
storage_format: unpacked8
bayer_pattern: GBRG
black_level: 0
white_level: 255
""")
    at.file_uploader[0].upload("frame.raw", bytes([64, 32, 255, 64])).run()
    assert not at.exception
    assert not at.error
    assert any("GB / RG" in caption.value for caption in at.caption)
    assert len(at.get("plotly_chart")) == 1
