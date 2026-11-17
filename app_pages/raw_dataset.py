"""Inspect single files and save uploaded RAW sequences for downstream pages."""

import cv2
import streamlit as st
import yaml

from src.computational.sequence import Exposure, RawSequence, publish_raw_sequence
from src.storage.artifact_store import ArtifactStore
from src.visualization.catalog import PROJECT, artifact_root, burst_label, catalog, choose_artifact
from src.visualization.dataset import burst_viewer
from src.visualization.viewer import raw_viewer, read_upload

st.title("RAW 数据集")
store = ArtifactStore(artifact_root() / "burst")
layout_path = PROJECT / "configs/raw12_example.yaml"
view = st.segmented_control("数据集操作", ["Burst 序列", "RAW 文件", "导入序列"], default="Burst 序列")

if view == "RAW 文件":
    raw_viewer(layout_path)
elif view == "导入序列":
    kind = st.segmented_control("序列编码", ["相机 RAW 容器", "无头 Bayer RAW"], default="相机 RAW 容器")
    with st.form("upload_burst", border=True):
        uploads = st.file_uploader("按采集顺序上传 RAW 帧", accept_multiple_files=True, max_upload_size=64)
        row = st.columns(2)
        exposure = row[0].text_input("逐帧曝光 (ms，逗号分隔)", placeholder="4, 10, 25")
        gains = row[1].text_input("逐帧模拟增益 (×，逗号分隔)", placeholder="4, 2, 1")
        layout = st.text_area("公共 RAW 布局", layout_path.read_text(encoding="utf-8"), height=165) if kind == "无头 Bayer RAW" else None
        save = st.form_submit_button("导入并保存序列", type="primary", icon=":material/upload:")
    if save:
        try:
            if not 1 <= len(uploads) <= 16:
                raise ValueError("请选择 1 至 16 帧 RAW。")
            times = [float(part.strip()) / 1000 for part in exposure.split(",")]
            analog = [float(part.strip()) for part in gains.split(",")]
            if len(uploads) != len(times) or len(uploads) != len(analog):
                raise ValueError("每帧必须有对应曝光与增益。")
            frames = [read_upload(file.getvalue(), layout) for file in uploads]
            sequence = RawSequence(tuple(frame.mosaic for frame in frames), tuple(frame.config for frame in frames),
                                   tuple(Exposure(exposure_s=t, analog_gain=g) for t, g in zip(times, analog)),
                                   {"uploaded_files": [file.name for file in uploads], "formats": [frame.source_format for frame in frames]})
            st.session_state["dataset_imported"] = publish_raw_sequence(store, sequence)
        except (ValueError, OSError, yaml.YAMLError, cv2.error) as exc:
            st.error(str(exc))
    if identifier := st.session_state.get("dataset_imported"):
        st.success(f"已保存序列 `{identifier}`。可在计算摄影或 Agent 工作台选择。")
        burst_viewer(store, identifier)
else:
    if identifier := choose_artifact("选择 RAW Burst", catalog("burst"), key="dataset_burst", names=burst_label):
        burst_viewer(store, identifier)
    else:
        st.page_link("app_pages/capture_planner.py", label="前往采集规划", icon=":material/photo_camera:")
