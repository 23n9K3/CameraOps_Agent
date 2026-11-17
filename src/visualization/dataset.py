"""Render RAW Burst artifacts with recorded layouts and exposures."""

import streamlit as st

from src.camera.raw_config import RawConfig
from src.storage.artifact_store import ArtifactStore

from .viewer import _show_frame


def burst_viewer(store: ArtifactStore, artifact_id: str) -> None:
    metadata = store.load_metadata(artifact_id)
    simulated = metadata["artifact_type"] == "synthetic_raw_burst"
    frames = metadata["frames"]
    reference = metadata["simulation_config"]["capture_plan"]["reference_frame"] if simulated else (metadata["reference_frame"] or 0)
    if simulated:
        preview = store.path(artifact_id) / "preview.png"
        if preview.is_file():
            st.image(preview.read_bytes(), width="stretch", alt="各曝光与增益对应的 Burst 缩略图")
    index = st.selectbox("查看帧", range(len(frames)), index=reference, key=f"frame_{artifact_id}",
                         format_func=lambda i: f'第 {i + 1} 帧 · {frames[i]["exposure_s"] * 1000:g} ms' +
                         (" · 参考帧" if i == reference else ""))
    config = RawConfig.model_validate(metadata["raw_config"] if simulated else metadata["raw_layouts"][index])
    with st.container(horizontal=True):
        st.metric("传感器布局", f"RAW{config.bit_depth} · {config.bayer_pattern}", border=True)
        st.metric("帧数", len(frames), border=True)
        if simulated:
            st.metric("热／死像素", f'{metadata["hot_pixel_count"]} / {metadata["dead_pixel_count"]}', border=True)
    raw = store.load_array(artifact_id, "raw_burst", mmap_mode="r")[index]
    views = {label: store.load_array(artifact_id, key, mmap_mode="r")[index] for label, key in (
        ("Ground Truth · 曝光中点", "ground_truth_rgb"), ("Ground Truth · 曝光平均", "exposure_average_rgb"),
        ("主体掩码", "subject_mask"))} if simulated else None
    _show_frame(raw, config, views)
    st.subheader("采集元数据")
    st.dataframe([{"帧": i + 1, "曝光 (ms)": frame["exposure_s"] * 1000, "模拟增益 (×)": frame["analog_gain"],
                   "开始 (ms)": round(frame["start_s"] * 1000, 3) if simulated else None,
                   "平均 RAW (DN)": round(frame["mean_raw_dn"], 2) if simulated else None,
                   "饱和 (%)": round(frame["saturated_site_ratio"] * 100, 4) if simulated else None}
                  for i, frame in enumerate(frames)], hide_index=True, alt="逐帧曝光、增益和饱和度")
    export_layout = RawConfig.model_validate(config.model_dump() | {
        "storage_format": "unpacked16", "endianness": "little", "row_stride_bytes": None})
    with st.container(horizontal=True):
        st.download_button("下载当前帧 RAW", raw.astype("<u2", copy=False).tobytes(), f"frame_{index + 1:02d}.raw", on_click="ignore")
        st.download_button("下载 RAW 布局", export_layout.model_dump_json(indent=2), "raw_layout.json",
                           "application/json", on_click="ignore")
        st.download_button("下载 Burst 元数据", (store.path(artifact_id) / "metadata.json").read_bytes(),
                           "metadata.json", "application/json", on_click="ignore")
