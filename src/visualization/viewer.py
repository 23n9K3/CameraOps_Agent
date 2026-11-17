"""Streamlit inspection views for sensor codes, exposure and scene Ground Truth."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.camera.bayer import demosaic, normalize_mosaic
from src.camera.raw_config import RawConfig, site_levels
from src.camera.raw_reader import RawFrame, decode_raw, read_raw

from .preview import display_rgb
from .inspection import compare_views


def _exposure_histogram(raw: np.ndarray, config: RawConfig) -> None:
    white = site_levels(config.white_level)
    upper = max(int(raw.max()), *white)
    width = (upper + 128) // 128
    counts = np.bincount(raw.ravel() // width)
    centers = np.arange(len(counts)) * width + (width - 1) / 2
    figure = go.Figure(go.Bar(x=centers, y=counts, width=width, marker_color="#368477",
                             hovertemplate="码值中心 %{x}<br>采样点 %{y:,}<extra></extra>"))
    for levels, color, label in ((site_levels(config.black_level), "#5B678B", "黑电平"),
                                 (white, "#B76A3E", "白电平")):
        for level in sorted(set(levels)):
            figure.add_vline(x=level, line_dash="dot", line_color=color)
        figure.add_annotation(x=max(levels), y=1, yref="paper", yshift=13, text=label, showarrow=False,
                              xanchor="right" if label == "白电平" else "left", font_color=color)
    figure.update_layout(height=290, margin=dict(l=5, r=9, t=25, b=0), bargap=0,
                         xaxis=dict(title="RAW 码值（DN）", range=[0, upper + width]), yaxis_title="采样点数")
    st.plotly_chart(figure, alt="RAW 码值曝光直方图，标示黑电平和白电平",
                    config={"displayModeBar": False})


def _show_frame(raw: np.ndarray, config: RawConfig, views: dict[str, np.ndarray] | None = None) -> None:
    linear, saturated = normalize_mosaic(raw, config)
    choices = ["RAW 去马赛克", "Bayer 灰度", "饱和掩码", *(views or {})]
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        mode = st.selectbox("图像视图", choices)
        if mode == "RAW 去马赛克":
            preview = display_rgb(demosaic(linear, config.bayer_pattern))
        elif mode == "Bayer 灰度":
            preview = np.rint(linear * 255).astype(np.uint8)
        elif mode == "饱和掩码":
            preview = saturated.astype(np.uint8) * 255
        else:
            values = views[mode]
            preview = values.astype(np.uint8) * 255 if values.dtype == bool else display_rgb(values)
        compare_views([(mode, preview)], key="raw_zoom", histogram=False)
        pattern = config.bayer_pattern
        st.caption(f"{config.width} × {config.height}  ·  {pattern[:2]} / {pattern[2:]}  ·  "
                   f"饱和点 {saturated.mean():.3%}")
    with right:
        st.subheader("曝光直方图")
        _exposure_histogram(raw, config)


@st.cache_data(max_entries=2, show_spinner=False)
def read_upload(payload: bytes, layout: str | None) -> RawFrame:
    if layout is not None:
        config = RawConfig.model_validate(yaml.safe_load(layout))
        return RawFrame(decode_raw(payload, config), config, "headerless")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "uploaded_camera.raw"
        path.write_bytes(payload)
        return read_raw(path)


def raw_viewer(layout_path: Path) -> None:
    camera_format = st.segmented_control("输入格式", ["相机 RAW 容器", "无头 Bayer RAW"], default="相机 RAW 容器")
    layout = None
    if camera_format == "无头 Bayer RAW":
        layout = st.text_area("RAW 布局（YAML 或 JSON）", layout_path.read_text(encoding="utf-8"), height=175)
    upload = st.file_uploader("上传 RAW 文件", max_upload_size=64)
    if upload is None:
        return
    try:
        frame = read_upload(upload.getvalue(), layout)
        _show_frame(frame.mosaic, frame.config)
        with st.expander("解码元数据"):
            st.json(frame.config.model_dump(mode="json"))
    except (OSError, ValueError, yaml.YAMLError, cv2.error) as exc:
        st.error(f"{upload.name} · {exc}")
