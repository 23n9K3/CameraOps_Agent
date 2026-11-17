"""Aligned image comparisons, ROI zoom, histograms, masks and weights."""

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.computational.fusion import tone_map
from src.storage.artifact_store import ArtifactStore

from .catalog import final_download
from .preview import display_rgb

MODES = {"night": "Night Mode", "hdr": "HDR", "motion": "Motion-aware"}


def crop_box(shape: tuple[int, int], zoom: int, x: int, y: int) -> tuple[slice, slice]:
    height, width = shape
    visible_w, visible_h = max(1, width // zoom), max(1, height // zoom)
    left = round((width - visible_w) * x / 100)
    top = round((height - visible_h) * y / 100)
    return slice(top, top + visible_h), slice(left, left + visible_w)


def _region(shape, key):
    zoom = st.select_slider("放大倍率", [1, 2, 4, 8], value=1, format_func=lambda value: f"{value}×", key=f"{key}_zoom")
    x = y = 50
    if zoom > 1:
        left, right = st.columns(2)
        x = left.slider("水平位置 (%)", 0, 100, 50, key=f"{key}_x")
        y = right.slider("垂直位置 (%)", 0, 100, 50, key=f"{key}_y")
    rows, columns = crop_box(shape, zoom, x, y)
    st.caption(f"画面区域 x {columns.start}–{columns.stop - 1}，y {rows.start}–{rows.stop - 1} px")
    return rows, columns


def compare_views(images: list[tuple[str, np.ndarray]], *, key: str, histogram: bool = True) -> None:
    if len({image.shape for _, image in images}) != 1:
        st.warning("图像尺寸不同，无法做同坐标对比。")
        return
    region = _region(images[0][1].shape[:2], key)
    crops = [(label, image[region]) for label, image in images]
    if len(crops) == 1:
        st.image(crops[0][1], width="stretch", alt=crops[0][0], caption=crops[0][0])
    else:
        for column, (label, image) in zip(st.columns(len(crops), gap="medium"), crops):
            column.image(image, width="stretch", caption=label, alt=label)
    if histogram:
        with st.expander("局部亮度直方图"):
            figure = go.Figure()
            for label, pixels in crops:
                gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if pixels.ndim == 3 else pixels
                counts, edges = np.histogram(gray, bins=64, range=(0, 256))
                figure.add_scatter(x=(edges[:-1] + edges[1:]) / 2, y=counts, name=label, mode="lines")
            figure.update_layout(height=250, margin={"t": 15, "b": 0}, xaxis_title="显示亮度 (0–255)", yaxis_title="像素数")
            st.plotly_chart(figure, key=f"{key}_hist", alt="同坐标裁剪区域的亮度直方图")


def photography_viewer(store: ArtifactStore, identifier: str) -> None:
    metadata = store.load_metadata(identifier)
    pipeline = metadata["pipeline"]
    color_matrix = metadata.get("source", {}).get("display_color_matrix")
    if color_matrix is not None:
        color_matrix = np.asarray(color_matrix, dtype=np.float32)
    reference = metadata["reference_index"]
    count = len(metadata["registration"])
    left, right = st.columns(2)
    modes = ["参考帧与最终图", "配准前后", "去鬼影前后", "运动掩码", "融合权重", "最终图"]
    view = left.selectbox("图像对比", modes, key=f"view_{identifier}")
    frame = right.selectbox("中间结果帧", list(range(count)), index=metadata.get("diagnostic_frame_index", reference),
                            format_func=lambda i: f"第 {i + 1} 帧" + (" · 参考帧" if i == reference else ""),
                            key=f"inspect_frame_{identifier}")

    def array(name):
        return store.load_array(identifier, name, mmap_mode="r")

    def preview(linear):
        if color_matrix is not None:
            linear = cv2.transform(linear.astype(np.float32, copy=False), color_matrix)
        return display_rgb(tone_map(linear, metadata["tone_exposure"], pipeline.get("tone_mapping_strength", 1.0)))

    if view == "配准前后":
        peer = array("demosaiced_sensor_rgb")[frame] / metadata["dose_scales"][frame]
        images = [("配准前", preview(peer)), ("配准后", preview(array("aligned_rgb")[frame]))]
    elif view == "运动掩码":
        mask = array("motion_masks")[frame]
        st.caption(f"运动区覆盖率 {mask.mean():.2%}")
        images = [("当前帧运动掩码", mask.astype(np.uint8) * 255)]
    elif view == "融合权重":
        channels = array("fusion_weights")[frame]
        channel = st.segmented_control("权重通道", ["RGB 平均", "R", "G", "B"], default="RGB 平均")
        weight = channels.mean(axis=-1) if channel not in ("R", "G", "B") else channels[..., "RGB".index(channel)]
        st.caption(f"黑色 0，白色 1 · 当前帧平均权重 {weight.mean():.3f}")
        images = [("当前帧融合权重", np.rint(weight * 255).astype(np.uint8))]
    else:
        final = display_rgb(array("tone_mapped_rgb"))
        images = [("最终结果", final)]
        if view == "参考帧与最终图":
            images.insert(0, ("参考帧", preview(array("aligned_rgb")[reference])))
        elif view == "去鬼影前后":
            images.insert(0, ("关闭去鬼影的融合", preview(array("fusion_without_deghosting"))))
    compare_views(images, key=f"photo_{identifier}", histogram=view not in ("运动掩码", "融合权重"))
    with st.expander("配准与处理耗时"):
        st.dataframe(metadata["registration"], hide_index=True, alt="逐帧配准方法和残差")
        st.json(metadata["runtime_ms"], expanded=False)
    final_download("photography", identifier, key=f"photo_download_{identifier}", show=False)
