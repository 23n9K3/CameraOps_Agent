"""Run pipeline processing and inspect saved photography artifacts."""

import cv2
import streamlit as st
import yaml

from src.computational.pipeline import PipelineConfig, run_pipeline
from src.computational.sequence import read_burst
from src.iq.evaluation import evaluate_image
from src.memory.experiment_memory import Experiment, ExperimentMemory
from src.storage.artifact_store import ArtifactStore
from src.visualization.catalog import artifact_root, burst_label, catalog, choose_artifact
from src.visualization.inspection import MODES, photography_viewer

st.title("计算摄影")
root = artifact_root()
raw_store, photos = ArtifactStore(root / "burst"), ArtifactStore(root / "photography")
bursts = catalog("burst")
if bursts:
    with st.form("pipeline_run", border=True):
        left, right = st.columns([1.4, 1])
        raw = left.selectbox("RAW Burst", list(bursts), format_func=lambda item: burst_label(bursts[item], item))
        mode = right.selectbox("流水线", list(MODES), format_func=MODES.__getitem__)
        with st.expander("处理参数"):
            configuration = st.text_area("PipelineConfig YAML", "{}", height=100)
        run = st.form_submit_button("运行摄影流水线", type="primary", icon=":material/play_arrow:")
    if run:
        try:
            parsed = yaml.safe_load(configuration)
            if not isinstance(parsed, dict):
                raise ValueError("处理参数必须是 YAML 映射。")
            options = PipelineConfig.model_validate(parsed | {"mode": mode})
            with st.spinner("正在配准和融合 RAW 序列…"):
                output = run_pipeline(read_burst(raw_store, raw), options, photos, truth_store=raw_store)
                st.session_state["photo_output"] = output
                iq = evaluate_image(photos, raw_store, output)
                source = bursts[raw].get("simulation_config") or {}
                ExperimentMemory(root / "experiments.sqlite3").record(Experiment(
                    group_id=output, number=0, kind="workflow", mode=mode, source_kind="manual_pipeline", status="complete",
                    input_artifact_id=raw, output_artifact_id=output, seed=source.get("seed"), runtime_ms=iq.metrics["runtime"].value,
                    capture_plan=source.get("capture_plan", {}), parameters=options.model_dump(mode="json"),
                    metrics=iq.model_dump(mode="json"), configuration={"source_format": bursts[raw]["source_format"]},
                ))
        except (OSError, ValueError, yaml.YAMLError, cv2.error) as exc:
            st.error(str(exc))
else:
    st.page_link("app_pages/raw_dataset.py", label="选择或导入 RAW 序列", icon=":material/folder_open:")

outputs = catalog("photography")
if selected := choose_artifact("摄影产物", outputs, key="photo_output",
                               names=lambda meta, item: f'{MODES[meta["pipeline"]["mode"]]} · {item[:12]}'):
    photography_viewer(photos, selected)
