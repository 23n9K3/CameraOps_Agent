"""Inspect a live-scene capture plan and execute it through the simulator."""

from uuid import uuid4

import cv2
import streamlit as st
import yaml

from src.agents.capture_planner_agent import consult_history, plan_capture
from src.agents.supervisor import interpret_goal
from src.camera.capture_simulator import SimulationConfig
from src.memory.experiment_memory import ExperimentMemory
from src.storage.artifact_store import ArtifactStore
from src.tools.contracts import CaptureInput, SourceSpec, SurveyInput
from src.tools.operations import capture, survey
from src.visualization.catalog import PROJECT, artifact_root

st.title("采集规划")
root = artifact_root()
with st.form("plan_capture", border=True):
    goal = st.text_input("拍摄目标", value="处理夜景，降低噪声并减少运动拖影")
    with st.expander("场景与传感器模板"):
        template_text = st.text_area("模拟配置 YAML", (PROJECT / "configs/synthetic_burst.yaml").read_text(encoding="utf-8"), height=230)
    generate = st.form_submit_button("生成采集计划", type="primary", icon=":material/photo_camera:")
if generate:
    st.session_state.pop("planner_snapshot", None)
    st.session_state.pop("planner_burst", None)
    try:
        if not goal.strip():
            raise ValueError("请输入拍摄目标。")
        template = SimulationConfig.model_validate(yaml.safe_load(template_text))
        measured = survey(SurveyInput(artifact_root=root, source=SourceSpec(simulation=template))).features
        plan = plan_capture(interpret_goal(goal), measured, simulated=True, gap=template.capture_plan.inter_frame_gap_s)
        plan, review = consult_history(plan, measured, ExperimentMemory(root / "experiments.sqlite3"),
                                       source_kind="synthetic_survey", task_id=uuid4().hex,
                                       max_frames=4_194_304 // (template.scene.width * template.scene.height))
        st.session_state["planner_snapshot"] = (template, plan, measured, review)
    except (OSError, ValueError, yaml.YAMLError, cv2.error) as exc:
        st.error(str(exc))

if saved := st.session_state.get("planner_snapshot"):
    template, plan, measured, review = saved
    summary, exposures = st.columns([0.85, 1.15], gap="large")
    with summary:
        st.subheader("场景测量")
        with st.container(horizontal=True):
            st.metric("亮度中值", f"{measured.brightness_median:.3f}", border=True)
            st.metric("检测到的运动区域", f"{(measured.motion_fraction or 0):.1%}", border=True)
        st.write(plan.rationale)
        with st.expander("结构化计划与历史检查"):
            st.json({"计划": plan.model_dump(mode="json"), "历史检查": review}, expanded=False)
    with exposures:
        st.subheader("逐帧曝光计划")
        st.dataframe([{"帧": i + 1, "曝光 (ms)": round(seconds * 1000, 3), "模拟增益 (×)": gain,
                       "角色": "参考帧" if i == plan.reference_frame else "参与融合"}
                      for i, (seconds, gain) in enumerate(zip(plan.exposure_times, plan.gains))],
                     hide_index=True, alt="每帧曝光、模拟增益和参考帧选择")
    if st.button("执行模拟采集", type="primary", icon=":material/play_arrow:"):
        try:
            with st.spinner("按计划模拟曝光和传感器读出…"):
                result = capture(CaptureInput(artifact_root=root, simulation=template, plan=plan))
            st.session_state["planner_burst"] = result.artifact_id
        except (OSError, ValueError, cv2.error) as exc:
            st.error(str(exc))
    if identifier := st.session_state.get("planner_burst"):
        st.success(f"模拟 Burst 已发布 `{identifier}`")
        preview = ArtifactStore(root / "burst").path(identifier) / "preview.png"
        st.image(preview.read_bytes(), width="stretch", alt="按计划逐帧模拟采集的 RAW 缩略图")
        st.page_link("app_pages/raw_dataset.py", label="检查模拟 RAW 与直方图", icon=":material/folder_open:")
