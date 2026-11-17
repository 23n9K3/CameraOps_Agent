"""Launch a LangGraph run and keep its live events visible across reruns."""

import streamlit as st
import yaml

from src.agents.deepseek import configured
from src.orchestration.graph import run_workflow
from src.orchestration.state import WorkflowRequest
from src.tools.contracts import SourceSpec
from src.visualization.catalog import artifact_root, burst_label, catalog, final_download
from src.visualization.jobs import Job
from src.visualization.workspace import live_workspace

st.title("Agent 工作台")
root = artifact_root()
bursts = catalog("burst", artifact_type="imported_raw_burst")
bursts = dict(sorted(bursts.items(), key=lambda item: not item[1].get("source", {}).get("scene_label")))
job = st.session_state.get("workflow_job") if st.session_state.get("workflow_job_source") in bursts else None
running = job is not None and not job.done
if not bursts:
    st.info("请先导入同一场景的实拍 RAW 序列。")
    st.page_link("app_pages/raw_dataset.py", label="打开 RAW 数据集", icon=":material/folder_open:")
selected = st.selectbox("实拍 RAW 序列", list(bursts),
                        format_func=lambda raw: burst_label(bursts[raw], raw), disabled=not bursts or running)
with st.form("agent_request", border=True):
    goal = st.text_area("拍摄目标", "给这组实拍 RAW 去噪，保留纹理和细节。", height=85)
    with st.expander("执行设置"):
        brain = st.selectbox("Agent 大脑", ["offline_mock", "deepseek"], index=int(configured()),
                             format_func=lambda value: "DeepSeek" if value == "deepseek" else "离线规则")
        st.caption("DeepSeek 接收拍摄目标和数值场景摘要。API 密钥从环境变量读取。")
        columns = st.columns(2)
        iterations = columns[0].number_input("最大策略修订次数", min_value=0, max_value=8, value=2)
        experiments = columns[1].number_input("最大摄影实验数", min_value=1, max_value=9, value=3)
        feedback = st.checkbox("首轮关闭去鬼影以演示故障反馈", value=False)
    submit = st.form_submit_button("运行 Agent 工作流", type="primary", icon=":material/play_arrow:", disabled=running or not bursts)

if submit and selected in bursts and not running:
    try:
        if brain == "deepseek" and not configured():
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量后再使用 DeepSeek。")
        request = WorkflowRequest(user_request=goal.strip(), source=SourceSpec(burst_id=selected),
                                  brain=brain, max_iterations=iterations, max_experiments=experiments,
                                  initial_parameters={"deghost": False} if feedback else {})
        st.session_state["workflow_job"] = Job(lambda emit: run_workflow(request, root, on_event=emit))
        st.session_state["workflow_job_source"] = selected
        st.rerun()
    except (ValueError, OSError, yaml.YAMLError) as exc:
        st.error(str(exc))

if job is not None and (running or st.session_state.get("workflow_job_source") == selected):
    live_workspace(job)
elif selected:
    for report in catalog("reports", artifact_type="camera_workflow_report").values():
        output = report.get("final_artifact_id")
        if report.get("status") == "completed" and report.get("input_artifacts") == [selected] and output:
            st.subheader("最近实拍处理结果")
            final_download("photography", output, key="recent_real_photo")
            break
