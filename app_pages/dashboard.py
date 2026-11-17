"""Summarize real tasks, trials and recently published results."""

import streamlit as st

from src.memory.experiment_memory import ExperimentMemory
from src.storage.artifact_store import ArtifactStore
from src.visualization.catalog import artifact_root, catalog
from src.visualization.workspace import STATUS

st.title("计算摄影实验室")
root = artifact_root()
reports = catalog("reports", artifact_type="camera_workflow_report")
counts = ExperimentMemory(root / "experiments.sqlite3").counts()
successful = sum(report["status"] == "completed" for report in reports.values())
with st.container(horizontal=True):
    st.metric("Agent 任务", len(reports), border=True)
    st.metric("任务完成率", f"{successful / len(reports):.0%}" if reports else "待运行", border=True)
    st.metric("摄影实验", counts.get("workflow", 0), border=True)
    st.metric("优化试验", counts.get("optimization", 0), border=True)

left, right = st.columns([0.9, 1.1], gap="large")
with left:
    st.subheader("最近工作流")
    if reports:
        st.dataframe([{"拍摄目标": report["user_request"], "模式": report["task_type"],
                       "状态": STATUS.get(report["status"], report["status"]), "摄影轮次": report["experiment_count"]}
                      for report in list(reports.values())[:8]], hide_index=True, alt="最近 Agent 工作流的实测状态")
    else:
        st.info("运行一次 Agent 工作流即可在这里查看任务记录。")
    st.page_link("app_pages/agent_workspace.py", label="打开 Agent 工作台", icon=":material/account_tree:")
with right:
    st.subheader("最近处理结果")
    photos = catalog("photography", limit=1)
    if photos:
        identifier, metadata = next(iter(photos.items()))
        file = ArtifactStore(root / "photography").path(identifier) / "preview.png"
        if file.is_file():
            st.image(file.read_bytes(), width="stretch", alt="最近处理生成的实际照片")
            st.caption(f'{metadata["pipeline"]["mode"].upper()} · `{identifier[:12]}` · '
                       f'{metadata["runtime_ms"]["total_before_artifact_write"]:.1f} ms')
    else:
        st.info("摄影产物将在处理完成后显示。")
    st.page_link("app_pages/photography.py", label="查看摄影产物", icon=":material/tune:")
