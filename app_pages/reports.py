"""Browse completed report artifacts and download common export formats."""

import streamlit as st

from src.storage.artifact_store import ArtifactStore
from src.visualization.catalog import artifact_root, catalog, choose_artifact, final_download
from src.visualization.report_exports import export_csv, export_html, metric_rows
from src.visualization.workspace import STATUS

st.title("报告中心")
category = st.segmented_control("报告类型", ["Agent 工作流", "多目标优化"], default="Agent 工作流")
workflow = category != "多目标优化"
folder = "reports" if workflow else "optimization"
reports = catalog(folder)
if identifier := choose_artifact("选择报告", reports, key=f"reports_{folder}",
                                 names=lambda meta, item: f'{(meta.get("user_request") or meta["study_name"])[:55]} · {item[:10]}'):
    report = reports[identifier]
    rows = metric_rows(report)
    if workflow:
        with st.container(horizontal=True):
            st.metric("任务状态", STATUS.get(report["status"], report["status"]), border=True)
            st.metric("摄影轮次", report["experiment_count"], border=True)
            st.metric("策略修订", report["iteration_count"], border=True)
        st.caption(report.get("termination_reason") or "")
        with st.expander("执行记录"):
            st.json(report.get("agent_trace", []), expanded=False)
    else:
        with st.container(horizontal=True):
            st.metric("成功试验", report["completed_trials"], border=True)
            st.metric("Pareto 点", len(report["pareto_front"]), border=True)
            st.metric("选中试验", report["selected"]["number"], border=True)
    st.subheader("报告导出")
    path = ArtifactStore(artifact_root() / folder).path(identifier)
    with st.container(horizontal=True):
        for label, payload, mime in (("JSON", (path / "metadata.json").read_bytes(), "application/json"),
                                     ("CSV", export_csv(rows), "text/csv"), ("HTML", export_html(report, rows), "text/html")):
            st.download_button(f"下载报告 {label}", payload, f"aicpl_{identifier}.{label.lower()}", mime,
                               on_click="ignore", icon=":material/download:")
    if rows:
        st.dataframe(rows, hide_index=True, alt="报告中的实际质量指标及有效性")
    final_id = report.get("final_artifact_id") if workflow else report["selected"]["output_artifact_id"]
    if final_id:
        st.subheader("最终处理结果")
        final_download("photography", final_id, key="report_final")
