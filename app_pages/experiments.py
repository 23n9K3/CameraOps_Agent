"""Query SQLite experiments and run or inspect measured Optuna studies."""

import streamlit as st

from src.computational.pipeline import PipelineConfig
from src.memory.experiment_memory import ExperimentMemory
from src.optimization.optimizer import OptimizationConfig, optimize_burst, pareto_figure
from src.storage.artifact_store import ArtifactStore
from src.visualization.catalog import artifact_root, burst_label, catalog, choose_artifact
from src.visualization.inspection import MODES
from src.visualization.jobs import Job

st.title("实验中心")
root = artifact_root()
memory = ExperimentMemory(root / "experiments.sqlite3")
view = st.segmented_control("实验操作", ["历史记录", "Optuna 与 Pareto"], default="历史记录")

if view == "Optuna 与 Pareto":
    job = st.session_state.get("optimization_job")
    busy = job is not None and not job.done
    bursts = catalog("burst", artifact_type="synthetic_raw_burst")
    if bursts:
        with st.form("optimize", border=True):
            columns = st.columns([1.3, 1, 1])
            raw = columns[0].selectbox("带 Ground Truth 的 Burst", list(bursts), format_func=lambda i: burst_label(bursts[i], i))
            mode = columns[1].selectbox("优化模式", list(MODES), index=1, format_func=MODES.__getitem__)
            trials = columns[2].number_input("搜索试验数", min_value=1, max_value=250, value=18)
            submit = st.form_submit_button("运行 Optuna 优化", type="primary", disabled=busy, icon=":material/play_arrow:")
        if submit and not busy:
            st.session_state["optimization_job"] = Job(lambda emit: {"report_id": optimize_burst(
                root, raw, PipelineConfig(mode=mode), OptimizationConfig(trials=trials))})
            st.rerun()
    else:
        st.page_link("app_pages/capture_planner.py", label="生成带 Ground Truth 的 Burst", icon=":material/photo_camera:")

    if busy:
        @st.fragment(run_every=1.0)
        def optimization_progress():
            snapshot = job.snapshot()
            if snapshot["done"]:
                st.rerun()
            st.info(f'正在执行试验 · {snapshot["elapsed_ms"] / 1000:.1f} s')
        optimization_progress()
    elif job:
        snapshot = job.snapshot()
        if snapshot["error"]:
            st.error(snapshot["error"])
        elif snapshot["result"]:
            identifier = snapshot["result"]["report_id"]
            if st.session_state.get("displayed_optimization_job") != identifier:
                st.session_state["displayed_optimization_job"] = identifier
                st.session_state["optimization_report"] = identifier
            st.success(f"优化报告 `{identifier}` 已发布。")

    reports = catalog("optimization")
    if identifier := choose_artifact("优化研究", reports, key="optimization_report",
                                     names=lambda meta, item: f'{MODES[meta["mode"]]} · {meta["trial_count"]} 次 · {item[:10]}'):
        report = reports[identifier]
        front = {row["number"] for row in report["pareto_front"]}
        # Preserve the stored objective order even for older reports.
        active = [name for name, definition in report["objectives"].items() if definition["active"]]
        with st.container(horizontal=True):
            st.metric("有效试验", f'{report["completed_trials"]}/{report["trial_count"]}', border=True)
            st.metric("Pareto 点", len(front), border=True)
            st.metric("选中试验", report["selected"]["number"], border=True)
        st.plotly_chart(pareto_figure(report["trials"], active, front, report["selected"]["number"]),
                        alt="真实试验在目标空间中的 Pareto 前沿、基线和选中方案")
        st.subheader("基线与选中方案")
        st.dataframe([{"目标": name, **values} for name, values in report["objective_comparison"].items()],
                     hide_index=True, alt="优化前后实测目标")
        with st.expander("全部试验"):
            st.dataframe([{"试验": row["number"], "状态": row["status"], "耗时 (ms)": row["runtime_ms"],
                           "摄影产物": row["output_artifact_id"], **row["metrics"]} for row in report["trials"]],
                         hide_index=True, alt="每个优化试验的实际状态、耗时和指标")
        path = ArtifactStore(root / "optimization").path(identifier)
        with st.container(horizontal=True):
            st.download_button("下载 Pareto HTML", (path / "pareto.html").read_bytes(), "pareto.html", "text/html", on_click="ignore")
            st.download_button("下载试验 CSV", (path / "trials.csv").read_bytes(), "trials.csv", "text/csv", on_click="ignore")
else:
    columns = st.columns(3)
    mode = columns[0].selectbox("摄影模式", [None, *MODES], format_func=lambda value: MODES[value] if value else "全部")
    kind = columns[1].selectbox("记录类型", [None, "workflow", "optimization"], format_func=lambda value: value or "全部")
    artifact = columns[2].text_input("输入或输出产物 ID", placeholder="可选")
    rows = memory.history(mode=mode, kind=kind, artifact_id=artifact.strip() or None, limit=100)
    if rows:
        st.dataframe([{"时间 (UTC)": row["created_at"], "模式": MODES[row["mode"]], "类型": row["kind"],
                       "状态": row["status"], "耗时 (ms)": round(row["runtime_ms"], 2), "产物": row["output_artifact_id"]}
                      for row in rows], hide_index=True, alt="SQLite 历史摄影及优化实验")
        index = st.selectbox("记录详情", list(range(len(rows))),
                             format_func=lambda i: f'{rows[i]["group_id"][:25]} · 试验 {rows[i]["number"]}')
        row = rows[index]
        metrics = row["metrics"].get("metrics")
        if metrics:
            st.dataframe([{"指标": name, "数值": m["value"], "单位": m["unit"], "测量说明": m["limitations"]}
                          for name, m in metrics.items()], hide_index=True, alt="落盘实验的实测 IQ 指标")
        with st.expander("完整配置与测量记录"):
            st.json(row, expanded=False)
    else:
        st.info("没有匹配的实验记录。")
