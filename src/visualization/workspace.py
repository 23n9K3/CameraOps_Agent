"""Live readout of LangGraph nodes and registered tool attempts."""

import streamlit as st

from src.storage.artifact_store import ArtifactStore

from .catalog import artifact_root, final_download
from .inspection import photography_viewer
from .jobs import Job

NODES = {
    "supervisor": ("Supervisor", "解析拍摄目标"),
    "survey": ("Scene & Motion", "调查亮度、曝光和运动条件"),
    "brain": ("DeepSeek", "依据拍摄目标和场景测量选择摄影策略"),
    "capture_planner": ("Capture Planner", "制定计划、选择参考帧并核验历史建议"),
    "capture": ("Capture Simulator", "按曝光与增益模拟采集，或沿用输入序列"),
    "scene_motion": ("Scene & Motion", "测量 Burst 的亮度、清晰度和运动"),
    "photography": ("Computational Photography", "运行配准、融合和色调映射"),
    "iq": ("IQ Evaluation", "从图像产物计算质量指标"),
    "decision": ("Supervisor", "检查实测指标与故障门限"),
    "debugging": ("Camera Debugging", "从产物重算证据并验证故障"),
    "strategy_revision": ("Supervisor", "检查预算并应用经验证的修复参数"),
    "report": ("Report", "保存诊断、实测对照和最终结果"),
}
STATUS = {"running": "运行中", "ok": "完成", "error": "失败", "invalid_input": "参数无效",
          "completed": "完成", "needs_review": "需复核", "failed": "失败"}


def diagnosis_details(records: list[dict]) -> None:
    for record in records:
        st.markdown(f'**第 {record["iteration"]} 次修订对应的诊断** · `{record["artifact_id"][:12]}`')
        for diagnosis in record["diagnoses"]:
            title = f'{diagnosis["fault"]} · {diagnosis["validation_result"]} · 置信度 {diagnosis["confidence"]:.2f}'
            with st.expander(title):
                st.write(diagnosis["hypothesis"])
                st.write(diagnosis["recommended_action"])
                st.json(diagnosis["evidence"], expanded=False)
        if record["proposed_changes"]:
            st.json({"建议参数": record["proposed_changes"]}, expanded=False)


def render_live(job: Job, polling: bool) -> None:
    snapshot = job.snapshot()
    if polling and snapshot["done"]:
        st.rerun()
    events = snapshot["events"]
    starts = [event for event in events if event["type"] == "node_started"]
    finished = [event for event in events if event["type"] == "node_finished"]
    tools = [(index, event) for index, event in enumerate(events) if event["type"] == "tool_finished"]
    iteration = max([0, *[event.get("updates", {}).get("iteration_count", event["iteration"]) for event in finished]])
    state = snapshot["result"] or {}
    active = starts[-1] if len(starts) > len(finished) and not snapshot["done"] else None
    label, task = NODES[active["node"]] if active else ("准备中", "等待执行节点")
    if snapshot["done"]:
        label = STATUS.get(state.get("status"), "执行失败" if snapshot["error"] else "完成")
        task = state.get("termination_reason", "")
    newest = events[-1] if events else {}
    active_tool = newest if newest.get("type") == "tool_started" and not snapshot["done"] else None

    with st.container(horizontal=True):
        st.metric("当前状态", label, border=True)
        st.metric("策略修订次数", iteration, border=True)
        st.metric("累计耗时", f'{snapshot["elapsed_ms"] / 1000:.2f} s', border=True)
        st.metric("已返回的工具调用", len(tools), border=True)
    if task:
        st.caption(task)
    if active_tool:
        spent = max(0, snapshot["elapsed_ms"] - active_tool["elapsed_ms"])
        st.info(f'正在调用 `{active_tool["tool"]}` · 尝试 {active_tool["attempt"]} · {spent / 1000:.2f} s')
    if snapshot["error"]:
        st.error(snapshot["error"])
    for error in state.get("errors", []):
        st.error(f'{error["node"]} · {error["error"]}')

    timeline = [{"Agent": NODES[event["node"]][0], "任务": NODES[event["node"]][1], "迭代": event["iteration"],
                 "状态": STATUS[event["status"]], "耗时 (ms)": round(event.get("runtime_ms", 0), 1)} for event in finished]
    if active:
        timeline.append({"Agent": label, "任务": task, "迭代": active["iteration"], "状态": "运行中",
                         "耗时 (ms)": round(snapshot["elapsed_ms"] - active["elapsed_ms"], 1)})
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Agent 执行轨迹")
        if timeline:
            st.dataframe(timeline[::-1], hide_index=True, alt="各节点运行状态、迭代与实测耗时")
    with right:
        st.subheader("Tool 调用与返回")
        if tools:
            tool_rows = dict(tools)
            selected = st.selectbox("查看工具返回", [None, *list(tool_rows)[::-1]], key=f"live_tool_{id(job)}",
                                    format_func=lambda i: ("最新返回" if i is None else
                                                           f'{tool_rows[i]["tool"]} · 第 {tool_rows[i]["attempt"]} 次 · {i + 1}'))
            outcome = tools[-1][1] if selected is None else tool_rows[selected]
            st.caption(f'{STATUS[outcome["status"]]} · {outcome["runtime_ms"]:.1f} ms')
            if outcome.get("error"):
                st.error(outcome["error"])
            else:
                st.json(outcome["result"], expanded=1)
        else:
            st.caption("工具返回后会在这里显示结果。")

    records = [item for event in finished for item in event.get("updates", {}).get("debugging_results", [])]
    published = [event["updates"]["photography_results"]["artifact_id"] for event in finished
                 if "photography_results" in event.get("updates", {})]
    if records:
        st.subheader("故障诊断")
        diagnosis_details(records)
    decision = next((event["updates"]["brain_decision"] for event in finished
                     if "brain_decision" in event.get("updates", {})), None)
    if decision:
        with st.expander("DeepSeek 策略", expanded=True):
            st.json(decision, expanded=True)
    if published and not snapshot["done"]:
        st.caption(f"最新摄影产物 `{published[-1]}` · IQ 评估与决策在执行轨迹中更新")
    if state.get("report_id"):
        st.caption(f'报告产物 `{state["report_id"]}`')
    if state.get("final_artifact_id"):
        st.subheader("最终处理结果")
        if st.toggle("检查局部、掩码与融合权重"):
            photography_viewer(ArtifactStore(artifact_root() / "photography"), state["final_artifact_id"])
        else:
            final_download("photography", state["final_artifact_id"], key="workspace_final")


def live_workspace(job: Job) -> None:
    polling = not job.done
    st.fragment(render_live, run_every=0.5 if polling else None)(job, polling)
