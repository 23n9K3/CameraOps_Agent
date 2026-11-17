"""Review and revalidate diagnostic evidence, with before/after image crops."""

import cv2
import streamlit as st

from src.debugging.diagnosis import QualityLimits
from src.storage.artifact_store import ArtifactStore
from src.tools.contracts import ValidationInput
from src.tools.operations import validate
from src.visualization.catalog import artifact_root, catalog, choose_artifact
from src.visualization.inspection import compare_views
from src.visualization.preview import display_rgb
from src.visualization.workspace import diagnosis_details

st.title("Camera 调试")
root = artifact_root()
reports = catalog("reports", artifact_type="camera_workflow_report")
if identifier := choose_artifact("工作流报告", reports, key="debug_report",
                                 names=lambda meta, item: f'{meta["user_request"][:44]} · {item[:8]}'):
    report = reports[identifier]
    records = report.get("debugging_results", [])
    if records:
        st.subheader("故障假设与验证")
        diagnosis_details(records)
    else:
        st.info("该工作流没有记录到故障诊断。")
    comparisons = report.get("repair_comparisons", [])
    if comparisons:
        st.subheader("修复前后")
        index = st.selectbox("对照轮次", list(range(len(comparisons))), format_func=lambda i: f"修订 {i + 1}")
        pair = comparisons[index]
        photos = ArtifactStore(root / "photography")
        try:
            images = [(label, display_rgb(photos.load_array(pair[key], "tone_mapped_rgb", mmap_mode="r")))
                      for label, key in (("修复前", "before_artifact_id"), ("修复后", "after_artifact_id"))]
            compare_views(images, key=f"repair_{identifier}_{index}")
        except (OSError, ValueError) as exc:
            st.warning(str(exc))
        st.dataframe([{"指标": name, "修复前": values["before"], "修复后": values["after"]}
                      for name, values in pair["metrics"].items()], hide_index=True, alt="真实图像重处理前后的质量指标")
    attempts = report.get("attempts", [])
    if attempts:
        with st.form("revalidate", border=True):
            i = st.selectbox("复核摄影轮次", list(range(len(attempts))), format_func=lambda j: f"第 {j + 1} 轮")
            fault = st.selectbox("故障假设", ["registration_failure", "ghosting"])
            check = st.form_submit_button("从产物重新验证", icon=":material/fact_check:")
        if check:
            try:
                result = validate(ValidationInput(artifact_root=root, artifact_id=attempts[i]["artifact_id"], fault=fault,
                                                   limits=QualityLimits.model_validate(report["quality_limits"])))
                st.json(result.model_dump(mode="json"), expanded=2)
            except (OSError, ValueError, cv2.error) as exc:
                st.error(str(exc))
else:
    st.page_link("app_pages/agent_workspace.py", label="执行带反馈的 Agent 工作流", icon=":material/account_tree:")
