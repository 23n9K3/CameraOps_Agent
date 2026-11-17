"""AI-CPL laboratory navigation."""

import streamlit as st

st.set_page_config(page_title="AI-CPL · 计算摄影实验室", page_icon=":material/camera:", layout="wide")
st.sidebar.markdown("**AI-CPL**  \n计算摄影实验室")
pages = [
    st.Page("app_pages/dashboard.py", title="总览", icon=":material/dashboard:", default=True),
    st.Page("app_pages/raw_dataset.py", title="RAW 数据集", icon=":material/folder_open:", url_path="raw"),
    st.Page("app_pages/capture_planner.py", title="采集规划", icon=":material/photo_camera:", url_path="capture"),
    st.Page("app_pages/photography.py", title="计算摄影", icon=":material/tune:", url_path="photography"),
    st.Page("app_pages/debugging.py", title="Camera 调试", icon=":material/bug_report:", url_path="debugging"),
    st.Page("app_pages/agent_workspace.py", title="Agent 工作台", icon=":material/account_tree:", url_path="workspace"),
    st.Page("app_pages/experiments.py", title="实验中心", icon=":material/science:", url_path="experiments"),
    st.Page("app_pages/reports.py", title="报告中心", icon=":material/description:", url_path="reports"),
]
st.navigation(pages).run()
