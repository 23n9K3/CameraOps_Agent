# AI Computational Photography Lab（AI-CPL）

基于多智能体协同的智能计算摄影与 Camera 分析优化平台

项目包含 RAW 数据集、采集规划、计算摄影、Agent 闭环、实验优化、Streamlit 可视化及 P9 工程验收。运行需要 Python 3.10+，无需相机硬件或 API Key。功能状态见 [验收清单](docs/engineering_acceptance.md)。

## 安装与运行

在仓库根目录运行。Windows PowerShell 示例如下。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_capture_simulation.py --config configs/synthetic_burst.yaml
.\.venv\Scripts\python.exe -m streamlit run app.py
```

从 **RAW 数据集** 可逐帧查看 Bayer、去马赛克预览、饱和掩码、Ground Truth、曝光直方图与元数据，也可上传 LibRaw 支持的相机容器或提供布局的无头 Bayer RAW。**采集规划** 根据用户目标与当前场景测量生成 CapturePlan，并执行模拟采集。

## P1 RAW 命令行

```powershell
.\.venv\Scripts\python.exe scripts/generate_raw_example.py
.\.venv\Scripts\aicpl-raw.exe --config configs/raw12_example.yaml data/synthetic/p1_raw12.rggb
```

处理同布局的多帧时，可在命令末尾按采集顺序添加输入路径。对于 LibRaw 能解码的 2×2 Bayer 相机文件，省略 `--config`。

```powershell
.\.venv\Scripts\aicpl-raw.exe data/raw/iss030e122639.NEF
```

每帧输出一行 JSON，包含输入路径、RAW 布局、饱和采样点比例、预处理耗时和产物 ID。产物目录默认为 `artifacts/raw/<artifact_id>/`，可用 `--artifacts` 指定其他目录。失败写入标准错误，并以退出码 `1` 标记。

## P1 交付范围

- 无头 Bayer RAW 支持 8 位单字节、有效精度为 8、10、12、14 或 16 位的 16 位容器，以及 MIPI Packed RAW10、RAW12、RAW14。
- 支持 RGGB、BGGR、GRBG、GBRG，以及 16 位容器大小端、逐行填充和按 Bayer 位置配置的黑白电平。
- 输出原始采样码、归一化 Bayer、饱和掩码、线性 RGB 和 PNG 预览；源文件以只读方式打开。
- 真实相机容器由 LibRaw 解码，兼容性取决于文件和相机型号。[rawpy 格式说明](https://github.com/letmaik/rawpy#raw-file-format-support)

## P2 模拟产物与测试

模拟器按每帧配置执行曝光积分和模拟增益，加入 Shot Noise、Read Noise、热／死像素、DSNU、PRNU、行列噪声，并模拟运动模糊、传感器饱和和 ADC 量化。

`artifacts/burst/<artifact_id>/` 保存 RAW Burst、曝光中点和曝光平均的无噪场景 Ground Truth、主体与缺陷像素掩码、预览和 `metadata.json`。详见 [P2 模型与验收说明](docs/p2_capture_simulation.md) 和 [RAW 布局与验收记录](docs/p1_raw_reader.md)。

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## P3 多帧计算摄影

```powershell
.\.venv\Scripts\python.exe scripts/run_photography_demo.py
.\.venv\Scripts\python.exe scripts/run_photography.py --mode motion --burst <BURST_ID>
.\.venv\Scripts\python.exe scripts/run_photography.py --mode hdr --raw-manifest sequence.yaml
```

每次处理在 `artifacts/photography/<artifact_id>/` 发布最终 PNG、诊断预览、线性中间数组、配准矩阵、光流、运动掩码、融合权重和分阶段耗时。合成 Burst 会自动用匹配参考帧的 Ground Truth 计算 PSNR 与分区 RMSE。见 [P3 输入格式、算法与验收说明](docs/p3_photography.md)。

## P4 至 P6 场景、IQ 与 Agent 闭环

```powershell
.\.venv\Scripts\python.exe scripts/run_agent_workflow.py --goal "处理夜景 RAW，降低噪声并减少运动行人鬼影"
.\.venv\Scripts\python.exe scripts/run_agent_feedback_demo.py
```

六个 Agent 在 LangGraph 中传递 CapturePlan、产物 ID、结构化测量和修复参数。`--burst` 可使用已有 Burst，`--raw-manifest` 可读入本地序列。每次运行发布 `artifacts/reports/<report_id>/metadata.json`，记录每轮 IQ、诊断验证、工具调用和最终产物 ID。见 [工作流、诊断规则与验收](docs/p4_p6_agent_workflow.md)。

Agent 工作台支持选择 **DeepSeek**。配置 `DEEPSEEK_API_KEY` 后可用 `--brain deepseek` 从命令行运行。用 `scripts/import_real_burst.py` 导入具有 EXIF 曝光数据的实拍连拍。见 [DeepSeek 配置与真实 DNG 测试](docs/deepseek_real_raw.md)。

## P7 实验记忆与多目标优化

```powershell
.\.venv\Scripts\python.exe scripts/query_experiments.py --mode hdr --limit 5
.\.venv\Scripts\python.exe scripts/query_experiments.py --import-reports artifacts
.\.venv\Scripts\python.exe scripts/run_optimization.py --mode hdr --trials 24 --seed 71
```

LangGraph 的摄影轮次和 Optuna trial 写入 `artifacts/experiments.sqlite3`。优化报告发布于 `artifacts/optimization/<artifact_id>/`，包含 Pareto HTML、试验 CSV、基线与选中方案的实测对照。见 [P7 参数、指标、历史采纳规则和复现说明](docs/p7_experiment_memory.md)。

## P8 Streamlit 工作台

Windows 重启后，可双击 `Open-Agent-Workspace.cmd` 启动本地 Streamlit 服务并打开 Agent 工作台。服务正常运行时，启动器直接打开页面。启动失败可查看 `artifacts/streamlit.log`。

界面包含总览、RAW 数据集、采集规划、计算摄影、Camera 调试、Agent 工作台、实验中心和报告中心。Agent 工作台在执行期间按事件更新当前任务、工具调用及返回、耗时、迭代和诊断。摄影结果支持同坐标局部对比、直方图、运动掩码、融合权重和 PNG 导出。[界面与验收说明](docs/p8_streamlit_workbench.md)

## P9 工程验收与基准

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_agent_benchmark.py
```

默认基准对 30 个确定性合成 RAW 任务实际运行五种策略。每次运行在 `artifacts/benchmarks/<run_id>/` 保存输入、结果、Agent 轨迹、实测对照、PSNR／SSIM 曲线、Noise／Ghosting 图和消融报告。查看 [实验协议与判定门槛](docs/experiment_protocol.md) 和 [验收清单与未完成功能](docs/engineering_acceptance.md)。
