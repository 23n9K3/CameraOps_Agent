# P7 实验记忆与多目标优化

## SQLite 实验记忆

`artifacts/experiments.sqlite3` 按任务 ID 或研究名、试验序号保存记录，包含输入和输出 Artifact ID、场景观测、CapturePlan、参数、种子、运行时间、指标、诊断和失败原因。修复历史要求故障在复测时解除，且相应的有效指标比首轮下降。旧工作流报告可导入并重复执行，使用任务 ID 和轮次去重。导入会核对现存输入、输出和 IQ 记录的 Artifact ID。

```powershell
python scripts/query_experiments.py --mode hdr --kind optimization --status complete --limit 5
python scripts/query_experiments.py --group <STUDY_OR_TASK_ID>
python scripts/query_experiments.py --artifact <INPUT_OR_OUTPUT_ID>
python scripts/query_experiments.py --similar-to <WORKFLOW_OUTPUT_ID>
python scripts/query_experiments.py --similar-to <WORKFLOW_OUTPUT_ID> --fault ghosting
python scripts/query_experiments.py --import-reports artifacts
```

相似性依据当前与历史场景的亮度中值、P99、高光比例、暗部比例和运动比例，在同模式、可比的观测来源中排序。至少需要三个有值的观测。距离阈值限制整体距离和单项偏离。

- Capture Planner 先依据当前观测拟定计划。历史记录可提出 Burst 帧数或 HDR 包围跨度，然后用当前曝光预测、运动时间窗和采集容量检查。增益和曝光量由当前场景计算。已有 RAW 沿用其固定采集设置。
- Debugging Agent 先从当前图像重算故障证据，再查找经复测成功的同类修复。鬼影严重度和支持区域达到条件时，历史阈值可按当前测量调整。修复后重新处理，IQ 重新计算，历史引用与采纳状态写入报告的 `memory_reviews`。

## 优化参数、目标与复现

```powershell
python scripts/run_optimization.py --mode hdr --trials 24 --seed 71
python scripts/run_optimization.py --mode hdr --burst <SYNTHETIC_BURST_ID> --trials 24 --seed 71
```

每项研究固定同一 RAW Burst、参考帧、评价区域和 Ground Truth，先实跑默认配置，再逐个实跑采样配置。`--trials` 指基线之外的试验数。固定种子的 NSGA-II 搜索五个参数，Optuna 自身的过程存入 `artifacts/optuna.sqlite3`。每个试验的真实数组与渲染图保存在其摄影产物目录，失败试验记录原因。[Optuna 多目标优化](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)

| 参数 | 作用位置 |
| --- | --- |
| `denoise_strength` | 噪声模型相对权重及时间降噪强度 |
| `motion_threshold` | 可比较区域的运动判定阈值 |
| `fusion_weight` | 参考帧有效时其他帧的相对贡献 |
| `exposure_weight` | 对黑白端点测量的降权强度 |
| `tone_mapping_strength` | 高光肩部曲线压缩强度 |

Pareto 使用四个最小化目标。噪声为平滑静态暗部的亮度误差标准差，细节为强边缘的归一化 Laplacian 误差，鬼影为合成移动主体扫掠区域的线性 RGB RMSE，颜色为与固定 Ground Truth 色调渲染的 CIE76 ΔE。颜色指标含明度差异。运动区域为空或支持不足时，该目标在整项研究中停用。当前优化入口需要配准和辐射尺度已知的 P2 Ground Truth，结果适用于所用 Burst。

报告标明每个 Pareto 试验和产物 ID。代表方案从前沿中选取，最小化各目标相对基线变化的等权和。HTML 将高维前沿投影到前两个目标，悬停可读全部目标。`trials.csv` 列出所有试验和失败记录，`metadata.json` 保存目标掩码样本数、范围、对照、参数和数据库位置，`preview.png` 为选中方案。

## 实测对照

同一组合成 Burst 上，HDR 研究完成 25 次试验，得到 12 个 Pareto 点。选中的第 18 号试验在鬼影、细节和色差上改善，暗部噪声代理略升。

| 最小化目标 | 基线 | 第 18 号试验 |
| --- | ---: | ---: |
| 静态暗部噪声残差 STD | 0.009787 | 0.009819 |
| 细节误差 | 0.551663 | 0.551149 |
| 鬼影区域 RMSE | 0.045461 | 0.033430 |
| CIE76 ΔE | 3.5849 | 3.3784 |

Motion-aware 研究完成 19 次试验，得到 7 个 Pareto 点。等权规则保留了基线。
