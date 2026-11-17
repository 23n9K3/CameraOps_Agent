# P2 离线采集模拟

## 复现

从项目根目录运行 `python scripts/run_capture_simulation.py --config configs/synthetic_burst.yaml`。CLI 输出完整元数据，并把产物存到 `artifacts/burst/`。启动 `python -m streamlit run app.py` 后可选择该产物，也可在侧栏编辑 YAML 并重新模拟。配置中的 `capture_plan.exposure_times` 单位为秒，`gains` 是线性模拟增益倍率，两列按帧一一对应；`reference_frame` 从 0 开始。

## 成像模型

- 场景含低亮度区域、彩色物体、纹理、过亮窗口和移动主体。每帧使用起止时间内均匀采样的子时刻渲染并平均线性 RGB，再按 Bayer 排列取样。曝光时长改变积累电子数，也改变曝光积分中的主体运动长度。
- 模拟电子数由辐亮度、光电转换率、曝光、PRNU、DSNU 和热像素暗电流决定。死像素的光电子信号归零，读出噪声仍可使其码值在黑电平附近波动。Shot Noise 采用泊松采样，在满阱裁剪后加入高斯读出、行及列噪声。
- 读出结果乘以模拟增益并映射至 RAW 码值。ADC 添加黑电平后执行裁剪和四舍五入。Burst RAW 使用小端、右对齐的 16 位容器；实际位深、CFA 和黑白电平记录在 `raw_config`。
- SeedSequence 为传感器固定图样和各帧噪声分配独立随机流。在同一软件环境中，相同配置和种子复现数组。产物 ID 和耗时随运行分配及测量。
- 该模型使用全局快门、线性传感器响应、解析场景和有限时间采样。Ground Truth 为场景辐亮度，取值可以大于 1；与曝光相关的 RAW 码值比较时需先匹配时间、曝光和辐射尺度。

## 数据约定

| 文件 | 顺序和含义 |
| --- | --- |
| `raw_burst.npy` | `N × H × W`，带噪声和曝光效应的 `uint16` Bayer 码值 |
| `ground_truth_rgb.npy` | `N × H × W × 3`，曝光中点的无噪、无曝光内模糊的 `float32` RGB |
| `exposure_average_rgb.npy` | `N × H × W × 3`，曝光内积分平均的无噪 `float32` RGB，保留运动模糊 |
| `subject_mask.npy` | `N × H × W`，曝光中点的主体像素 |
| `hot_pixel_mask.npy`、`dead_pixel_mask.npy` | `H × W`，Burst 共享的缺陷位置 |
| `prnu_map.npy`、`dsnu_e_s_map.npy` | `H × W`，相对响应倍率与每秒暗电子率偏移 |
| `preview.png` | 由实际模拟码值解马赛克并加显示曲线生成的序列图 |
| `metadata.json` | 配置、种子、逐帧时间线、曝光、增益、饱和值比例、阵列描述 |

查看器可导出任意帧的无头 `.raw` 和匹配的 `raw_layout.json`；两者可直接交给 P1 RAW Reader。直方图基于原始 Bayer 码值并标注黑白电平。Ground Truth 的显示预览应用显示曲线，`.npy` 保留原始辐亮度。

## 验收

默认配置包含 4 帧 320×192 RAW12 RGGB、种子 `20261116`，在同一传感器上生成了 61 个热像素和 61 个死像素。末帧为 45 ms、1×，约 2.8825% 采样点达到白电平；曝光内主体移动 24.3 px。

pytest 验证了 CapturePlan 参数及时间线、曝光和增益对实际码值的独立影响、噪声统计、缺陷位置、可复现性、运动 Ground Truth、P1 RAW 回读、查看器生成与切帧、RAW 上传及直方图呈现。复杂 Agent 和后续计算摄影流水线留待后续阶段。
