# P3 多帧计算摄影

## 运行与输入

从项目根目录运行 `python scripts/run_photography_demo.py`，为三种模式分别生成固定种子的合成 RAW Burst 并处理。命令输出各组输入及输出 ID、分阶段耗时和 Ground Truth 对照指标。

也可用 `python scripts/run_photography.py --mode motion --burst <BURST_ID>` 处理已存在的 P2 Burst，或用 `--raw-manifest sequence.yaml` 读取本地 RAW 序列。无头 RAW 在清单里提供公共布局，路径相对于清单文件。读取 LibRaw 支持的相机容器时省略 `raw_config`。曝光单位为秒，模拟增益使用线性倍数。

```yaml
raw_config:
  width: 320
  height: 192
  bit_depth: 12
  storage_format: unpacked16
  endianness: little
  bayer_pattern: RGGB
  black_level: 64
  white_level: 4095
frames:
  - {path: frame_01.raw, exposure_s: 0.004, analog_gain: 1.0}
  - {path: frame_02.raw, exposure_s: 0.035, analog_gain: 1.0}
reference_frame: 1
```

`--reference-frame` 可覆盖清单或 CapturePlan 的零基参考索引。无显式参考时，Night 使用首帧，HDR 使用居中的有效曝光档，Motion-aware 使用最短曝光。`--parameters` 接受包含 `motion_threshold`、`flow_threshold_px`、`denoise_strength`、`tone_exposure`、`spatial_sigma`、`registration`、`deghost` 和 `repair_isolated_sites` 等字段的 YAML。剂量按秒数乘线性模拟增益计算，要求序列来自同一相机和可比较的光学与 RAW 读出设置。

## 算法

1. 在 Bayer 码值域按位置扣除黑电平，记录饱和点，归一化到线性信号，再做双线性去马赛克。插值传播受饱和采样影响的通道标记。Night 默认用同色 Bayer 邻域检测孤立异常码值，并用中位数代替。对星点等真实小亮点场景，可关闭 `repair_isolated_sites`。
2. 去马赛克后的 RGB 按曝光剂量对齐亮度标尺。ECC 在压缩后的线性亮度引导图上估计全局平移；失败时尝试 ORB 匹配的稳健平移，最终可回退到单位矩阵。矩阵、方法、可用重叠比例及有效未饱和区域的中值光度残差写入产物。配准模型覆盖平移，相机旋转、视差和滚动快门可能残留误差。[OpenCV ECC 与 Farneback API](https://docs.opencv.org/4.12.0/dc/d6b/group__video__track.html)
3. Farneback 估计全局配准后的残余位移，结合光度差、形态学过滤生成运动掩码。Motion-aware 同时检查 RGB 差异，减少等亮异色运动漏检。光流用于识别不宜融合的区域；融合在全局配准的参考网格上进行。
4. Night 使用依据信号、增益和剂量计算的噪声近似权重进行多帧降噪。HDR 进一步降低接近黑白端点的样本权重并剔除饱和通道，用有效曝光重建线性高动态范围。Motion-aware 在检测到运动的区域优先保留有效参考像素，对静态区域继续多帧降噪。无有效曝光的参考通道保留原值，元数据记录其数量。去鬼影和高光恢复依赖帧间有效重叠及可用曝光。
5. Tone Mapping 使用连续高光肩部曲线压缩动态范围并保持通道比例。PNG 加显示曲线，线性中间数据独立保存。

## 产物约定

| 产物 | 内容 |
| --- | --- |
| `linear_bayer.npy`、`demosaiced_sensor_rgb.npy` | 黑电平校正后的 Bayer 和进入配准前的线性 RGB |
| `saturated_cfa_sites.npy`、`repaired_cfa_sites.npy` | 原始饱和位置、孤立异常检测位置 |
| `aligned_rgb.npy`、`transforms.npy`、`aligned_geometry.npy` | 参考剂量线性 RGB、参考到源的 2×3 矩阵、有效几何覆盖 |
| `optical_flow.npy`、`motion_masks.npy` | 参考到全局配准后源帧的 `(dx, dy)` 和逐帧运动掩码 |
| `fusion_weights.npy`、`fusion_valid_channels.npy` | `N × H × W × 3`，逐通道归一化权重和可测性掩码 |
| `unregistered_average.npy`、`fusion_without_deghosting.npy` | 曝光归一化均值和关闭去鬼影的对照 |
| `fused_linear.npy`、`final_linear.npy`、`tone_mapped_rgb.npy` | 加权融合、可选空间降噪后线性图像、色调映射后 RGB |
| `preview.png`、`diagnostics.png` | 最终预览和参考、配准、去鬼影前后、掩码、参考权重对照 |
| `metadata.json` | 参数、输入、坐标方向、分阶段耗时、Ground Truth 对照 |

## Ground Truth 对照

比较使用参考曝光中点的清晰合成 RGB。按 P2 传感器的光电转换率、满阱值和参考剂量还原到场景线性辐亮度，在线性域计算整体 PSNR 和区域 RMSE。运动区域由合成主体的扫掠范围确定。Ground Truth 仅进入评分函数。缺少 Ground Truth 的本地文件序列会记录无效原因，空区域会返回空值。耗时包含算法、图像编码和评分，在读取输入之后、写入产物之前测量。

| 合成场景 | 单帧 PSNR | 最终 PSNR | 重点区域对照 |
| --- | ---: | ---: | --- |
| Night，8 帧低照度静态 Burst | 23.36 dB | 30.71 dB | 静态暗部 RMSE 从同样做过坏点修正的单帧 0.01422 降至 0.01051 |
| HDR，4 帧静态曝光包围 | 28.14 dB | 29.49 dB | 参考过曝区域 RMSE 从 0.17207 降至 0.13235 |
| Motion-aware，4 帧移动主体 | 29.61 dB | 29.71 dB | 运动区域 RMSE 从关闭去鬼影的 0.09435 降至 0.03401，与参考单帧相同 |

上述数字来自默认合成样例。pytest 另用已知平移验证 ECC 方向和覆盖范围，用移动纹理验证 Farneback 位移，并用已知曝光和裁剪值验证 HDR 重建。P3 未对真实相机序列开展有参考评分。
