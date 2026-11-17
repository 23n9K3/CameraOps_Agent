# DeepSeek 与真实 RAW 连拍

## 配置 DeepSeek

在 Windows PowerShell 输入下面两行，密钥在提示时输入，并保存在当前 Windows 用户的环境变量中。刷新 Agent 工作台后，在「执行设置 → Agent 大脑」选择 DeepSeek。

```powershell
$deepseekCredential = [pscredential]::new('DeepSeek', (Read-Host 'DeepSeek API key' -AsSecureString))
[Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', $deepseekCredential.GetNetworkCredential().Password, 'User')
```

默认模型为 `deepseek-flash`，可通过环境变量 `DEEPSEEK_MODEL` 修改。DeepSeek 接收拍摄目标和本地测出的数值场景摘要，返回摄影模式、运动保护选项和理由。LangGraph 校验结果后调用本地摄影工具，IQ 和故障验证从落盘产物计算。原始文件和像素保存在本地。工作流报告记录所用模式和已校验的决策。[DeepSeek API](https://api-docs.deepseek.com/) · [JSON Output](https://api-docs.deepseek.com/guides/json_mode/)

## 工作台实拍场景

在 Agent 工作台用「实拍 RAW 序列」切换场景。已有完成的工作流时，切换场景后会直接显示该组最近一次的处理结果。以下四组来自 Google HDR+ Burst Photography Dataset（Hasinoff 等，CC BY-SA）。[数据来源和授权](https://hdrplusdata.org/dataset.html)

| 场景 | 数据集连拍编号 |
| --- | --- |
| 人像 · 滨海暮色 | `33TJ_20150612_205903_877` |
| 风景 · 高山湖泊 | `0543_20150918_143723_450` |
| 夜景 · 街头双人 | `33TJ_20150731_214303_007` |
| 运动场景 · 海浪拍岸 | `5066_20160413_171343_904` |

每组使用开头三张原生 DNG。示例导入时按 Bayer 相位缩小整幅画面，按相机方向旋转。工作流在传感器线性空间执行配准和融合，预览用 DNG 白平衡与同组 `rgb2rgb.txt` 矩阵转换为 sRGB。下面的命令假定三张 DNG 与颜色矩阵已放在相应的 `data/raw/hdrplus_<编号>/` 目录。

```powershell
$sceneFolder = 'data/raw/hdrplus_33TJ_20150612_205903_877'
$rawFrames = @(Get-ChildItem "$sceneFolder/payload_N00[0-2].dng" | Sort-Object Name | ForEach-Object FullName)
$burst = .\.venv\Scripts\python.exe -m scripts.import_real_burst --binning 3 --scene-label '人像 · 滨海暮色' --color-matrix "$sceneFolder/rgb2rgb.txt" @rawFrames | ConvertFrom-Json
.\.venv\Scripts\python.exe -m scripts.run_agent_workflow --brain offline_mock --burst $burst.artifact_id --goal '保留人像五官、天空色彩和背景细节'
```

## 早期中心裁剪测试

使用 Google HDR+ Burst Photography Dataset 的 `20171106_subset/bursts/0006_20160722_115157_431` 中 `payload_N000.dng` 至 `payload_N002.dng`。该组为原生相机 DNG 连拍，来源采用 CC BY-SA。把三张文件放在 `data/raw/hdrplus_0006_115157/`。[数据来源和授权](https://hdrplusdata.org/dataset.html)

```powershell
$rawFrames = @(Get-ChildItem data/raw/hdrplus_0006_115157/*.dng | Sort-Object Name | ForEach-Object FullName)
.\.venv\Scripts\python.exe scripts/import_real_burst.py --crop 768 512 @rawFrames
.\.venv\Scripts\python.exe scripts/run_agent_workflow.py --brain deepseek --burst <BURST_ID> --goal "给这组实拍 RAW 去噪，保留细节"
```

导入器读取 EXIF 快门和 ISO，以本组最低 ISO 为基准记录相对增益。相对 ISO 用作剂量比例代理。原始文件每张为 4048×3036 BGGR，EXIF 快门约 0.419 ms，ISO 51。此次处理使用同一 Bayer 相位的 768×512 中心裁剪。

离线规则工作流实际处理两轮，鬼影代理从 `0.020866` 降至 `0.0`，配准残差为 `0.001493`。鬼影代理衡量运动区域最终结果偏离参考帧的程度。此组缺少经校准的 Ground Truth，PSNR、SSIM 返回 `null`。预览沿用传感器方向与颜色，使用当前流水线的色调映射。DeepSeek 接口使用模拟 HTTP 响应验证模式选择和故障路径。在线运行需要配置真实 API 密钥。
