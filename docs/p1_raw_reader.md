# P1 RAW 读取与验收

## 布局约定

| `storage_format` | `bit_depth` | 字节布局 |
| --- | --- | --- |
| `unpacked8` | 8 | 每个采样点占一个字节 |
| `unpacked16` | 8、10、12、14、16 | 每个采样点占两个字节，低位对齐，按 `endianness` 解码 |
| `mipi_packed` | 10、12、14 | RAW10 每 4 点占 5 字节，RAW12 每 2 点占 3 字节，RAW14 每 4 点占 7 字节 |

MIPI 按行分组，宽度必须包含完整采样组。组内前段为各采样点的高 8 位，剩余低位按采样顺序打包。[RAW10 字节布局](https://www.kernel.org/doc/html/v4.12/media/uapi/v4l/pixfmt-srggb10p.html) · [RAW12 字节布局](https://kernel.org/doc/html/v5.0/media/uapi/v4l/pixfmt-srggb12p.html) · [RAW14 字节布局](https://docs.kernel.org/userspace-api/media/v4l/pixfmt-srggb14p.html)

`row_stride_bytes` 包含每行尾部填充，省略时按紧密排列读取。文件长度必须等于行跨度乘以高度。厂商自定义打包、左对齐样本、额外文件头和不完整 MIPI 分组需要先转换为受支持的布局。读取器检查长度、分组和可检测的码值越界。字节布局、CFA、大小端和黑白电平应以传感器或采集工具提供的参数为准。

`black_level` 与 `white_level` 可分别指定单个整数或四个整数。四值依次对应 2×2 Bayer 单元的左上、右上、左下、右下。每个采样点按 `(code - black) / (white - black)` 归一化并裁剪到 `[0, 1]`。饱和掩码在裁剪前按 `code >= white` 计算。双线性去马赛克保留各 CFA 位置的已测样本。

LibRaw 路径使用可视区域起点确定 CFA 相位，并读取解码器提供的黑白电平。元数据中的 `bit_depth: 16` 表示解码数组使用 `uint16` 码值域，实际有效白点由 `white_level` 表示。2×2 Bayer 以外的相机容器版式报错。本阶段用 Nikon NEF 验证了容器路径；DNG 和其他机型的兼容性需用相应样本验收。[rawpy 可视区域及电平接口](https://github.com/letmaik/rawpy/blob/main/rawpy/_rawpy.pyx)

## 产物

同一帧的 `raw_mosaic.npy`、`linear_mosaic.npy`、`linear_rgb.npy`、`saturation_mask.npy`、`preview.png` 和 `metadata.json` 通过同一个产物 ID 发布。后续算法可用 `ArtifactStore.load_array(artifact_id, "linear_rgb")` 读取浮点线性数据。PNG 预览对传感器 RGB 应用显示传递曲线，画面色彩未经白平衡与色彩矩阵校准。

## 运行记录

- Windows、Python 3.10.9、NumPy 2.2.6、OpenCV 4.14.0.94、rawpy 0.27.1。
- `python -m pytest -q` 通过 34 项测试。覆盖有效位深、字节序、行跨度、三种 MIPI 已知向量、四种 Bayer 排列、黑白电平、错误输入和命令行产物回读。
- 确定性合成 RAW12 测试图实际生成了 320×192 预览及线性数据，产物 ID 为 `c16d383e5b7a432ca107d264b28e426a`。
- 真实样本 `iss030e122639.NEF` 实际解码为 4284×2844 RGGB，产物 ID 为 `5410ed1c7b374b189e7793704d2ec9c5`。样本 SHA-256 为 `5922721d13f11795557d97fdebd0a60b900086c402bc82a848ff280d15b99ffd4`。[NEF 样本来源](https://github.com/letmaik/rawpy/blob/main/test/iss030e122639.NEF)
- 对 `M0054341_01_00005.cr2` 的试读触发 LibRaw `LibRawDataError`。直接调用 rawpy 复现了相同错误。

## 后续阶段

P2 将引入带曝光、增益、噪声和运动参数的传感器仿真与 RAW Burst。P1 的确定性测试图用于验证读码和颜色通道，无随机噪声或采集仿真。P3 至 P9 的算法、Agent、界面及基准进展将在各阶段分别记录。
