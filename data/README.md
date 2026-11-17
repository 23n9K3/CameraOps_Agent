# RAW 数据

- `data/raw/` 存放本地相机 RAW 文件或带配置的无头 RAW。
- `data/synthetic/` 存放 `python scripts/generate_raw_example.py` 生成的确定性 RAW12 测试图。
- `artifacts/burst/` 存放模拟脚本和查看器生成的 Burst、Ground Truth 与元数据。配置保存在每组产物的 `metadata.json` 中。
- 数据和产物按本地文件管理。相机测试样例 `iss030e122639.NEF` 来自 [rawpy 测试数据](https://github.com/letmaik/rawpy/blob/main/test/iss030e122639.NEF)。
