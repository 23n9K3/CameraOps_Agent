"""Export traceable comparison tables and publication-ready benchmark figures."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .benchmark import POLICIES

SHORT = ("Fixed", "Single", "Multi", "Feedback", "Memory")
COLORS = ("#636363", "#0072B2", "#D55E00", "#009E73", "#CC79A7")
MEASURES = ("psnr", "ssim", "noise_std", "ghost_rmse")


def _average(values):
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None, len(valid)


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
    summary = []
    fixed = {row["case"]["key"]: row.get("measurements", {}).get("psnr") for row in groups["fixed"]}
    for policy in POLICIES:
        group = groups[policy]
        inject = [item for item in group if item["case"]["fault"]]
        datum = {"policy": policy, "cases": len(group), "successes": sum(row["success"] for row in group),
                 "tool_correct": sum(row["tool_correct"] for row in group), "target_passes": sum(row["goal_pass"] for row in group),
                 "execution_errors": sum(row["execution"]["status"] == "failed" for row in group),
                 "labeled_faults": len(inject), "detected_labeled_faults": sum(row["case"]["fault"] in row.get("confirmed_faults", []) for row in inject),
                 "validated_repairs": sum(row["success"] and bool((row.get("repair") or {}).get("passed")) for row in inject),
                 "memory_hits": sum(any(review["candidates"] for review in row["execution"].get("memory_reviews", [])) for row in group),
                 "memory_applied": sum(any(review["applied"] for review in row["execution"].get("memory_reviews", [])) for row in group),
                 "grounded_metrics": sum(row.get("grounded", 0) for row in group),
                 "valid_metrics": sum(row.get("valid_metrics", 0) for row in group),
                 "mean_iterations": mean(max(0, len(row["execution"].get("attempts", [])) - 1) for row in group),
                 "wall_ms_mean": mean(row["execution"]["wall_ms"] for row in group),
                 "wall_ms_median": median(row["execution"]["wall_ms"] for row in group)}
        for name in MEASURES:
            datum[name], datum[f"{name}_n"] = _average(row.get("measurements", {}).get(name) for row in group)
        deltas = [current - fixed[row["case"]["key"]] for row in group
                  if (current := row.get("measurements", {}).get("psnr")) is not None and fixed.get(row["case"]["key"]) is not None]
        datum["psnr_delta_vs_fixed_db"], datum["psnr_delta_n"] = _average(deltas)
        summary.append(datum)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plots(root: Path, rows: list[dict], summary: list[dict]) -> None:
    cases = list(dict.fromkeys(row["case"]["key"] for row in rows))
    table = {(row["case"]["key"], row["policy"]): row.get("measurements", {}) for row in rows}
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True, layout="constrained")
    for axis, metric, unit in zip(axes, ("psnr", "ssim"), ("PSNR (dB)", "SSIM")):
        for style, ((policy, _), label, color) in enumerate(zip(POLICIES.items(), SHORT, COLORS)):
            axis.plot(range(len(cases)), [table[(case, policy)].get(metric, np.nan) for case in cases], color=color,
                      linestyle=("-", "--", ":", "-.", "-")[style], marker=("o", "s", "^", "x", "+")[style],
                      linewidth=1.25, markersize=3.6, label=label, alpha=0.87)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.27), frameon=False)
    axes[1].set_xticks(range(len(cases)), cases, rotation=55, ha="right", fontsize=8)
    fig.suptitle("Paired RAW benchmark · fixed reference and calibrated Ground Truth", fontsize=14)
    fig.savefig(root / "psnr_ssim.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), layout="constrained")
    panels = (("noise_std", "Static shadow noise STD"), ("ghost_rmse", "Moving-region RMSE"),
              ("wall_ms_median", "Median runtime (ms)"), ("successes", "Task success (%)"),
              ("tool_correct", "Tool choice accuracy (%)"), ("validated_repairs", "Verified repairs (count)"))
    for axis, (metric, title) in zip(axes.ravel(), panels):
        values = [r[metric] for r in summary]
        if metric in ("successes", "tool_correct"):
            values = [100 * value / row["cases"] for value, row in zip(values, summary)]
            axis.set_ylim(0, 112)
        bars = axis.bar(SHORT, [value if value is not None else 0 for value in values], color=COLORS)
        axis.bar_label(bars, labels=["NA" if v is None else f"{v:.4g}" for v in values], padding=3, fontsize=8)
        axis.margins(y=0.20)
        axis.set_title(title, loc="left", fontsize=10, fontweight="semibold")
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    fig.suptitle("Architecture comparison · benchmark aggregates", fontsize=14)
    fig.savefig(root / "noise_ghosting_runtime.png", dpi=180)
    plt.close(fig)


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(root: Path, rows: list[dict], summary: list[dict], training: list[dict]) -> None:
    lines = ["# AI-CPL P9 工程验收与消融", "", f"本次运行包含 {len(rows)} 个配对任务结果和 {len(training)} 个独立记忆预热样本。",
             "任务成功要求输出、目标流水线和独立的场景门槛均通过。统计定义见项目的 `docs/experiment_protocol.md`。", "",
             "| 方案 | 任务成功 | 工具选择正确 | PSNR dB | SSIM | Noise STD (n) | Ghost RMSE (n) | 耗时中位 ms | 标签故障命中 | 验证修复 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f'| {POLICIES[row["policy"]]} | {row["successes"]}/{row["cases"]} | {row["tool_correct"]}/{row["cases"]} | '
                     f'{_fmt(row["psnr"], 2)} | {_fmt(row["ssim"])} | {_fmt(row["noise_std"], 5)} ({row["noise_std_n"]}) | '
                     f'{_fmt(row["ghost_rmse"], 5)} ({row["ghost_rmse_n"]}) | {_fmt(row["wall_ms_median"], 1)} | '
                     f'{row["detected_labeled_faults"]}/{row["labeled_faults"]} | {row["validated_repairs"]} |')
    lines += ["", "![逐任务 PSNR 和 SSIM](psnr_ssim.png)", "", "![Noise、Ghosting 与运行表现](noise_ghosting_runtime.png)", "",
              "## 分场景成功数", "", "| 场景 | " + " | ".join(SHORT) + " |", "|---|---:|---:|---:|---:|---:|"]
    for family in dict.fromkeys(row["case"]["family"] for row in rows):
        cells = []
        for policy in POLICIES:
            matches = [row for row in rows if row["policy"] == policy and row["case"]["family"] == family]
            cells.append(f"{sum(row['success'] for row in matches)}/{len(matches)}")
        lines.append(f"| {family} | " + " | ".join(cells) + " |")
    keyed = {row["policy"]: row for row in summary}
    lines += ["", "## 消融", ""]
    for name, left, right in (("意图选择", "fixed", "single"), ("多节点分工", "single", "multi"),
                              ("调试反馈", "multi", "feedback"), ("历史记忆", "feedback", "memory")):
        a, b = keyed[left], keyed[right]
        lines.append(f'- {name}的配对对照为 {POLICIES[left]} 到 {POLICIES[right]}，成功数 {a["successes"]} → {b["successes"]}，'
                     f'工具选择正确数 {a["tool_correct"]} → {b["tool_correct"]}。')
    lines += [f'- 记忆方案在 {keyed["memory"]["memory_hits"]} 个任务检索到历史，'
              f'在 {keyed["memory"]["memory_applied"]} 个任务采纳经当前证据调整的策略。',
              "- `capture_ablation.csv` 记录三类场景上的固定、亮度规则和 Agent 采集计划及其实际模拟量。",
              "- `photography_ablation.csv` 记录单 RAW、曝光归一化平均、配准融合及启用去鬼影的实际评分。",
              "- 固定流水线对所有场景启用 Night 的坏点修复。跨场景平均噪声受这一配置差异影响，需结合逐任务表、场景目标与高光和鬼影误差阅读。",
              "", "## 可追溯性与边界", "", "逐任务指标与产物号见 `cases.csv`，完整工具结果、诊断和错误见 `cases.jsonl`。"
              "记忆训练来自独立种子。本研究使用合成输入，运行时间使用本地进程内工具调用，生产环境的子进程超时与重试由集成测试覆盖。",
              "", "[验收清单与未完成功能](../../../docs/engineering_acceptance.md)", ""]
    (root / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
