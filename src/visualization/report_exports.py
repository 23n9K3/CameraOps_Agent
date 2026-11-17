"""Export measured workflow and optimization report fields."""

import csv
import io
import json
from html import escape


def metric_rows(report: dict) -> list[dict]:
    rows = []
    if report["artifact_type"] == "camera_workflow_report":
        for attempt in report.get("attempts", []):
            for name, metric in attempt["iq"]["metrics"].items():
                rows.append({"轮次": attempt["iteration"], "产物": attempt["artifact_id"], "指标": name,
                             "数值": metric["value"], "单位": metric["unit"], "有效": metric["valid"]})
    else:
        for label in ("baseline", "selected"):
            trial = report[label]
            rows.extend({"轮次": label, "产物": trial["output_artifact_id"], "指标": name,
                         "数值": value, "单位": "", "有效": value is not None}
                        for name, value in trial["metrics"].items())
    return rows


def export_csv(rows: list[dict]) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=["轮次", "产物", "指标", "数值", "单位", "有效"])
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8-sig")


def export_html(report: dict, rows: list[dict]) -> bytes:
    title = escape(report.get("user_request") or report.get("study_name") or "AI-CPL")
    details = escape(json.dumps(report, ensure_ascii=False, indent=2))
    header = "".join(f"<th>{head}</th>" for head in ("轮次", "产物", "指标", "数值", "单位", "有效"))
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value)) if value is not None else ''}</td>"
                                    for value in row.values()) + "</tr>" for row in rows)
    return (f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>AI-CPL · {title}</title>'
            '<style>body{font:16px system-ui;margin:3rem auto;max-width:76rem;padding:0 1rem;color:#172D29}'
            'table{width:100%;border-collapse:collapse}td,th{padding:.5rem;text-align:left;border-bottom:1px solid #CFD9D0}'
            'pre{overflow:auto;background:#EDEFE8;padding:1rem}summary{cursor:pointer}</style>'
            f'<main><h1>{title}</h1><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'
            f'<details><summary>完整实验记录</summary><pre>{details}</pre></details></main></html>').encode("utf-8")
