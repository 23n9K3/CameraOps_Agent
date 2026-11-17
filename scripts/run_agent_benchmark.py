"""Run the complete paired AI-CPL engineering benchmark and write evidence locally."""

import argparse
import json
import platform
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from src.evaluation.ablation import capture_ablation, photography_ablation
from src.evaluation.benchmark import POLICIES, run_case
from src.evaluation.cases import FAMILIES, make_case
from src.evaluation.reporting import markdown, plots, summarize, write_csv
from src.memory.experiment_memory import ExperimentMemory
from src.storage.artifact_store import ArtifactStore


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--seed", type=int, default=46001)
    parser.add_argument("--variants", type=int, choices=(1, 2, 3), default=3, help="seeded variations per scenario family")
    args = parser.parse_args()
    if not 0 <= args.seed <= (2**32 - 20000):
        parser.error("seed is outside the supported benchmark seed range")
    root = (args.artifacts / uuid4().hex).resolve()
    root.mkdir(parents=True)
    # Keep thread count and measured backend identical across the five methods.
    cv2.setNumThreads(1)
    versions = {"python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "opencv": cv2.__version__}
    manifest = {"status": "running", "seed": args.seed, "variants": args.variants,
                "families": list(FAMILIES), "policies": POLICIES, "versions": versions,
                "tool_backend": "registered implementations in process", "opencv_threads": 1}
    write_json(root / "manifest.json", manifest)
    print(f"Benchmark output: {root}", flush=True)

    training_root = root / "training"
    training_inputs = ArtifactStore(training_root / "burst")
    training = [run_case(make_case(training_inputs, "ghosting", v, args.seed + 10000 + v, training=True),
                         "feedback", training_root) for v in range(3)]
    write_json(root / "memory_training.json", training)
    history = ExperimentMemory(training_root / "experiments.sqlite3")
    sources = ArtifactStore(root / "burst")
    cases = [make_case(sources, family, variant, args.seed + index * 10 + variant)
             for index, family in enumerate(FAMILIES) for variant in range(args.variants)]
    rows = []
    policies = list(POLICIES)
    with (root / "cases.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for index, case in enumerate(cases):
            offset = index % len(policies)
            for policy in policies[offset:] + policies[:offset]:
                result = run_case(case, policy, root, history)
                rows.append(result)
                log.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"[{index + 1}/{len(cases)}] {case.key}", flush=True)

    comparison = summarize(rows)
    write_csv(root / "comparison.csv", comparison)
    write_csv(root / "cases.csv", [
        {"case": row["case"]["key"], "family": row["case"]["family"], "seed": row["case"]["seed"],
         "fault": row["case"]["fault"], "policy": row["policy"], "input_artifact_id": row["case"]["input_id"],
         "output_artifact_id": row["execution"].get("artifact_id"), "report_id": row["execution"].get("report_id"),
         "expected_mode": row["case"]["expected_mode"], "actual_mode": row.get("mode"), "success": row["success"],
         "target": row.get("target"), "target_pass": row["goal_pass"], "tool_correct": row["tool_correct"],
         "status": row["execution"]["status"], "wall_ms": row["execution"]["wall_ms"],
         "translation_error_px": row.get("translation_error_px"), "repair_pass": (row.get("repair") or {}).get("passed"),
         **row.get("measurements", {})} for row in rows])
    capture_results = capture_ablation(root)
    write_csv(root / "capture_ablation.csv", [{**row, "capture_plan": json.dumps(row["capture_plan"])} for row in capture_results])
    write_csv(root / "photography_ablation.csv", photography_ablation(root, rows))
    plots(root, rows, comparison)
    markdown(root, rows, comparison, training)
    failures = sum(row["execution"]["status"] == "failed" for row in rows)
    write_json(root / "manifest.json", manifest | {"status": "completed_with_execution_errors" if failures else "completed",
               "case_count": len(cases), "result_count": len(rows), "execution_errors": failures,
               "reports": ["comparison.csv", "cases.csv", "cases.jsonl", "ablation_report.md", "psnr_ssim.png",
                           "noise_ghosting_runtime.png", "capture_ablation.csv", "photography_ablation.csv", "memory_training.json"],
               "summary": comparison})
    print(f"Report: {root / 'ablation_report.md'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
