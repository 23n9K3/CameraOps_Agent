"""Query stored experiments or find strategies similar to a recorded scene."""

import argparse
import json
from pathlib import Path

from src.memory.experiment_memory import ExperimentMemory
from src.memory.retrieval import similar_repairs, similar_strategies
from src.memory.strategy_history import import_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("artifacts/experiments.sqlite3"))
    parser.add_argument("--mode", choices=("night", "hdr", "motion"))
    parser.add_argument("--kind", choices=("workflow", "optimization"))
    parser.add_argument("--status", choices=("passed", "needs_review", "complete", "failed"))
    parser.add_argument("--group", help="workflow task ID or Optuna study name")
    parser.add_argument("--artifact", help="input or output artifact ID")
    parser.add_argument("--similar-to", help="use this recorded photography artifact's scene as the query")
    parser.add_argument("--fault", choices=("registration_failure", "ghosting"), help="find verified repairs for --similar-to")
    parser.add_argument("--import-reports", type=Path, metavar="ARTIFACT_ROOT", help="index earlier LangGraph reports")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.limit <= 2000:
        parser.error("limit must be from 1 to 2000")
    if args.fault and not args.similar_to:
        parser.error("--fault requires --similar-to")
    if args.similar_to and any((args.mode, args.kind, args.status, args.group, args.artifact)):
        parser.error("similarity uses the reference experiment's mode and source, so omit history filters")
    memory = ExperimentMemory(args.database)
    if args.import_reports:
        if any((args.similar_to, args.mode, args.kind, args.status, args.group, args.artifact)):
            parser.error("report import is a standalone operation")
        rows = import_reports(memory, args.import_reports)
    elif args.similar_to:
        current = next((row for row in memory.history(artifact_id=args.similar_to, limit=2000)
                        if row["output_artifact_id"] == args.similar_to), None)
        if not current:
            parser.error("no experiment found for that output artifact ID")
        options = {"mode": current["mode"], "source_kind": current["source_kind"],
                   "exclude_group": current["group_id"], "limit": args.limit}
        rows = similar_repairs(memory, current["scene"], fault=args.fault, **options) if args.fault else (
            similar_strategies(memory, current["survey"], **options))
    else:
        rows = memory.history(mode=args.mode, kind=args.kind, status=args.status, group_id=args.group,
                              artifact_id=args.artifact, limit=args.limit)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
