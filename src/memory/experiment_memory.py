"""Persist evaluated workflow passes, Optuna trials and verified repair history."""

import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    group_id: str
    number: int = Field(ge=0)
    kind: Literal["workflow", "optimization"]
    mode: Literal["night", "hdr", "motion"]
    source_kind: str
    status: Literal["passed", "needs_review", "complete", "failed"]
    input_artifact_id: str
    output_artifact_id: str | None
    seed: int | None = None
    runtime_ms: float = Field(ge=0)
    survey: dict = Field(default_factory=dict)
    scene: dict = Field(default_factory=dict)
    capture_plan: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    configuration: dict = Field(default_factory=dict)
    diagnoses: list[dict] = Field(default_factory=list)
    error: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    group_id TEXT NOT NULL, number INTEGER NOT NULL, kind TEXT NOT NULL, mode TEXT NOT NULL,
    source_kind TEXT NOT NULL, status TEXT NOT NULL, input_artifact_id TEXT NOT NULL,
    output_artifact_id TEXT, runtime_ms REAL NOT NULL, payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (group_id, number), UNIQUE (output_artifact_id)
);
CREATE INDEX IF NOT EXISTS experiment_modes ON experiments(mode, kind, status, source_kind, created_at);
CREATE INDEX IF NOT EXISTS experiment_inputs ON experiments(input_artifact_id, created_at);
CREATE TABLE IF NOT EXISTS repairs (
    group_id TEXT NOT NULL, before_number INTEGER NOT NULL, after_number INTEGER NOT NULL,
    fault TEXT NOT NULL, before_value REAL NOT NULL, after_value REAL NOT NULL,
    PRIMARY KEY (group_id, after_number, fault),
    FOREIGN KEY (group_id, before_number) REFERENCES experiments(group_id, number),
    FOREIGN KEY (group_id, after_number) REFERENCES experiments(group_id, number)
);
CREATE INDEX IF NOT EXISTS repair_faults ON repairs(fault);
"""


class ExperimentMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.path, timeout=15)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            yield db

    def record(self, experiment: Experiment) -> bool:
        data = experiment.model_dump(mode="json")
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
        keys = ("group_id", "number", "kind", "mode", "source_kind", "status", "input_artifact_id",
                "output_artifact_id", "runtime_ms")
        with self._connect() as db:
            cursor = db.execute("""INSERT INTO experiments (group_id, number, kind, mode, source_kind, status,
                       input_artifact_id, output_artifact_id, runtime_ms, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(group_id, number) DO NOTHING""", (*[data[name] for name in keys], payload))
            return cursor.rowcount == 1

    def set_diagnoses(self, group_id: str, number: int, diagnoses: list[dict]) -> None:
        with self._connect() as db:
            db.execute("""UPDATE experiments SET payload = json_set(payload, '$.diagnoses', json(?))
                       WHERE group_id = ? AND number = ?""",
                       (json.dumps(diagnoses, ensure_ascii=False, allow_nan=False), group_id, number))

    def record_repair(self, group_id: str, before_number: int, after_number: int,
                      fault: str, before_value: float, after_value: float) -> None:
        if not 0 <= after_value < before_value:
            raise ValueError("verified repair requires a decreased, valid measured value")
        with self._connect() as db:
            db.execute("""INSERT INTO repairs (group_id, before_number, after_number, fault, before_value, after_value)
                       VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(group_id, after_number, fault) DO NOTHING""",
                       (group_id, before_number, after_number, fault, before_value, after_value))

    def history(self, *, mode=None, kind=None, status=None, source_kind=None, group_id=None,
                artifact_id=None, limit=30) -> list[dict]:
        values = {"mode": mode, "kind": kind, "status": status, "source_kind": source_kind, "group_id": group_id}
        clauses = [f"{key} = ?" for key, value in values.items() if value is not None]
        arguments = [value for value in values.values() if value is not None]
        if artifact_id is not None:
            clauses.append("(input_artifact_id = ? OR output_artifact_id = ?)")
            arguments.extend([artifact_id, artifact_id])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as db:
            rows = db.execute("SELECT payload, created_at FROM experiments" + where +
                              " ORDER BY created_at DESC, rowid DESC LIMIT ?", (*arguments, max(1, min(limit, 2000))))
            return [json.loads(row["payload"]) | {"created_at": row["created_at"]} for row in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as db:
            return {row["kind"]: row["count"] for row in db.execute("SELECT kind, COUNT(*) AS count FROM experiments GROUP BY kind")}

    def repairs(self, *, mode: str, fault: str, source_kind: str, limit: int = 200) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("""SELECT r.fault, r.before_number, r.before_value, r.after_value, e.payload
                              FROM repairs r JOIN experiments e ON e.group_id = r.group_id AND e.number = r.after_number
                              WHERE r.fault = ? AND e.mode = ? AND e.source_kind = ?
                              ORDER BY e.created_at DESC, e.rowid DESC LIMIT ?""", (fault, mode, source_kind, limit))
            return [json.loads(row["payload"]) | {"fault": row["fault"], "before_number": row["before_number"],
                    "before_value": row["before_value"], "after_value": row["after_value"]} for row in rows]
