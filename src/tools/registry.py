"""Run allowlisted, schema-validated tools with bounded retries and process timeouts."""

import errno
import os
import subprocess
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from pydantic import BaseModel, ValidationError

from src.debugging.diagnosis import Diagnosis
from src.iq.evaluation import IQResult
from src.scene.analysis import SceneFeatures

from . import operations
from .contracts import (ArtifactInput, ArtifactOutput, CaptureInput, PhotographyInput,
                        SurveyInput, SurveyOutput, ValidationInput)


@dataclass(frozen=True)
class Tool:
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    execute: Callable[[BaseModel], BaseModel]


REGISTRY = {
    "survey_capture_conditions": Tool(SurveyInput, SurveyOutput, operations.survey),
    "simulate_capture": Tool(CaptureInput, ArtifactOutput, operations.capture),
    "analyze_scene_motion": Tool(ArtifactInput, SceneFeatures, operations.inspect_burst),
    "run_computational_pipeline": Tool(PhotographyInput, ArtifactOutput, operations.run_photography),
    "evaluate_final_image": Tool(ArtifactInput, IQResult, operations.evaluate),
    "validate_diagnosis": Tool(ValidationInput, Diagnosis, operations.validate),
}


class ToolFailure(RuntimeError):
    """A registered image operation exhausted its allowed attempts."""


class ToolRunner:
    def __init__(self, timeout: float, retries: int, *, on_event: Callable[[dict], None] | None = None):
        self.timeout = timeout
        self.retries = retries
        self.on_event = on_event
        self.calls: list[dict] = []

    def _finish(self, call: dict, result: BaseModel | None = None) -> None:
        self.calls.append(call)
        if self.on_event is not None:
            self.on_event({"type": "tool_finished", **call,
                           "result": result.model_dump(mode="json") if result is not None else None})

    def call(self, name: str, arguments: BaseModel) -> BaseModel:
        spec = REGISTRY.get(name)
        if spec is None:
            raise ToolFailure(f"unregistered tool: {name}")
        try:
            payload = spec.input_schema.model_validate(arguments.model_dump()).model_dump_json()
        except ValidationError as exc:
            self._finish({"tool": name, "attempt": 0, "status": "invalid_input", "error": str(exc),
                          "artifact_id": getattr(arguments, "artifact_id", None), "runtime_ms": 0.0})
            raise ToolFailure(f"{name}: invalid input") from exc
        input_id = getattr(arguments, "artifact_id", None)
        if isinstance(arguments, SurveyInput) and arguments.source.burst_id:
            input_id = arguments.source.burst_id
        env = {**os.environ, "PYTHONUTF8": "1", "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}
        for attempt in range(1, self.retries + 2):
            started = perf_counter()
            if self.on_event is not None:
                self.on_event({"type": "tool_started", "tool": name, "attempt": attempt, "status": "running",
                               "input_artifact_id": input_id})
            retryable = False
            try:
                completed = subprocess.run([sys.executable, "-m", "src.tools.registry", name], input=payload,
                                           capture_output=True, encoding="utf-8", errors="replace", env=env,
                                           timeout=self.timeout)
                if completed.returncode == 0:
                    result = spec.output_schema.model_validate_json(completed.stdout)
                    output_id = getattr(result, "artifact_id", None) or getattr(result, "input_artifact_id", None)
                    self._finish({"tool": name, "attempt": attempt, "status": "ok", "artifact_id": output_id or input_id,
                                  "input_artifact_id": input_id, "runtime_ms": (perf_counter() - started) * 1000}, result)
                    return result
                reason = completed.stderr.strip()[-1200:] or f"worker exited with code {completed.returncode}"
                retryable = completed.returncode == 75
            except subprocess.TimeoutExpired:
                reason, retryable = f"tool exceeded its {self.timeout:g}s process timeout", True
            except (OSError, ValidationError) as exc:
                reason = str(exc)
            self._finish({"tool": name, "attempt": attempt, "status": "error", "error": reason,
                          "artifact_id": input_id, "runtime_ms": (perf_counter() - started) * 1000})
            if not retryable:
                break
        raise ToolFailure(f"{name}: {reason}")


def _worker(name: str) -> int:
    spec = REGISTRY.get(name)
    if spec is None:
        print(f"unregistered tool: {name}", file=sys.stderr)
        return 2
    try:
        arguments = spec.input_schema.model_validate_json(sys.stdin.read())
        result = spec.execute(arguments)
        print(spec.output_schema.model_validate(result).model_dump_json())
    except OSError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 75 if exc.errno in (errno.EAGAIN, errno.EBUSY, errno.ETIMEDOUT) else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker(sys.argv[1] if len(sys.argv) == 2 else ""))
