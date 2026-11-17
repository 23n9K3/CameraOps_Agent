"""Run UI jobs independently from Streamlit reruns and collect execution events."""

from threading import Lock, Thread
from time import perf_counter
from typing import Callable


class Job:
    def __init__(self, target: Callable):
        self._started = perf_counter()
        self._lock = Lock()
        self._events: list[dict] = []
        self._finished: float | None = None
        self._result: dict | None = None
        self._error: str | None = None
        self._thread = Thread(target=self._run, args=(target,), daemon=True, name="aicpl-workbench")
        self._thread.start()

    def _emit(self, event: dict) -> None:
        with self._lock:
            self._events.append({**event, "elapsed_ms": (perf_counter() - self._started) * 1000})

    def _run(self, target: Callable) -> None:
        result, error = None, None
        try:
            result = target(self._emit)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._result, self._error = result, error
                self._finished = perf_counter()

    @property
    def done(self) -> bool:
        with self._lock:
            return self._finished is not None

    def snapshot(self) -> dict:
        with self._lock:
            return {"done": self._finished is not None, "result": self._result, "error": self._error,
                    "elapsed_ms": ((self._finished or perf_counter()) - self._started) * 1000,
                    "events": list(self._events)}
