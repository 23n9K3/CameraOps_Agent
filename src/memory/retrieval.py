"""Rank compatible histories by distance in observed scene conditions."""

from math import isfinite, log, sqrt

from .experiment_memory import ExperimentMemory

_FRACTION_SCALES = {"shadow_ratio": 0.20, "highlight_ratio": 0.12, "motion_fraction": 0.08}


def scene_distance(current: dict, historical: dict) -> float | None:
    distances = []
    for key, scale in _FRACTION_SCALES.items():
        left, right = current.get(key), historical.get(key)
        if left is not None and right is not None:
            distances.append(abs(left - right) / scale)
    for key in ("brightness_median", "luminance_p99"):
        left, right = current.get(key), historical.get(key)
        if left is not None and right is not None and left >= 0 and right >= 0:
            distances.append(abs(log((left + 0.01) / (right + 0.01))) / log(1.7))
    if len(distances) < 3 or any(not isfinite(value) or value > 1.5 for value in distances):
        return None
    return sqrt(sum(value**2 for value in distances) / len(distances))


def _rank(rows: list[dict], current: dict, feature_key: str, exclude_group: str | None) -> list[dict]:
    matches = []
    for row in rows:
        if row["group_id"] == exclude_group:
            continue
        distance = scene_distance(current, row[feature_key])
        if distance is not None and distance <= 0.85:
            matches.append(row | {"scene_distance": distance})
    return sorted(matches, key=lambda row: row["scene_distance"])


def similar_strategies(memory: ExperimentMemory, current: dict, *, mode: str, source_kind: str,
                       exclude_group: str | None = None, limit: int = 5) -> list[dict]:
    rows = memory.history(mode=mode, kind="workflow", status="passed", source_kind=source_kind, limit=200)
    return _rank(rows, current, "survey", exclude_group)[:limit]


def similar_repairs(memory: ExperimentMemory, current: dict, *, mode: str, fault: str, source_kind: str,
                    exclude_group: str | None = None, limit: int = 5) -> list[dict]:
    rows = memory.repairs(mode=mode, fault=fault, source_kind=source_kind)
    return _rank(rows, current, "scene", exclude_group)[:limit]
