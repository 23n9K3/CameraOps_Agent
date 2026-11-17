"""Deterministic natural-language intent parsing and evidence-based routing."""

from typing import Literal

from src.debugging.diagnosis import QualityLimits, detected_faults
from src.iq.evaluation import IQResult

Mode = Literal["night", "hdr", "motion"]


def interpret_goal(goal: str) -> dict:
    text = goal.casefold()
    return {
        "hdr": any(word in text for word in ("hdr", "高动态", "高光", "曝光包围", "highlights", "bracket")),
        "night": any(word in text for word in ("夜", "降噪", "低照", "暗部", "night", "low-light", "denoise", "noise")),
        "motion": any(word in text for word in ("运动", "鬼影", "行人", "拖影", "motion", "ghost", "moving", "blur")),
    }


def assess_result(iq: IQResult, limits: QualityLimits) -> dict:
    faults = detected_faults(iq, limits)
    measured = iq.metrics
    # Rank by measured failure count and severity, using reference PSNR to break ties when available.
    registration_score = max([
        0.0, *[max((row["alignment_residual"] or 0.0) / limits.max_alignment_error,
                    max(1 - row["overlap_ratio"], 0) / max(1 - limits.min_overlap_ratio, 1e-6))
                for row in iq.registration_checks]
    ])
    ghost_score = measured["ghosting_score"].value
    severity = sum((registration_score if "registration_failure" in faults else 0,
                    (ghost_score or 0) / limits.max_ghosting_score if "ghosting" in faults else 0))
    return {"faults": faults, "rank": [len(faults), severity, -(measured["psnr"].value or 0)]}
