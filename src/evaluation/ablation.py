"""Supplement architecture comparisons with actual capture and image-stage ablations."""

from pathlib import Path

from src.agents.capture_planner_agent import plan_capture
from src.agents.supervisor import interpret_goal
from src.camera.capture_plan import CapturePlan
from src.storage.artifact_store import ArtifactStore
from src.tools.contracts import CaptureInput, SourceSpec, SurveyInput
from src.tools.operations import capture, survey

from .cases import FAMILIES, simulation_for


def capture_ablation(root: Path) -> list[dict]:
    rows = []
    for index, family in enumerate(("night", "hdr", "motion")):
        template = simulation_for(family, 0, 77001 + index)
        features = survey(SurveyInput(artifact_root=root, source=SourceSpec(simulation=template))).features
        fixed = CapturePlan(exposure_times=[0.012] * 4, gains=[1] * 4)
        exposure = max(0.004, min(0.030, 0.020 * 0.68 / max(features.luminance_p99, 1e-4)))
        rule = CapturePlan(exposure_times=[exposure] * 4, gains=[1] * 4)
        agent = plan_capture(interpret_goal(FAMILIES[family][1]), features, simulated=True,
                             gap=template.capture_plan.inter_frame_gap_s)
        for label, plan in (("fixed", fixed), ("brightness_rule", rule), ("agent_planner", agent)):
            artifact = capture(CaptureInput(artifact_root=root, simulation=template, plan=plan)).artifact_id
            metadata = ArtifactStore(root / "burst").load_metadata(artifact)
            reference = metadata["frames"][plan.reference_frame]
            rows.append({"scene": family, "policy": label, "seed": template.seed, "artifact_id": artifact,
                         "capture_plan": plan.model_dump(mode="json"), "reference_saturation": reference["saturated_site_ratio"],
                         "reference_mean_dn": reference["mean_raw_dn"],
                         "reference_motion_px": reference["subject_displacement_during_exposure_px"],
                         "runtime_ms": metadata["simulation_and_preview_ms"]})
    return rows


def photography_ablation(root: Path, rows: list[dict]) -> list[dict]:
    photos = ArtifactStore(root / "photography")
    results = []
    for row in rows:
        artifact = row["execution"].get("artifact_id")
        if row["policy"] != "feedback" or not artifact:
            continue
        scores = photos.load_metadata(artifact)["evaluation"]["scores"]
        for name, score in scores.items():
            results.append({"case": row["case"]["key"], "artifact_id": artifact, "stage": name,
                            "psnr_db": score["full_frame"]["psnr_db"], "shadow_rmse": score["static_shadows"]["rmse"],
                            "ghost_rmse": score["motion_region"]["rmse"], "highlight_rmse": score["reference_clipped_highlights"]["rmse"]})
    return results
