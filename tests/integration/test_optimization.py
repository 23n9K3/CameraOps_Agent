from pathlib import Path

import optuna
import pytest

from src.camera.capture_simulator import SimulationConfig, simulate_capture
from src.computational.pipeline import PipelineConfig
from src.memory.experiment_memory import ExperimentMemory
from src.optimization import optimizer
from src.optimization.optimizer import OptimizationConfig, RANGES, optimize_burst
from src.storage.artifact_store import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]


def test_optuna_records_actual_images_metrics_pareto_and_baseline(tmp_path):
    template = SimulationConfig.from_file(ROOT / "configs/synthetic_burst.yaml")
    simulation = template.model_dump(mode="json")
    simulation["scene"].update(width=144, height=96)
    bursts = ArtifactStore(tmp_path / "burst")
    raw_id = simulate_capture(SimulationConfig.model_validate(simulation), bursts)
    report_id = optimize_burst(tmp_path, raw_id, PipelineConfig(mode="motion"), OptimizationConfig(seed=81, trials=7))
    reports = ArtifactStore(tmp_path / "optimization")
    report = reports.load_metadata(report_id)
    assert report["completed_trials"] == report["trial_count"] == 8
    assert set(name for name, objective in report["objectives"].items() if objective["active"]) == {
        "noise_residual_std", "detail_error", "ghosting_rmse", "color_delta_e"}
    assert report["selected"]["number"] in {row["number"] for row in report["pareto_front"]}
    assert (reports.path(report_id) / "pareto.html").is_file()
    assert (reports.path(report_id) / "trials.csv").is_file()
    study = optuna.load_study(study_name=report["study_name"], storage=f"sqlite:///{Path(report['storage']).as_posix()}")
    assert {trial.number for trial in study.best_trials} == {row["number"] for row in report["pareto_front"]}
    assert all(set(trial.params) == set(RANGES) for trial in study.trials)
    memory = ExperimentMemory(tmp_path / "experiments.sqlite3")
    rows = memory.history(group_id=report["study_name"])
    assert len(rows) == 8
    photos = ArtifactStore(tmp_path / "photography")
    for row in rows:
        artifact = photos.load_metadata(row["output_artifact_id"])
        assert artifact["source"]["input_artifact_id"] == raw_id
        assert row["parameters"] == artifact["pipeline"]
        assert row["metrics"]["psnr_db"] == pytest.approx(artifact["evaluation"]["scores"]["final_fusion"]["full_frame"]["psnr_db"])
        assert row["runtime_ms"] > 0 and row["seed"] == simulation["seed"]
    for key, values in report["objective_comparison"].items():
        assert values["baseline"] == report["baseline"]["metrics"][key]
        assert values["selected"] == report["selected"]["metrics"][key]


def test_invalid_trial_is_recorded_and_static_burst_has_no_motion_objective(tmp_path, monkeypatch):
    template = SimulationConfig.from_file(ROOT / "configs/synthetic_burst.yaml").model_dump(mode="json")
    template["scene"].update(width=144, height=96, velocity_px_s=0)
    raw_id = simulate_capture(SimulationConfig.model_validate(template), ArtifactStore(tmp_path / "burst"))
    original = optimizer.run_pipeline
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("fixture trial failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(optimizer, "run_pipeline", fail_once)
    report_id = optimize_burst(tmp_path, raw_id, PipelineConfig(mode="hdr"), OptimizationConfig(trials=2))
    report = ArtifactStore(tmp_path / "optimization").load_metadata(report_id)
    assert report["trial_count"] == 3 and report["completed_trials"] == 2
    assert not report["objectives"]["ghosting_rmse"]["active"]
    assert "ghosting_rmse" not in report["objective_comparison"]
    failed = report["trials"][1]
    assert failed["status"] == "failed" and failed["metrics"] == {} and failed["output_artifact_id"] is None
    assert "fixture trial failure" in failed["error"]
    assert 1 not in {entry["number"] for entry in report["pareto_front"]}
