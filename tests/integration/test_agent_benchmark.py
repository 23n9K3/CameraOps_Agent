import pytest

from src.evaluation.benchmark import POLICIES, run_case
from src.evaluation.cases import make_case
from src.evaluation.reporting import summarize
from src.memory.experiment_memory import ExperimentMemory
from src.storage.artifact_store import ArtifactStore


def test_paired_benchmark_compares_five_policies_and_measures_actual_repair(tmp_path):
    case = make_case(ArtifactStore(tmp_path / "burst"), "registration", 0, 46051)
    frozen = ExperimentMemory(tmp_path / "held_out_memory.sqlite3")
    rows = [run_case(case, policy, tmp_path, frozen) for policy in POLICIES]
    result = {row["policy"]: row for row in rows}
    assert all(row["execution"]["status"] != "failed" for row in rows)
    assert all(row["case"]["input_id"] == case.input_id for row in rows)
    assert not result["fixed"]["tool_correct"]
    assert not result["multi"]["success"] and not result["multi"]["confirmed_faults"]
    assert result["feedback"]["success"] and result["feedback"]["translation_error_px"] < 1
    assert result["feedback"]["confirmed_faults"] == ["registration_failure"]
    assert result["memory"]["measurements"]["psnr"] == pytest.approx(result["feedback"]["measurements"]["psnr"])
    assert not frozen.history()
    summary = summarize(rows)
    assert summary[3]["validated_repairs"] == 1 and summary[3]["ssim_n"] == 1


def test_bayer_wrong_metadata_is_a_failure_even_when_graph_scores_an_output(tmp_path):
    case = make_case(ArtifactStore(tmp_path / "burst"), "bayer", 2, 501)
    result = run_case(case, "feedback", tmp_path)
    assert not result["success"] and not result["goal_pass"]
    assert result["measurements"]["psnr"] is not None
    assert "bayer_mismatch" not in result["confirmed_faults"]
