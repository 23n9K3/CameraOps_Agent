"""Execute Optuna trials against one fixed RAW Burst and retain each result."""

import csv
import io
from math import isfinite
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import optuna
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, Field

from src.computational.pipeline import PipelineConfig, run_pipeline
from src.computational.sequence import read_burst
from src.memory.experiment_memory import Experiment, ExperimentMemory
from src.scene.analysis import describe_burst
from src.storage.artifact_store import ArtifactStore

from .objectives import OBJECTIVES, ObjectiveTarget

RANGES = {"denoise_strength": (0.25, 1.0), "motion_threshold": (0.02, 0.18),
          "fusion_weight": (0.35, 1.0), "exposure_weight": (0.0, 1.0), "tone_mapping_strength": (0.65, 1.35)}


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=71, ge=0, le=2**32 - 1)
    trials: int = Field(default=18, ge=1, le=250)


class TrialFailed(RuntimeError):
    """A recorded image trial could not yield a complete objective vector."""


def pareto_figure(rows: list[dict], active: list[str], pareto: set[int], selected: int) -> go.Figure:
    figure = go.Figure()
    x, y = active[:2]
    groups = {"Evaluated trials": [], "Pareto front": [], "Baseline": [], "Selected": []}
    for row in rows:
        if row["status"] != "complete":
            continue
        number = row["number"]
        if number in pareto:
            groups["Pareto front"].append(row)
        else:
            groups["Evaluated trials"].append(row)
        if number == 0:
            groups["Baseline"].append(row)
        if number == selected:
            groups["Selected"].append(row)
    colors = ("#9ca8b6", "#146d68", "#ce7924", "#6c46a4")
    symbols = ("circle", "circle", "diamond-open", "star")
    for (label, items), color, symbol in zip(groups.items(), colors, symbols):
        figure.add_scatter(name=label, mode="markers", x=[r["metrics"][x] for r in items],
                           y=[r["metrics"][y] for r in items],
                           marker={"color": color, "size": 13 if label in ("Baseline", "Selected") else 9, "symbol": symbol},
                           text=[f'Trial {r["number"]}<br>' + "<br>".join(f'{key} {r["metrics"][key]:.6g}' for key in active)
                                 for r in items], hovertemplate="%{text}<extra></extra>")
    figure.update_layout(title="AI-CPL measured Pareto front", xaxis_title=x, yaxis_title=y,
                         template="plotly_white", margin={"t": 65, "l": 60, "r": 30, "b": 80},
                         annotations=[{"text": f"2D projection of {len(active)} minimized objectives",
                                       "xref": "paper", "yref": "paper", "x": 0, "y": -0.16, "showarrow": False}])
    return figure


def _csv(rows: list[dict], objectives: list[str], pareto: set[int], selected: int) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["trial", "status", "artifact_id", "runtime_ms", "pareto", "selected", *RANGES, *objectives, "psnr_db", "error"])
    for row in rows:
        writer.writerow([row["number"], row["status"], row["output_artifact_id"], row["runtime_ms"], row["number"] in pareto,
                         row["number"] == selected, *[row["parameters"].get(key) for key in RANGES],
                         *[row["metrics"].get(key) for key in objectives], row["metrics"].get("psnr_db"), row["error"]])
    return output.getvalue().encode("utf-8-sig")


def optimize_burst(root: str | Path, burst_id: str, baseline: PipelineConfig, config: OptimizationConfig) -> str:
    root = Path(root).resolve()
    bursts, photos = ArtifactStore(root / "burst"), ArtifactStore(root / "photography")
    source = bursts.load_metadata(burst_id)
    if source.get("artifact_type") != "synthetic_raw_burst":
        raise ValueError("optimization requires a P2 Burst with calibrated, time-matched Ground Truth")
    sequence = read_burst(bursts, burst_id)
    reference = sequence.reference_frame if baseline.reference_frame is None else baseline.reference_frame
    if not 0 <= reference < len(sequence.mosaics):
        raise ValueError("baseline reference frame is outside the Burst")
    simulation = source["simulation_config"]
    sensor = simulation["sensor"]
    factor = sensor["photon_rate_e_s"] * sequence.exposures[reference].dose / sensor["full_well_e"]
    exposure = baseline.tone_exposure or {"night": 4.0, "hdr": 2.0, "motion": 2.5}[baseline.mode]
    target = ObjectiveTarget.prepare(bursts, burst_id, reference, factor, exposure, baseline.tone_mapping_strength)
    active = target.active
    if len(active) < 2:
        raise ValueError("the Burst has insufficient support for multi-objective comparison")
    scene = describe_burst(sequence).model_dump(mode="json")
    memory = ExperimentMemory(root / "experiments.sqlite3")
    study_name = f"aicpl-{baseline.mode}-{uuid4().hex}"
    optuna_db = root / "optuna.sqlite3"
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{optuna_db.as_posix()}",
                                directions=["minimize"] * len(active),
                                sampler=optuna.samplers.NSGAIISampler(seed=config.seed, population_size=6))
    preset = baseline.model_dump(mode="json")
    preset["reference_frame"] = reference
    if preset["exposure_weight"] is None:
        preset["exposure_weight"] = float(baseline.mode == "hdr")
    default_params = {key: preset[key] for key in RANGES}
    ranges = {key: (min(low, default_params[key]), max(high, default_params[key])) for key, (low, high) in RANGES.items()}
    study.set_user_attr("objectives", active)
    study.set_user_attr("burst_id", burst_id)
    study.set_user_attr("seed", config.seed)
    study.enqueue_trial(default_params)
    study_context = {"study_name": study_name, "sampler": "NSGAII", "optimizer_seed": config.seed,
                     "simulation": simulation, "objective_names": active, "search_ranges": ranges}

    def trial_run(trial):
        params = {key: trial.suggest_float(key, low, high) for key, (low, high) in ranges.items()}
        being_tested = PipelineConfig.model_validate(preset | params)
        started = perf_counter()
        output_id, error, metrics = None, None, {}
        try:
            output_id = run_pipeline(sequence, being_tested, photos, truth_store=bursts)
            final = photos.load_array(output_id, "final_linear", mmap_mode="r")
            tone_mapped = photos.load_array(output_id, "tone_mapped_rgb", mmap_mode="r")
            metrics = target.score(final, tone_mapped)
            if set(metrics) != set(active) or any(not isfinite(value) for value in metrics.values()):
                raise ValueError("trial returned unavailable or non-finite objective measurements")
            result = photos.load_metadata(output_id)["evaluation"]["scores"]["final_fusion"]["full_frame"]
            metrics["psnr_db"] = result["psnr_db"]
        except (OSError, ValueError, OverflowError, cv2.error) as exc:
            error = f"{type(exc).__name__}: {exc}"
            metrics = {}
        runtime = (perf_counter() - started) * 1000
        memory.record(Experiment(group_id=study_name, number=trial.number, kind="optimization", mode=baseline.mode,
                                 source_kind="synthetic_burst", status="failed" if error else "complete",
                                 input_artifact_id=burst_id, output_artifact_id=output_id, seed=simulation["seed"],
                                 runtime_ms=runtime, scene=scene, capture_plan=simulation["capture_plan"],
                                 parameters=being_tested.model_dump(mode="json"), metrics=metrics,
                                 configuration=study_context | {"baseline": trial.number == 0}, error=error))
        trial.set_user_attr("artifact_id", output_id)
        trial.set_user_attr("runtime_ms", runtime)
        if error:
            trial.set_user_attr("error", error)
            raise TrialFailed(error)
        return tuple(metrics[name] for name in active)

    study.optimize(trial_run, n_trials=config.trials + 1, catch=(TrialFailed,))
    rows = sorted(memory.history(group_id=study_name, limit=config.trials + 1), key=lambda row: row["number"])
    index = {row["number"]: row for row in rows}
    before = index[0]
    if before["status"] != "complete":
        raise TrialFailed(f"baseline failed in {study_name}: {before['error']}")
    front = sorted(study.best_trials, key=lambda trial: trial.number)
    baseline_values = [before["metrics"][name] for name in active]
    # Normalize changes by measured baseline scale with a small floor for near-zero objectives.
    def selection_rank(trial):
        return (sum((after - before) / max(abs(before), 1e-6) for before, after in zip(baseline_values, trial.values)),
                trial.number)

    chosen = min(front, key=selection_rank)
    after = index[chosen.number]
    pareto = {trial.number for trial in front}
    comparison = {name: {"baseline": before["metrics"][name], "selected": after["metrics"][name],
                         "delta": after["metrics"][name] - before["metrics"][name]} for name in active}
    preview = (photos.path(after["output_artifact_id"]) / "preview.png").read_bytes()
    return ArtifactStore(root / "optimization").save({}, {
        "artifact_type": "optuna_optimization_report", "schema_version": 1, "study_name": study_name,
        "storage": str(optuna_db), "experiment_memory": str(memory.path), "input_artifact_id": burst_id,
        "optimizer": config.model_dump(mode="json"), "raw_seed": simulation["seed"], "mode": baseline.mode,
        "reference_frame": reference, "fixed_tone_target": {"exposure": exposure, "strength": baseline.tone_mapping_strength},
        "objectives": {name: {"method": OBJECTIVES[name], "active": name in active,
                               "pixels": int(target.masks[name].sum()), "direction": "minimize"} for name in OBJECTIVES},
        "search_ranges": ranges, "trial_count": len(rows), "completed_trials": sum(row["status"] == "complete" for row in rows),
        "pareto_front": [index[trial.number] for trial in front], "baseline": before, "selected": after,
        "selection_rule": "Smallest sum of changes scaled by baseline objective values across Pareto trials. Ties favor earlier trials.",
        "objective_comparison": comparison, "trials": rows,
    }, preview_png=preview, attachments={"pareto.html": pareto_figure(rows, active, pareto, after["number"]).to_html(
                                             full_html=True, include_plotlyjs=True).encode("utf-8"),
                                         "trials.csv": _csv(rows, active, pareto, after["number"])})
