"""Concrete tools called by the LangGraph agents."""

from src.camera.capture_simulator import SimulationConfig, simulate_capture
from src.computational.pipeline import run_pipeline
from src.computational.sequence import import_file_sequence, read_burst, select_frames
from src.debugging.diagnosis import Diagnosis, validate_fault
from src.iq.evaluation import IQResult, evaluate_image
from src.scene.analysis import SceneFeatures, describe_burst, describe_scene
from src.scene.synthetic import SyntheticScene
from src.storage.artifact_store import ArtifactStore

from .contracts import (ArtifactInput, ArtifactOutput, CaptureInput, PhotographyInput,
                        SurveyInput, SurveyOutput, ValidationInput)


def survey(args: SurveyInput) -> SurveyOutput:
    store = ArtifactStore(args.artifact_root / "burst")
    source = args.source
    if source.simulation is not None:
        config = source.simulation
        scene = SyntheticScene(config.scene)
        scale = config.sensor.photon_rate_e_s / config.sensor.full_well_e * 0.02
        first, _ = scene.render(0)
        last, _ = scene.render(0.045)
        features = describe_scene(first * scale, last * scale, exposures=[], saturated=[], sharpness=[], align=False)
        return SurveyOutput(features=features, input_artifact_id=None)
    artifact_id = source.burst_id if source.burst_id else import_file_sequence(store, source.raw_manifest)
    return SurveyOutput(features=describe_burst(read_burst(store, artifact_id)), input_artifact_id=artifact_id)


def capture(args: CaptureInput) -> ArtifactOutput:
    settings = args.simulation.model_dump(mode="json")
    settings["capture_plan"] = args.plan.model_dump(mode="json")
    config = SimulationConfig.model_validate(settings)
    return ArtifactOutput(artifact_id=simulate_capture(config, ArtifactStore(args.artifact_root / "burst")))


def inspect_burst(args: ArtifactInput) -> SceneFeatures:
    return describe_burst(read_burst(ArtifactStore(args.artifact_root / "burst"), args.artifact_id))


def run_photography(args: PhotographyInput) -> ArtifactOutput:
    bursts = ArtifactStore(args.artifact_root / "burst")
    sequence = read_burst(bursts, args.artifact_id)
    actual = tuple((frame.exposure_s, frame.analog_gain) for frame in sequence.exposures)
    if actual != tuple(zip(args.plan.exposure_times, args.plan.gains)):
        raise ValueError("CapturePlan exposure metadata does not match the input Burst")
    selected = args.plan.selected_frames or tuple(range(len(actual)))
    sequence = select_frames(sequence, selected, args.plan.reference_frame)
    return ArtifactOutput(artifact_id=run_pipeline(sequence, args.config, ArtifactStore(args.artifact_root / "photography"),
                                                   truth_store=bursts))


def evaluate(args: ArtifactInput) -> IQResult:
    return evaluate_image(ArtifactStore(args.artifact_root / "photography"),
                          ArtifactStore(args.artifact_root / "burst"), args.artifact_id)


def validate(args: ValidationInput) -> Diagnosis:
    return validate_fault(args.fault, evaluate(args), args.limits)
