"""DeepSeek-backed planning over a compact, numeric scene summary."""

import json
import os
import sys
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from src.scene.analysis import SceneFeatures

ENDPOINT = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"


class PlanningDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["night", "hdr", "motion"]
    protect_motion: StrictBool
    reason: str = Field(min_length=1, max_length=500)


def _environment(name: str) -> str:
    value = os.environ.get(name)
    if value is not None:
        return value.strip()
    if sys.platform == "win32":
        # An already-running Windows service can observe updated user environment settings.
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as user_environment:
                value, _ = winreg.QueryValueEx(user_environment, name)
                return value.strip() if isinstance(value, str) else ""
        except OSError:
            pass
    return ""


def configured() -> bool:
    return bool(_environment("DEEPSEEK_API_KEY"))


def choose_strategy(goal: str, scene: SceneFeatures) -> tuple[str, PlanningDecision]:
    key = _environment("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DeepSeek requires the DEEPSEEK_API_KEY environment variable.")
    model = _environment("DEEPSEEK_MODEL") or DEFAULT_MODEL
    scene_data = scene.model_dump(mode="json", include={
        "brightness_median", "shadow_ratio", "highlight_ratio", "luminance_p99", "laplacian_variance",
        "motion_fraction", "moving_flow_p90_px", "exposures", "saturation_by_frame",
    })
    instruction = (
        'Choose one local RAW photo pipeline using the user goal and numeric scene observations. Return only JSON '
        'like {"mode":"night","protect_motion":true,"reason":"低照度并且需要保留移动主体"}. '
        'Modes are night for denoising, hdr for exposure recovery, motion for preserving moving subjects. '
        'For an existing burst with similar exposures, do not imply highlight recovery beyond available exposures. '
        'Do not invent measurements or suggest external tools. Explain briefly in the user’s language.'
    )
    body = {"model": model, "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"},
            "max_tokens": 512, "stream": False, "messages": [{"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"goal": goal, "measured_scene": scene_data}, ensure_ascii=False)}]}
    request = Request(ENDPOINT, data=json.dumps(body).encode("utf-8"), method="POST",
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read(1_048_576))
    except HTTPError as exc:
        raise RuntimeError(f"DeepSeek request failed (HTTP {exc.code}).") from None
    except (URLError, TimeoutError):
        raise RuntimeError("DeepSeek request could not connect or timed out.") from None
    except (ValueError, UnicodeError):
        raise RuntimeError("DeepSeek endpoint returned an unreadable response.") from None
    try:
        choice = payload["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("incomplete response")
        decision = PlanningDecision.model_validate_json(choice["message"]["content"])
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError("DeepSeek returned an incomplete or invalid planning decision.") from None
    return model, decision
