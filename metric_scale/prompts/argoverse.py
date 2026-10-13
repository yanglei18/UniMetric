"""Argoverse's historical VLM prompts, made explicit without changing them.

The original Argoverse runs had no ``argoverse`` prompt module. The prompt
loader therefore selected ``metric_scale.prompts.default``. Keep that exact
driving-scene prompt behavior here instead of introducing new anchor priors.
The original default.py SHA-256 was
88dce8eacb1e48497db44280debcdd3d2f68e3da2930191697fa2877957f6ab5.
"""

from . import default as _legacy_default

DYNAMIC_HEIGHT_ANCHORS_PROMPT = _legacy_default.DYNAMIC_HEIGHT_ANCHORS_PROMPT
STATIC_HEIGHT_ANCHORS_PROMPT = _legacy_default.STATIC_HEIGHT_ANCHORS_PROMPT
SCENE_PLANES_PROMPT = _legacy_default.SCENE_PLANES_PROMPT


def get_prompt(task_name: str) -> str:
    """Return the same prompt text used by the legacy default fallback."""
    return _legacy_default.get_prompt(task_name)


__all__ = (
    "DYNAMIC_HEIGHT_ANCHORS_PROMPT",
    "STATIC_HEIGHT_ANCHORS_PROMPT",
    "SCENE_PLANES_PROMPT",
    "get_prompt",
)
