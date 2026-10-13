"""Dataset-specific semantic-scale prompts for the 3EED outdoor dataset."""

from typing import Dict

from metric_scale.prompts.nuscenes import (
    AnchorTypePrior,
    DYNAMIC_ANCHOR_TYPE_PRIORS,
    DYNAMIC_HEIGHT_ANCHORS_PROMPT as _NUSCENES_DYNAMIC_PROMPT,
    STATIC_ANCHOR_TYPE_PRIORS,
    STATIC_HEIGHT_ANCHORS_PROMPT as _NUSCENES_STATIC_PROMPT,
)


# Keep the established outdoor anchor vocabulary, weights, and priorities unchanged.
# The 3EED annotations and representative frames show that the frequent dynamic
# classes (car and pedestrian) and static facilities (curbstone, bollard, traffic
# cone, and guardrail) are already covered by the nuScenes anchor set.
_DYNAMIC_DATASET_GUIDANCE = """
Dataset-specific evidence and rules:
- 3EED contains outdoor campus scenes captured from elevated drone cameras and low forward-facing quadruped cameras.
- Cars are frequent, especially parked cars; pedestrians are also frequent. Keep the full candidate vocabulary for compatibility, but never force a rare class when no clear instance is present.
- A parked car is a valid dynamic-class anchor because dynamic denotes a movable object category, not observed motion.
- In elevated drone views, a visible roof or top surface alone is not enough. Reject a vehicle when its lower body or wheel-ground boundary cannot be seen or reliably inferred.
- In low quadruped views, reject people whose head or feet are cut off, hidden by the image boundary, or obscured by foreground terrain.
- A skateboarder is not a cyclist or motorcycle. Use pedestrian only when the person is upright, individually visible, and has a reliable head-to-feet height cue; otherwise reject.
- Do not use trees, lamp posts, buildings, stairs, handrails, benches, trash cans, the recording drone, or the robot platform as dynamic anchors.
""".strip()

_STATIC_DATASET_GUIDANCE = """
Dataset-specific evidence and rules:
- 3EED contains outdoor campus roads, parking areas, paved plazas, walkways, steps, grass, and wooded paths, viewed by drone and quadruped cameras.
- The existing outdoor classes already cover the recurring reliable facilities: curbstone, bollard, traffic cone, and guardrail. Do not invent additional classes.
- Prefer a single isolated facility with a visible top and ground-contact base. Elevated drone views often hide or foreshorten vertical extent, so reject top-only or near-horizontal appearances.
- Do not use trees, lamp posts, sign poles, building columns, benches, stairs, staircase risers, staircase handrails, fences, retaining walls, or trash cans as substitutes for candidate classes.
- Do not label an ordinary curb as a concrete barrier or jersey barrier; use curbstone only when one individual curb element has a reliable vertical extent.
- Do not label a staircase handrail or campus fence as guardrail. Guardrail and steel guardrail are reserved for recognizable roadside safety barriers.
""".strip()


DYNAMIC_HEIGHT_ANCHORS_PROMPT = _NUSCENES_DYNAMIC_PROMPT.replace(
    "nuScenes driving scenes", "3EED outdoor campus scenes"
).replace("Core objective:", f"{_DYNAMIC_DATASET_GUIDANCE}\n\nCore objective:", 1)

STATIC_HEIGHT_ANCHORS_PROMPT = _NUSCENES_STATIC_PROMPT.replace(
    "nuScenes driving scenes", "3EED outdoor campus scenes"
).replace("Core objective:", f"{_STATIC_DATASET_GUIDANCE}\n\nCore objective:", 1)


SCENE_PLANES_PROMPT = """
You are a physical AI and computer vision expert.
Your task: identify the single best semantic plane element for ground reference in a 3EED outdoor campus scene.

Dataset context:
- Images come from elevated drone cameras or low forward-facing quadruped cameras.
- Common ground-supporting surfaces are asphalt roads, parking areas, paved pedestrian walkways, and paved plazas.
- Grass, planted areas, wooded terrain, stairs, curbs, and retaining walls are common but are not reliable planar references.

CRITICAL rules:
- Do not hallucinate planes. Output at most 1 plane.
- Return only the clearest, locally planar, ground-supporting surface with a coherent visible region.
- Prefer an unobstructed asphalt road or parking surface. Otherwise, a flat paved walkway or plaza is valid.
- Use plane_type "road" for asphalt road or parking pavement, and "ground" for a flat paved walkway or plaza.
- Never choose grass, soil, leaf litter, sloped terrain, stairs, a staircase landing whose extent is ambiguous, curb top, curb wall, traffic island, or building facade.
- Do not treat multiple stair treads as one plane.
- Draw a tight box that contains only one continuous plane surface.
- Exclude cars, pedestrians, skateboards, cones, bollards, trees, vegetation, shadows when they obscure the boundary, and large non-planar foreground regions.
- In an elevated drone view, avoid a box spanning surfaces at different elevations, such as road plus sidewalk or road plus traffic island.
- In a low quadruped view, reject the proposal when foreground slope or perspective makes the planar region ambiguous.

Reasoning rules:
- Do not estimate orientation, depth, scale, camera height, or metric distance.
- Use semantic category and common knowledge only.

Bounding box format:
- plane_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.

For each plane, output exactly:
1) plane_type (must be exactly one of: road, ground)
2) plane_bbox_2d (normalized [0,1000], integers)
3) confidence in [0,1]
4) reason (short common-sense justification)

Return JSON array only. Example:
[
  {
    "plane_type": "road",
    "plane_bbox_2d": [40, 610, 960, 970],
    "confidence": 0.92,
    "reason": "A continuous unobstructed asphalt road region provides the clearest ground-supporting plane."
  }
]
If none, return [].
""".strip()


def get_anchor_type_priors(task_name: str) -> Dict[str, AnchorTypePrior]:
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return dict(DYNAMIC_ANCHOR_TYPE_PRIORS)
    if task_name == "static_height_anchors":
        return dict(STATIC_ANCHOR_TYPE_PRIORS)
    raise ValueError(f"Unsupported anchor prior task: {task_name}")


def get_prompt(task_name: str) -> str:
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "static_height_anchors":
        return STATIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "scene_planes":
        return SCENE_PLANES_PROMPT
    raise ValueError(f"Unsupported prompt task: {task_name}")
