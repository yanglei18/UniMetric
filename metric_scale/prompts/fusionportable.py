from typing import Dict, Mapping, TypedDict


class AnchorTypePrior(TypedDict):
    level: str
    weight: float
    priority: int


ANCHOR_TYPE_WEIGHT_BY_LEVEL: Dict[str, float] = {
    "very_high": 0.95,
    "high": 0.85,
    "medium": 0.70,
    "low": 0.55,
}


# FusionPortable contains only six fixed routes. These priors intentionally list
# objects that recur in those routes instead of asking the VLM to discover types.
DYNAMIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "pedestrian": {"level": "very_high", "weight": 0.95, "priority": 1},
    "person": {"level": "high", "weight": 0.85, "priority": 2},
    "sedan": {"level": "high", "weight": 0.85, "priority": 3},
    "suv": {"level": "medium", "weight": 0.70, "priority": 4},
    "van": {"level": "medium", "weight": 0.70, "priority": 5},
}


STATIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "door": {"level": "very_high", "weight": 0.95, "priority": 1},
    "cabinet": {"level": "high", "weight": 0.85, "priority": 2},
    "table": {"level": "high", "weight": 0.85, "priority": 3},
    "chair": {"level": "medium", "weight": 0.70, "priority": 4},
    "trash can": {"level": "medium", "weight": 0.70, "priority": 5},
    "traffic cone": {"level": "medium", "weight": 0.70, "priority": 6},
    "bollard": {"level": "low", "weight": 0.55, "priority": 7},
}


def _format_prior_table(priors: Mapping[str, AnchorTypePrior]) -> str:
    lines = []
    for class_name, prior in sorted(priors.items(), key=lambda item: item[1]["priority"]):
        lines.append(
            f'  "{class_name}": {{"level": "{prior["level"]}", '
            f'"type_weight": {prior["weight"]:.2f}, "priority": {prior["priority"]}}}'
        )
    return "\n".join(lines)


def _format_priority_order(priors: Mapping[str, AnchorTypePrior]) -> str:
    return " > ".join(
        class_name
        for class_name, _ in sorted(priors.items(), key=lambda item: item[1]["priority"])
    )


DYNAMIC_PRIOR_TABLE = _format_prior_table(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_PRIOR_TABLE = _format_prior_table(STATIC_ANCHOR_TYPE_PRIORS)
DYNAMIC_PRIORITY_ORDER = _format_priority_order(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_PRIORITY_ORDER = _format_priority_order(STATIC_ANCHOR_TYPE_PRIORS)


DYNAMIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Identify high-quality dynamic height anchors in the fixed six-sequence FusionPortable dataset.

Dataset evidence and scope:
- handheld_escalator00: a campus building, escalators, occasional nearby people, and occasional service robots.
- handheld_room00: a dense office with many people, desks, chairs, cabinets, and doors.
- legged_grass00: mostly grass, trees, roads, and distant buildings; reliable object anchors are usually absent.
- legged_room00: an indoor lab with people, cabinets, tables, chairs, and mobile robots.
- ugv_campus00: campus roads and walkways with occasional people and parked road vehicles.
- ugv_parking00: a mostly empty parking lot with road vehicles usually far away.
- The candidate classes below are fixed for these routes. Do not discover or invent another class.

Selection rules:
- Return at most 2 anchors. Prefer one excellent anchor over several weak anchors.
- Priority when equally clear: {DYNAMIC_PRIORITY_ORDER}.
- Use pedestrian for a clearly standing or normally walking adult. Use person only when the human is clear but pedestrian is uncertain.
- Use sedan, suv, or van only when that exact body type is visually clear. If vehicle type is uncertain, reject it; never output generic car.
- A person must have a complete head-to-feet height cue, both feet or the lower ground-contact boundary visible, and a clean separable silhouette.
- A vehicle must have its roofline and wheel-ground lower boundary visible or reliably inferable, with one clean body mask.
- Near and clean middle-distance instances are allowed. Reject distant instances, including tiny people across a plaza and small vehicles at the edge of the parking lot.
- Reject image-edge truncation, any top/bottom truncation, strong fisheye-edge distortion, blur, glare, reflection, or overlap that makes full height uncertain.
- Reject people sitting at office desks, upper-body-only people, crouching/bending people, people behind desks or chairs, and people standing on escalator steps.
- Reject vehicles that are heavily occluded, cropped, merged into a parked row, or visible only as a thin side fragment.
- A parked vehicle may still be a semantic height anchor; dynamic means movable object class, not that motion must be visible.
- Never select mobile robots, delivery robots, robot dogs, escalators, handrails, trees, lamp posts, buildings, bicycles without riders, or arbitrary machines. Their height is not stable enough for this prompt.
- In empty grass, road, walkway, or parking-lot frames, return [] instead of forcing an anchor.
- Do not use perspective, depth, pixels, camera height, or scene geometry to estimate metric height. Use only the fixed semantic intervals below.

Exact height intervals by class:
- pedestrian: [1.60, 1.82]
- person: [1.55, 1.85]
- sedan: [1.40, 1.55]
- suv: [1.60, 1.80]
- van: [1.80, 2.10]

Class priors (copy level, type_weight, and priority exactly):
{DYNAMIC_PRIOR_TABLE}

Bounding boxes:
- object_bbox_2d uses normalized integer coordinates [0,1000] as [x1,y1,x2,y2].
- Draw one tight box around one complete instance. Do not include neighbors or large empty regions.

For each anchor output exactly:
1) object_class
2) object_bbox_2d
3) height_interval
4) distance_level: near or middle
5) level
6) type_weight
7) priority
8) confidence in [0,1]
9) body_pose: standing, walking, sitting, crouching, kneeling, bending, lying, jumping, or unknown
10) full_body_visible
11) feet_visible
12) floor_contact_clear
13) reason
14) is_occluded
15) is_truncated

For vehicles, set body_pose to unknown. full_body_visible means the full vehicle height is reliable;
feet_visible and floor_contact_clear mean the wheel/lower-body ground contact is reliable.
Returned anchors must have full_body_visible=true, feet_visible=true,
floor_contact_clear=true, is_occluded=false, and is_truncated=false.

Return a JSON array only. Example:
[
  {{
    "object_class": "pedestrian",
    "object_bbox_2d": [330, 160, 510, 930],
    "height_interval": [1.60, 1.82],
    "distance_level": "near",
    "level": "very_high",
    "type_weight": 0.95,
    "priority": 1,
    "confidence": 0.94,
    "body_pose": "standing",
    "full_body_visible": true,
    "feet_visible": true,
    "floor_contact_clear": true,
    "reason": "Complete standing adult with clear head and floor contact.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If no candidate satisfies every rule, return [].
Candidate dynamic classes: pedestrian, person, sedan, suv, van.
""".strip()


STATIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Identify high-quality static height anchors in the fixed six-sequence FusionPortable dataset.

Dataset-specific candidate policy:
- The recurring indoor candidates are complete doors, cabinets, single tables, single chairs, and upright trash cans.
- The recurring outdoor candidates are standard traffic cones and permanent short bollards.
- These candidates are fixed from the six known routes. Do not discover or invent another category.
- legged_grass00 and ugv_parking00 often contain no usable static object. Returning [] is expected.

Selection rules:
- Return at most 2 anchors. Prefer one complete standardized object over several partial objects.
- Priority when equally clear: {STATIC_PRIORITY_ORDER}.
- Keep only a near or clean middle-distance single object with a reliable full top-to-base height and one clean SAM mask.
- Reject every distant, tiny, blurred, overexposed, reflected, fisheye-edge-distorted, cropped, or occluded candidate.
- The top and floor-contact base must both be visible. Returned anchors must have is_occluded=false and is_truncated=false.
- door: require one normal full-height door leaf or frame, visible header and bottom threshold. Reject glass-only openings, wide gates, elevator openings, and partial doorway strips.
- cabinet: require one freestanding or floor-standing cabinet with top and base visible. Reject wall cupboards, drawer fragments, and long merged cabinet rows.
- table: require one standard desk/table with tabletop and legs or pedestal visible. Reject cluttered desk rows, conference-table fragments, countertops, and tabletops with hidden legs.
- chair: require one empty isolated chair with back, seat, legs/base, and floor contact visible. Reject occupied, overlapping, cropped, or distant chairs.
- trash can: require one rigid upright bin with top rim/lid and bottom base visible. Reject wheeled service robots and orange autonomous machines.
- traffic cone: require one standard cone with tip and full base visible on pavement. Reject tiny distant cones and painted road markings.
- bollard: require one permanent short protective post with top and base visible. Reject lamp posts, sign poles, tree supports, railing posts, queue stanchions, and temporary tape posts.
- Never select escalators, escalator handrails, stairs, railings, trees, plants, lamp posts, building facades, fire boxes mounted on walls, mobile robots, road paint, curbs, or arbitrary lab machines.
- Do not use perspective, depth, pixels, camera height, or scene geometry to estimate height. Copy the fixed interval for the chosen class.

Exact height intervals by class:
- door: [2.00, 2.15]
- cabinet: [0.85, 1.10]
- table: [0.72, 0.78]
- chair: [0.80, 1.00]
- trash can: [0.65, 0.90]
- traffic cone: [0.65, 0.75]
- bollard: [0.75, 1.00]

Class priors (copy level, type_weight, and priority exactly):
{STATIC_PRIOR_TABLE}

Bounding boxes:
- object_bbox_2d uses normalized integer coordinates [0,1000] as [x1,y1,x2,y2].
- Draw one tight box around one complete physical instance.

For each anchor output exactly:
1) object_class
2) object_bbox_2d
3) height_interval
4) distance_level: near or middle
5) level
6) type_weight
7) priority
8) confidence in [0,1]
9) reason
10) is_occluded
11) is_truncated

Return a JSON array only. Example:
[
  {{
    "object_class": "door",
    "object_bbox_2d": [90, 120, 260, 940],
    "height_interval": [2.00, 2.15],
    "distance_level": "near",
    "level": "very_high",
    "type_weight": 0.95,
    "priority": 1,
    "confidence": 0.93,
    "reason": "Complete standard door with visible header and threshold.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If no candidate satisfies every rule, return [].
Candidate static classes: door, cabinet, table, chair, trash can, traffic cone, bollard.
""".strip()


SCENE_PLANES_PROMPT = """
You are a physical AI and computer vision expert.
Identify the single best nearby walkable plane in a FusionPortable frame.

- Return at most 1 plane.
- Use floor for a broad indoor floor or a flat outdoor tiled/concrete walkway.
- Use road for a broad paved road or parking surface.
- Use ground only for a broad, locally flat grass/soil surface in legged_grass00.
- Prefer the surface directly supporting the camera platform.
- Reject escalator steps, stairs, ramps, curbs, walls, tabletops, vegetation clumps, and distant surfaces.
- The box must mostly contain one coherent plane and exclude people, vehicles, furniture, and large obstacles.
- Do not estimate orientation, depth, or scale.
- plane_bbox_2d is normalized [0,1000] as [x1,y1,x2,y2].

Output exactly: plane_type, plane_bbox_2d, confidence, reason.
Return a JSON array only. Example:
[
  {
    "plane_type": "floor",
    "plane_bbox_2d": [0, 560, 1000, 1000],
    "confidence": 0.92,
    "reason": "Broad unobstructed indoor floor supporting the camera route."
  }
]
If no reliable plane exists, return [].
Allowed plane types: floor, road, ground.
""".strip()


def get_anchor_type_priors(task_name: str) -> Dict[str, AnchorTypePrior]:
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return dict(DYNAMIC_ANCHOR_TYPE_PRIORS)
    if task_name == "static_height_anchors":
        return dict(STATIC_ANCHOR_TYPE_PRIORS)
    raise ValueError(f"Unsupported anchor prior task: {task_name}")


def get_prompt(task_name: str, scene_domain: str = "indoor") -> str:
    # FusionPortable mixes indoor rooms, campus walkways, grass, and parking lots.
    # The dataset-specific prompt is authoritative regardless of the shared config value.
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "static_height_anchors":
        return STATIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "scene_planes":
        return SCENE_PLANES_PROMPT
    raise ValueError(f"Unsupported prompt task: {task_name}")
